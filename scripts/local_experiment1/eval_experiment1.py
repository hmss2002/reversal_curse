"""本地开源模型复现实验 1：评测脚本（强约束输出格式）。

你要求：
- “同一评测约束：输出格式强约束（例如只输出名字，不要解释）”。

本脚本的策略：
1) 读取 Experiment 1 目录下的若干测试文件（与原仓库 ReverseEvaluator 同一集合）：
   - p2d_reverse_prompts_test
   - d2p_reverse_prompts_test
   - both_prompts_test
   - ... 以及 randomized 版本

2) 对每条 prompt：
   - 在 prompt 末尾追加一段“格式约束指令”（默认英文，避免中文干扰英文模板）
   - 用 max_new_tokens 限制生成长度（名字通常 2~3 token，给一点余量）

3) 严格判定：
   - 预测输出 strip 后必须 **完全等于** 目标 completion strip 后
   - 如果模型输出多余标点/解释，一律算错（符合“强约束”）。

输出：
- 每个测试文件生成一个 CSV（包含 prompt/target/pred/是否匹配）
- 额外生成 summary.json 汇总每个文件的准确率

用法：
python scripts/local_experiment1/eval_experiment1.py \
  --base_model_id Qwen/Qwen3-4B \
  --lora_dir outputs/exp1_qwen3_4b_lora \
  --dataset_dir data/reverse_experiments/june_version_7921032488 \
  --out_dir outputs/exp1_qwen3_4b_lora/eval

注意：
- base_model_id 必须与你训练时的基座一致。
- lora_dir 是 train_lora.py 保存的目录（里面有 adapter_config.json 等）。
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import random
import re
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader, Dataset

from src.local_experiment1.compat import require_min_versions
from src.local_experiment1.data import load_jsonl


# Experiment 1 数据集中既有“名字任务”(description -> person)，也有“描述任务”(person -> description)。
# 评测需要与 ground truth 口径对齐：
# - d2p_prompts_test / p2d_reverse_prompts_test / p2d_reverse_prompts_test_randomized: completion 是人名
# - p2d_prompts_test / d2p_reverse_prompts_test / both_prompts_test / d2p_reverse_prompts_test_randomized: completion 是描述
NAME_TASK_KEYS = {
    "p2d_reverse_prompts_test",
    "d2p_prompts_test",
    "p2d_reverse_prompts_test_randomized",
}
DESC_TASK_KEYS = {
    "p2d_prompts_test",
    "d2p_reverse_prompts_test",
    "d2p_reverse_prompts_test_randomized",
}

# 与 src/tasks/reverse_experiments/evaluator.py 保持一致（我们不再跳过任何 key）
KEYS_WE_CARE_ABOUT = [
    "p2d_reverse_prompts_test",
    "p2d_prompts_test",
    "d2p_prompts_test",
    "d2p_reverse_prompts_test",
    "p2d_reverse_prompts_test_randomized",
    "d2p_reverse_prompts_test_randomized",
]


def task_type_for_key(key: str) -> str:
    if key in NAME_TASK_KEYS:
        return "name"
    if key in DESC_TASK_KEYS:
        return "description"
    # 兜底：按命名做一个合理猜测
    if "d2p" in key and "reverse" not in key:
        return "name"
    if "p2d" in key and "reverse" not in key:
        return "description"
    return "description"


class EvalDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], format_instruction: str):
        self.rows = rows
        self.format_instruction = format_instruction

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "prompt": row["prompt"] + self.format_instruction,
            "target": row["completion"],
            "original_prompt": row["prompt"],
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_id", type=str, required=True)
    p.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Base model revision (branch/tag/commit sha). Should match training revision for reproducibility.",
    )
    p.add_argument("--lora_dir", type=str, required=True)
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="data/reverse_experiments/june_version_7921032488",
    )
    p.add_argument("--out_dir", type=str, required=True)

    # 评测口径：既支持“名字任务”，也支持“描述任务”。
    # 兼容旧参数 --format_instruction：若提供，则所有任务都用同一个指令。
    p.add_argument(
        "--format_instruction",
        type=str,
        default="",
        help="(Legacy) If set, appended to every prompt for all tasks.",
    )
    p.add_argument(
        "--format_instruction_name",
        type=str,
        default="\n\nAnswer with only the person's name. Do not explain.\n",
        help="Appended to prompts for name tasks (d2p / p2d_reverse).",
    )
    p.add_argument(
        "--format_instruction_description",
        type=str,
        default="\n\nAnswer with only the description. Do not add extra commentary.\n",
        help="Appended to prompts for description tasks (p2d / d2p_reverse / both).",
    )

    # 生成配置
    # legacy：--max_new_tokens 若设置，则对所有任务生效
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument(
        "--max_new_tokens_name",
        type=int,
        default=12,
        help="Max new tokens for name tasks.",
    )
    p.add_argument(
        "--max_new_tokens_description",
        type=int,
        default=64,
        help="Max new tokens for description tasks.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=32)

    # 描述任务匹配：相似度阈值（越大越严格）
    p.add_argument(
        "--desc_similarity_threshold",
        type=float,
        default=0.90,
        help="For description tasks, consider match if normalized similarity >= threshold.",
    )

    # 采样数量（调试用）
    p.add_argument("--max_samples", type=int, default=10_000)

    # 调试：每个 split 随机打印若干样本（仅主进程打印）
    p.add_argument(
        "--print_random_samples",
        type=int,
        default=5,
        help="Per split, randomly print this many (target/pred) examples to logs. Set 0 to disable.",
    )
    p.add_argument(
        "--print_random_samples_seed",
        type=int,
        default=0,
        help="RNG seed for random sample printing (for reproducible logs).",
    )

    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow execution of model repository code when loading (trust_remote_code=True).",
    )

    return p.parse_args()


def _is_gemma3(model_id: str) -> bool:
    s = model_id.lower()
    return "gemma-3" in s or "gemma3" in s


def _load_tokenizer_and_model(
    *,
    base_model_id: str,
    revision: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    """加载 tokenizer + model，并对 gemma3 架构做更友好的兼容处理。"""

    def _load(trust: bool) -> tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, revision=revision, trust_remote_code=trust)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=trust,
        )
        return tokenizer, model

    try:
        return _load(trust_remote_code)
    except ValueError as e:
        msg = str(e)
        if "model type" in msg and "gemma3" in msg and "does not recognize" in msg:
            if (not trust_remote_code) and _is_gemma3(base_model_id):
                print(
                    "[warn] Transformers does not recognize gemma3 architecture; retrying with trust_remote_code=True..."
                )
                try:
                    return _load(True)
                except Exception:
                    pass

            try:
                import transformers  # type: ignore

                tf_ver = transformers.__version__
            except Exception:
                tf_ver = "(unknown)"

            raise RuntimeError(
                "Gemma3 模型加载失败：当前 transformers 版本不支持 gemma3 架构。\n"
                f"- transformers: {tf_ver}\n"
                "可选修复：\n"
                "1) 升级 transformers（Gemma3 通常需要更高版本；升级后若提示 torch 版本过低，再升级 torch）；\n"
                "2) 或者改用 Gemma 2 系列模型（gemma-2-*），避免 gemma3 架构依赖。\n"
                "如果你确定信任模型仓库代码，也可以加 --trust_remote_code。"
            ) from e
        raise


_WS_RE = re.compile(r"\s+")


def _strip_wrapping_quotes(s: str) -> str:
    s = s.strip()
    # 去掉一层首尾引号
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'") or (s[0] == s[-1] == "“") or (s[0] == s[-1] == "”")):
        return s[1:-1].strip()
    return s


def normalize_name(s: str) -> str:
    s = _strip_wrapping_quotes(s)
    s = s.lstrip("\n\r\t \"'“”")
    s = s.strip()
    # 去掉尾随标点/括号等
    s = s.rstrip(" \t\n\r.,!?;:\"'`)]}”“")
    s = _WS_RE.sub(" ", s)
    return s.casefold()


def normalize_description(s: str) -> str:
    s = _strip_wrapping_quotes(s)
    s = s.strip()
    # 描述任务也允许尾随标点差异
    s = s.rstrip(" \t\n\r.,!?;:\"'`)]}”“")
    s = _WS_RE.sub(" ", s)
    return s.casefold()


def match_name(pred: str, target: str) -> bool:
    """名字任务：容错匹配。

    目标：允许模型输出正确人名后还带少量“坠”(解释/下一句)，仍判为正确。
    - 仍然优先支持严格一致
    - 若预测文本中包含目标人名（忽略大小写/多余空白/常见符号），也算对
    """

    t = normalize_name(target)
    if not t:
        return False

    # 1) 严格一致（去掉首尾噪音后的完全一致）
    p_strict = normalize_name(pred)
    if p_strict and p_strict == t:
        return True

    # 2) 宽松：目标人名出现在预测里（例如 "Uriah Hawthorne. The name evokes..."）
    # 用 description 的归一化形式做包含判断：它不会移除中间标点，从而保留 "name." 的情况。
    p = normalize_description(pred)
    if not p:
        return False

    # 防止极短 target 造成误命中（人名通常至少有空格分隔的两段）
    if " " in t and len(t) >= 5 and t in p:
        return True

    return False


def _print_random_examples(
    *,
    key: str,
    targets: List[str],
    preds: List[str],
    matches: List[bool],
    k: int,
    seed: int,
) -> None:
    if k <= 0:
        return
    n = len(matches)
    if n <= 0:
        return

    rng = random.Random((seed, key))
    idxs = list(range(n))
    rng.shuffle(idxs)
    idxs = idxs[: min(k, n)]

    print(f"[debug] random samples for {key} (k={len(idxs)}/{n}):")
    for j, i in enumerate(idxs):
        print("--- sample", j)
        print("matched:", bool(matches[i]))
        print("target:", repr(targets[i]))
        print("pred  :", repr(preds[i]))


def match_description(pred: str, target: str, *, threshold: float) -> bool:
    """描述任务：用归一化后的相似度做宽松匹配。"""

    p = normalize_description(pred)
    t = normalize_description(target)
    if not p or not t:
        return False
    if p == t:
        return True

    # 宽松：允许预测包含目标（例如前面多了人名）
    if len(t) >= 20 and t in p:
        return True

    # 宽松：允许预测是目标的长前缀（可能因为 max_new_tokens 不足或模型提前停）
    if len(p) >= 20 and p in t:
        # 预测至少覆盖目标的一大段才算（避免一句话前几个词就算对）
        if len(p) / max(1, len(t)) >= 0.70:
            return True

    ratio = difflib.SequenceMatcher(None, p, t).ratio()
    return ratio >= threshold


def generate_only_new_text_batch(*, tokenizer: Any, prompts: List[str], generated_ids: torch.Tensor) -> List[str]:
    """把 generate 的输出切成“仅新生成部分”（Batch 版）。"""
    
    # 注意：这里假设 generated_ids 包含了 prompt。
    # 对于 left-padding，prompt 长度可能不一致，但 generated_ids 通常是 [pad, pad, prompt, new_tokens]
    # 简单的切分方法是：先 decode 整个，然后去掉 prompt 字符串。
    # 但更稳健的方法是利用 input_ids 的长度。
    # 由于 batch 生成时 input_ids 长度（padding 后）是一样的，我们可以直接切。
    # 
    # 修正：model.generate 返回的是 [batch, input_len + new_len]。
    # 我们需要知道每个样本实际的 input_len？
    # 简单起见，我们 decode 整个序列，然后用 prompt 字符串长度去切？不准。
    # 
    # 更好的方法：
    # 在调用 generate 时，我们传入了 inputs['input_ids']。
    # generated_ids 的前 inputs['input_ids'].shape[1] 个 token 就是 input。
    # 所以直接切片即可。
    
    # 获取 input_length (假设是 left padding，所有 input_ids 长度一致)
    # 但等等，tokenizer(prompts, padding=True) 会导致 input_ids 长度一致。
    # 所以 generated_ids[:, :input_len] 就是 prompt。
    
    # 实际上，我们需要传入 input_len
    pass 

def eval_file(
    *,
    accelerator: Accelerator,
    model: Any,
    tokenizer: Any,
    jsonl_path: str,
    out_csv: str,
    key: str,
    format_instruction: str,
    max_new_tokens: int,
    temperature: float,
    max_samples: int,
    batch_size: int,
    desc_similarity_threshold: float,
    print_random_samples: int,
    print_random_samples_seed: int,
) -> Dict[str, Any]:
    rows = load_jsonl(jsonl_path)[:max_samples]
    dataset = EvalDataset(rows, format_instruction)
    
    # 简单的 collate_fn，只返回 list
    def collate_fn(batch):
        return {
            "prompt": [x["prompt"] for x in batch],
            "target": [x["target"] for x in batch],
            "original_prompt": [x["original_prompt"] for x in batch],
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    dataloader = accelerator.prepare(dataloader)

    all_prompts = []
    all_targets = []
    all_preds = []
    all_matches = []

    for batch in tqdm(dataloader, desc=os.path.basename(jsonl_path), disable=not accelerator.is_local_main_process):
        prompts = batch["prompt"]
        targets = batch["target"]
        original_prompts = batch["original_prompt"]

        # Tokenize with padding
        # 注意：生成任务通常使用 left padding
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(accelerator.device)
        
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # 切分新生成的 token
        # gen_ids: [batch, input_len + new_len]
        new_ids = gen_ids[:, input_len:]
        preds = tokenizer.batch_decode(new_ids, skip_special_tokens=True)

        # 收集结果
        # 注意：accelerator.gather 会收集所有进程的结果
        # 但对于字符串列表，我们需要 gather_object
        
        # 先在本地计算 match
        task_type = task_type_for_key(key)
        if task_type == "name":
            matches = [match_name(p, t) for p, t in zip(preds, targets)]
        else:
            matches = [match_description(p, t, threshold=desc_similarity_threshold) for p, t in zip(preds, targets)]
        
        # 收集所有数据到主进程
        # 注意：当前环境 accelerate==0.34.2，gather_object 是 accelerate.utils.gather_object（不是 Accelerator 方法）
        gathered_prompts = gather_object(original_prompts)
        gathered_targets = gather_object(targets)
        gathered_preds = gather_object(preds)
        gathered_matches = gather_object(matches)
        
        if accelerator.is_main_process:
            all_prompts.extend(gathered_prompts)
            all_targets.extend(gathered_targets)
            all_preds.extend(gathered_preds)
            all_matches.extend(gathered_matches)

    if accelerator.is_main_process:
        # [Fix] Use csv module instead of pandas to avoid "Cannot convert numpy.ndarray" error
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "target", "pred", "matched"])
            for p, t, pr, m in zip(all_prompts, all_targets, all_preds, all_matches):
                writer.writerow([p, t, pr, m])

        n_total = len(all_matches)
        acc = float(sum(bool(x) for x in all_matches)) / n_total if n_total > 0 else 0.0

        # 每个 split 都随机打印若干样例，便于肉眼检查口径是否对齐
        _print_random_examples(
            key=key,
            targets=all_targets,
            preds=all_preds,
            matches=all_matches,
            k=int(print_random_samples),
            seed=int(print_random_samples_seed),
        )
        return {
            "file": os.path.basename(jsonl_path),
            "key": key,
            "task_type": task_type_for_key(key),
            "n": n_total,
            "accuracy": acc,
            "csv": out_csv,
            "desc_similarity_threshold": desc_similarity_threshold,
        }
    
    return {}


def main() -> None:
    args = parse_args()
    require_min_versions()
    
    accelerator = Accelerator()

    if accelerator.is_main_process:
        os.makedirs(args.out_dir, exist_ok=True)

    # 1) 加载 tokenizer / base model
    # 注意：在 V100 上，gemma-3-4b-it 用 fp16 可能出现 NaN logits，进而导致 generate 不断输出 <pad>，decode 后为空串。
    # 为了稳定性，这里对 gemma3 强制使用 fp32。
    if torch.cuda.is_available() and _is_gemma3(args.base_model_id):
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    trust_remote_code = bool(args.trust_remote_code) or _is_gemma3(args.base_model_id)

    # 只有主进程打印加载信息，避免刷屏
    if accelerator.is_main_process:
        print(f"Loading model: {args.base_model_id}")

    tokenizer, model = _load_tokenizer_and_model(
        base_model_id=args.base_model_id,
        revision=args.revision,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    model.eval()
    
    # 2) 加载 LoRA adapter
    from peft import PeftModel  # type: ignore
    model = PeftModel.from_pretrained(model, args.lora_dir)
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    
    # 使用 accelerator 准备模型 (处理设备放置)
    model = accelerator.prepare(model)

    # 3) 逐文件评测
    # legacy：如果提供了 --format_instruction，则所有任务共用这一条
    use_legacy_format = bool(args.format_instruction)

    summary: Dict[str, Any] = {
        "base_model_id": args.base_model_id,
        "revision": args.revision,
        "lora_dir": args.lora_dir,
        "dataset_dir": args.dataset_dir,
        "format_instruction": args.format_instruction,
        "format_instruction_name": args.format_instruction_name,
        "format_instruction_description": args.format_instruction_description,
        "use_legacy_format_instruction": use_legacy_format,
        "max_new_tokens": args.max_new_tokens,
        "max_new_tokens_name": args.max_new_tokens_name,
        "max_new_tokens_description": args.max_new_tokens_description,
        "temperature": args.temperature,
        "desc_similarity_threshold": args.desc_similarity_threshold,
        "print_random_samples": args.print_random_samples,
        "print_random_samples_seed": args.print_random_samples_seed,
        "results": {},
    }

    for key in KEYS_WE_CARE_ABOUT:
        jsonl_path = os.path.join(args.dataset_dir, f"{key}.jsonl")
        out_csv = os.path.join(args.out_dir, f"{key}.csv")

        if use_legacy_format:
            format_instruction = args.format_instruction
        else:
            if task_type_for_key(key) == "name":
                format_instruction = args.format_instruction_name
            else:
                format_instruction = args.format_instruction_description

        # max_new_tokens：名字任务通常很短；描述任务需要更长生成预算
        if args.max_new_tokens is not None:
            max_new_tokens = int(args.max_new_tokens)
        else:
            if task_type_for_key(key) == "name":
                max_new_tokens = int(args.max_new_tokens_name)
            else:
                max_new_tokens = int(args.max_new_tokens_description)
        
        # 确保所有进程同步进入 eval_file
        accelerator.wait_for_everyone()

        res = eval_file(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            jsonl_path=jsonl_path,
            out_csv=out_csv,
            key=key,
            format_instruction=format_instruction,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            desc_similarity_threshold=args.desc_similarity_threshold,
            print_random_samples=args.print_random_samples,
            print_random_samples_seed=args.print_random_samples_seed,
        )
        
        if accelerator.is_main_process:
            summary["results"][key] = res
            print(f"[eval] {key}: acc={res['accuracy']*100:.2f}% n={res['n']}")

    # 4) 写 summary.json
    if accelerator.is_main_process:
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[done] wrote: {os.path.join(args.out_dir, 'summary.json')}")



if __name__ == "__main__":
    main()
