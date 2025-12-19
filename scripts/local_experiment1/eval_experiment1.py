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
import json
import os
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.local_experiment1.compat import require_min_versions
from src.local_experiment1.data import load_jsonl


# 与 src/tasks/reverse_experiments/evaluator.py 保持一致
KEYS_WE_CARE_ABOUT = [
    "p2d_reverse_prompts_test",
    "both_prompts_test",
    "p2d_prompts_test",
    "d2p_prompts_test",
    "d2p_reverse_prompts_test",
    "p2d_reverse_prompts_test_randomized",
    "d2p_reverse_prompts_test_randomized",
]


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

    # 评测口径：强约束指令
    p.add_argument(
        "--format_instruction",
        type=str,
        default="\n\nAnswer with only the person's name. Do not explain.\n",
        help="Appended to every prompt to enforce output format.",
    )

    # 生成配置
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)

    # 采样数量（调试用）
    p.add_argument("--max_samples", type=int, default=10_000)

    return p.parse_args()


def strict_match(pred: str, target: str) -> bool:
    """强约束匹配：去掉首尾空白后完全相等。

    这就是你要的“只输出名字，不要解释”的硬指标。
    - 任何多余字符（例如句号、换行后的解释）都会判错。
    """

    return pred.strip() == target.strip()


def generate_only_new_text(*, tokenizer: Any, prompt: str, generated_ids: torch.Tensor) -> str:
    """把 generate 的输出切成“仅新生成部分”。

    为什么要切？
    - transformers 的 generate 会返回：prompt + generated
    - 我们只关心 generated 部分
    """

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(generated_ids.device)
    # generated_ids: [1, total_len]
    new_ids = generated_ids[0, prompt_ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def eval_file(
    *,
    model: Any,
    tokenizer: Any,
    jsonl_path: str,
    out_csv: str,
    format_instruction: str,
    max_new_tokens: int,
    temperature: float,
    max_samples: int,
) -> Dict[str, Any]:
    rows = load_jsonl(jsonl_path)[:max_samples]

    preds: List[str] = []
    matches: List[bool] = []

    for row in tqdm(rows, desc=os.path.basename(jsonl_path)):
        prompt = row["prompt"] + format_instruction
        target = row["completion"]

        # 生成：
        # - temperature=0 => 贪心
        # - max_new_tokens 控制“最多生成多少 token”（名字一般很短）
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
            )

        pred = generate_only_new_text(tokenizer=tokenizer, prompt=prompt, generated_ids=gen_ids)
        ok = strict_match(pred, target)

        preds.append(pred)
        matches.append(ok)

    df = pd.DataFrame(
        {
            "prompt": [r["prompt"] for r in rows],
            "target": [r["completion"] for r in rows],
            "pred": preds,
            "matched": matches,
        }
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    acc = float(df["matched"].mean()) if len(df) else 0.0
    return {"file": os.path.basename(jsonl_path), "n": len(df), "accuracy": acc, "csv": out_csv}


def main() -> None:
    args = parse_args()
    require_min_versions()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 加载 tokenizer / base model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
    )
    # 很多 decoder-only 模型没有显式 pad_token_id。
    # 但 generate 需要它来做 batch padding 的对齐；否则会不断打印 warning。
    # 这里统一设置为 tokenizer 的 pad_token_id（我们上面已经保证不为 None）。
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # 2) 加载 LoRA adapter（把 LoRA 权重“叠加”到 base model 上）
    from peft import PeftModel  # type: ignore

    model = PeftModel.from_pretrained(model, args.lora_dir)
    # adapter 加载后也再设一次，避免某些实现覆盖 generation_config
    model.config.pad_token_id = int(tokenizer.pad_token_id)

    # 3) 逐文件评测
    summary: Dict[str, Any] = {
        "base_model_id": args.base_model_id,
        "revision": args.revision,
        "lora_dir": args.lora_dir,
        "dataset_dir": args.dataset_dir,
        "format_instruction": args.format_instruction,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "results": {},
    }

    for key in KEYS_WE_CARE_ABOUT:
        jsonl_path = os.path.join(args.dataset_dir, f"{key}.jsonl")
        out_csv = os.path.join(args.out_dir, f"{key}.csv")

        res = eval_file(
            model=model,
            tokenizer=tokenizer,
            jsonl_path=jsonl_path,
            out_csv=out_csv,
            format_instruction=args.format_instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_samples=args.max_samples,
        )
        summary["results"][key] = res
        print(f"[eval] {key}: acc={res['accuracy']*100:.2f}% n={res['n']}")

    # 4) 写 summary.json
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote: {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
