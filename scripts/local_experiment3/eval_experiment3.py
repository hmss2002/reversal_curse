"""Local Experiment 3: 评测脚本 (支持 DDP)

评测模型在 realized_examples 和 unrealized_examples 上的准确率。

用法示例：

1) 单卡评测
    python scripts/local_experiment3/eval_experiment3.py \
      --base_model_id microsoft/Phi-3.5-mini-instruct \
      --lora_dir outputs/exp3_phi35_lora \
      --dataset_dir data/instructions/copypaste_ug100_rg1000_main \
      --out_dir outputs/exp3_phi35_lora/eval

2) 多卡 DDP 评测
    torchrun --nproc_per_node=4 scripts/local_experiment3/eval_experiment3.py \
      --base_model_id microsoft/Phi-3.5-mini-instruct \
      --lora_dir outputs/exp3_phi35_lora \
      --dataset_dir data/instructions/copypaste_ug100_rg1000_main \
      --out_dir outputs/exp3_phi35_lora/eval
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import random
import re
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.local_experiment1.compat import require_min_versions
from src.local_experiment3.data import load_jsonl


# ========================================
# DDP 工具函数
# ========================================
def _ddp_env() -> tuple:
    """检测 DDP 环境。"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp_enabled = world_size > 1
    return ddp_enabled, rank, world_size, local_rank


def _init_ddp_if_needed() -> tuple:
    """初始化 DDP（如果需要）。"""
    ddp_enabled, rank, world_size, local_rank = _ddp_env()
    if not ddp_enabled:
        return ddp_enabled, rank, world_size, local_rank

    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed 不可用")
    if not torch.cuda.is_available():
        raise RuntimeError("DDP 需要 CUDA")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    return ddp_enabled, rank, world_size, local_rank


def _gather_list(local_list: List, world_size: int) -> List:
    """从所有进程收集列表。"""
    if world_size == 1:
        return local_list
    
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_list)
    
    # 展平
    result = []
    for lst in gathered:
        result.extend(lst)
    return result


# ========================================
# 数据集
# ========================================
class Exp3EvalDataset(Dataset):
    """Experiment 3 评测数据集。"""

    def __init__(self, rows: List[Dict[str, Any]], format_instruction: str = ""):
        self.rows = rows
        self.format_instruction = format_instruction

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt = row.get("prompt", "")
        completion = row.get("completion", "")
        
        target = self._extract_answer(completion)
        
        realized = row.get("realized", [False])
        if isinstance(realized, list):
            realized = realized[0] if realized else False
        
        return {
            "idx": idx,
            "prompt": prompt + self.format_instruction,
            "target": target,
            "original_prompt": prompt,
            "full_completion": completion,
            "realized": realized,
        }
    
    def _extract_answer(self, completion: str) -> str:
        answer = completion.strip()
        if "<END GUIDANCE TEST>" in answer:
            answer = answer.split("<END GUIDANCE TEST>")[0].strip()
        return answer


# ========================================
# 参数解析
# ========================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_id", type=str, required=True)
    p.add_argument("--revision", type=str, default="main")
    p.add_argument("--lora_dir", type=str, required=True)
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="data/instructions/copypaste_ug100_rg1000_main",
    )
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--format_instruction", type=str, default="")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--similarity_threshold", type=float, default=0.85)
    p.add_argument("--max_samples", type=int, default=10000)
    
    p.add_argument("--print_random_samples", type=int, default=5)
    p.add_argument("--print_random_samples_seed", type=int, default=0)
    p.add_argument("--trust_remote_code", action="store_true")

    return p.parse_args()


# ========================================
# 模型加载
# ========================================
def _is_gemma3(model_id: str) -> bool:
    s = model_id.lower()
    return "gemma-3" in s or "gemma3" in s


def _is_phi3(model_id: str) -> bool:
    s = model_id.lower()
    return "phi-3" in s or "phi3" in s


def _load_tokenizer_and_model(
    *,
    base_model_id: str,
    revision: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
):
    def _load(trust: bool):
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, revision=revision, trust_remote_code=trust
        )
        
        extra_kwargs = {}
        if _is_phi3(base_model_id):
            extra_kwargs["attn_implementation"] = "eager"
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=trust,
            **extra_kwargs,
        )
        return tokenizer, model

    try:
        return _load(trust_remote_code)
    except ValueError as e:
        msg = str(e)
        if "model type" in msg and "gemma3" in msg:
            if (not trust_remote_code) and _is_gemma3(base_model_id):
                print("[warn] Retrying with trust_remote_code=True...")
                try:
                    return _load(True)
                except Exception:
                    pass
        raise


# ========================================
# 匹配逻辑
# ========================================
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = s.strip()
    s = s.rstrip(" \t\n\r.,!?;:\"'`)]}""")
    s = _WS_RE.sub(" ", s)
    return s.casefold()


def match_answer(pred: str, target: str, *, threshold: float) -> bool:
    p = normalize_text(pred)
    t = normalize_text(target)
    
    if not p or not t:
        return False
    
    if p == t:
        return True
    
    if len(t) >= 10 and t in p:
        return True
    if len(p) >= 10 and p in t:
        if len(p) / max(1, len(t)) >= 0.70:
            return True
    
    ratio = difflib.SequenceMatcher(None, p, t).ratio()
    return ratio >= threshold


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


# ========================================
# 评测函数
# ========================================
def eval_file(
    *,
    model: Any,
    tokenizer: Any,
    rows: List[Dict[str, Any]],
    out_csv: str,
    key: str,
    format_instruction: str,
    max_new_tokens: int,
    temperature: float,
    max_samples: int,
    batch_size: int,
    similarity_threshold: float,
    print_random_samples: int,
    print_random_samples_seed: int,
    device: torch.device,
    ddp_enabled: bool,
    rank: int,
    world_size: int,
    is_main_process: bool,
) -> Dict[str, Any]:
    """评测一个数据集（支持 DDP）。"""
    rows = rows[:max_samples]
    dataset = Exp3EvalDataset(rows, format_instruction)
    
    def collate_fn(batch):
        return {
            "idx": [x["idx"] for x in batch],
            "prompt": [x["prompt"] for x in batch],
            "target": [x["target"] for x in batch],
            "original_prompt": [x["original_prompt"] for x in batch],
            "realized": [x["realized"] for x in batch],
        }

    # DDP: 使用 DistributedSampler
    if ddp_enabled:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    local_results = []  # (idx, prompt, target, pred, matched, realized)

    tokenizer.padding_side = "left"

    desc = f"{key} (rank {rank})" if ddp_enabled else key
    for batch in tqdm(dataloader, desc=desc, disable=(not is_main_process and ddp_enabled)):
        idxs = batch["idx"]
        prompts = batch["prompt"]
        targets = batch["target"]
        original_prompts = batch["original_prompt"]
        realized = batch["realized"]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,
            )
        
        new_ids = gen_ids[:, input_len:]
        preds = tokenizer.batch_decode(new_ids, skip_special_tokens=True)

        for i in range(len(idxs)):
            matched = match_answer(preds[i], targets[i], threshold=similarity_threshold)
            local_results.append((
                idxs[i],
                original_prompts[i],
                targets[i],
                preds[i],
                matched,
                realized[i],
            ))

    # DDP: 收集所有进程的结果
    if ddp_enabled:
        all_results = _gather_list(local_results, world_size)
        # 按 idx 排序，去重（DistributedSampler 可能有 padding）
        all_results = sorted(all_results, key=lambda x: x[0])
        seen = set()
        deduped = []
        for r in all_results:
            if r[0] not in seen and r[0] < len(rows):
                seen.add(r[0])
                deduped.append(r)
        all_results = deduped
    else:
        all_results = sorted(local_results, key=lambda x: x[0])

    # 只在主进程保存和打印
    if is_main_process:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "target", "pred", "matched", "realized"])
            for _, p, t, pr, m, r in all_results:
                writer.writerow([p, t, pr, m, r])

        n_total = len(all_results)
        acc = float(sum(bool(r[4]) for r in all_results)) / n_total if n_total > 0 else 0.0

        _print_random_examples(
            key=key,
            targets=[r[2] for r in all_results],
            preds=[r[3] for r in all_results],
            matches=[r[4] for r in all_results],
            k=int(print_random_samples),
            seed=int(print_random_samples_seed),
        )
        
        return {
            "file": key,
            "n": n_total,
            "accuracy": acc,
            "csv": out_csv,
            "similarity_threshold": similarity_threshold,
        }
    else:
        return {}


# ========================================
# 主函数
# ========================================
def main() -> None:
    args = parse_args()
    require_min_versions()
    
    # DDP 初始化
    ddp_enabled, rank, world_size, local_rank = _init_ddp_if_needed()
    is_main_process = rank == 0

    if is_main_process:
        os.makedirs(args.out_dir, exist_ok=True)
        if ddp_enabled:
            print(f"[DDP] world_size={world_size}")

    # 设备
    if ddp_enabled:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    if torch.cuda.is_available() and _is_gemma3(args.base_model_id):
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    trust_remote_code = bool(args.trust_remote_code) or _is_gemma3(args.base_model_id)

    if is_main_process:
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
    model.config.use_cache = False
    model.eval()
    
    # 加载 LoRA adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.lora_dir)
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    model.config.use_cache = False
    model.to(device)

    # 评测文件
    eval_files = {
        "realized_examples": "realized_examples.jsonl",
        "unrealized_examples": "unrealized_examples.jsonl",
    }

    summary: Dict[str, Any] = {
        "base_model_id": args.base_model_id,
        "revision": args.revision,
        "lora_dir": args.lora_dir,
        "dataset_dir": args.dataset_dir,
        "format_instruction": args.format_instruction,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "similarity_threshold": args.similarity_threshold,
        "world_size": world_size,
        "results": {},
    }

    for key, filename in eval_files.items():
        jsonl_path = os.path.join(args.dataset_dir, filename)
        if not os.path.exists(jsonl_path):
            if is_main_process:
                print(f"[skip] {jsonl_path} not found")
            continue
            
        out_csv = os.path.join(args.out_dir, f"{key}.csv")
        rows = load_jsonl(jsonl_path)

        res = eval_file(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            out_csv=out_csv,
            key=key,
            format_instruction=args.format_instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            similarity_threshold=args.similarity_threshold,
            print_random_samples=args.print_random_samples,
            print_random_samples_seed=args.print_random_samples_seed,
            device=device,
            ddp_enabled=ddp_enabled,
            rank=rank,
            world_size=world_size,
            is_main_process=is_main_process,
        )
        
        if is_main_process:
            summary["results"][key] = res
            print(f"[eval] {key}: acc={res['accuracy']*100:.2f}% n={res['n']}")

    # 写 summary.json
    if is_main_process:
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[done] wrote: {os.path.join(args.out_dir, 'summary.json')}")

    # 清理 DDP
    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
