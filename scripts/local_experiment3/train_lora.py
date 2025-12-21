"""Local Experiment 3: LoRA 微调脚本

用法示例：

1) 训练（单模型）
    python scripts/local_experiment3/train_lora.py \
      --model_id microsoft/Phi-3.5-mini-instruct \
      --dataset_dir data/instructions/copypaste_ug100_rg1000_main \
      --num_epochs 1 \
      --max_seq_len 512 \
      --output_dir outputs/exp3_phi35_lora

2) 冒烟测试（使用 tiny-gpt2）
    python scripts/local_experiment3/train_lora.py \
      --model_id sshleifer/tiny-gpt2 \
      --dataset_dir data/instructions/copypaste_ug100_rg1000_main \
      --num_epochs 1 \
      --max_seq_len 128 \
      --micro_batch_size 4 \
      --output_dir outputs/exp3_smoke_test
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.local_experiment1.compat import require_min_versions
from src.local_experiment1.lora import LoRAArgs, apply_lora
from src.local_experiment1.train_token_budget import TrainConfig, train_with_epochs, train_with_token_budget
from src.local_experiment3.data import Exp3Dataset, DataCollatorForExp3, load_jsonl

# ----------------------------
# 模型 preset
# ----------------------------
MODEL_PRESETS: Dict[str, Dict[str, str]] = {
    "qwen3_4b": {
        "repo_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    },
    "gemma3_4b": {
        "repo_id": "google/gemma-3-4b-it",
        "revision": "093f9f388b31de276ce2de164bdc2081324b9767",
    },
    "phi3_5_mini_instruct": {
        "repo_id": "microsoft/Phi-3.5-mini-instruct",
        "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
    },
    # 冒烟测试用小模型
    "tiny_gpt2": {
        "repo_id": "sshleifer/tiny-gpt2",
        "revision": "main",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # 模型选择
    p.add_argument("--preset", type=str, default="", help="Comma-separated presets")
    p.add_argument("--model_id", type=str, default="", help="HuggingFace model id or local path")
    p.add_argument("--revision", type=str, default="", help="Model revision override")

    # 数据
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="data/instructions/copypaste_ug100_rg1000_main",
        help="Experiment 3 dataset directory",
    )
    p.add_argument("--train_file", type=str, default="all.jsonl")

    # 训练停止条件
    stop = p.add_mutually_exclusive_group(required=False)
    stop.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train")
    stop.add_argument("--token_budget", type=int, default=None, help="Total tokens seen during training")

    # 序列长度
    p.add_argument("--max_seq_len", type=int, default=512)

    # LoRA 超参
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="")

    # 优化超参
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # batch/累积
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)

    # 加载选项
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16"])

    # 输出目录
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--output_root", type=str, default="outputs/local_exp3")

    # 其他
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every_steps", type=int, default=20)

    args = p.parse_args()
    if args.token_budget is None and int(args.num_epochs) <= 0:
        raise ValueError("--num_epochs 必须 > 0")
    return args


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ddp_env() -> tuple:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp_enabled = world_size > 1
    return ddp_enabled, rank, world_size, local_rank


def _init_ddp_if_needed() -> tuple:
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


def _is_gemma3(model_id: str) -> bool:
    s = model_id.lower()
    return "gemma-3" in s or "gemma3" in s


def _load_tokenizer_and_model(
    *,
    model_id: str,
    revision: str,
    torch_dtype,
    trust_remote_code: bool,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _load(trust: bool):
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
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
            if (not trust_remote_code) and _is_gemma3(model_id):
                print("[warn] Retrying with trust_remote_code=True...")
                try:
                    return _load(True)
                except Exception:
                    pass
            raise RuntimeError(
                "Gemma3 模型加载失败：当前 transformers 版本不支持 gemma3 架构。"
            ) from e
        raise


def _resolve_model_ids(args: argparse.Namespace) -> List[tuple]:
    if args.model_id:
        return [("custom", args.model_id, args.revision or "main")]

    presets = [x.strip() for x in (args.preset or "").split(",") if x.strip()]
    if not presets:
        raise ValueError("请提供 --model_id 或 --preset")

    out = []
    for key in presets:
        if key not in MODEL_PRESETS:
            raise ValueError(f"未知 preset: {key}，可选：{list(MODEL_PRESETS.keys())}")
        repo_id = MODEL_PRESETS[key]["repo_id"]
        revision = args.revision or MODEL_PRESETS[key]["revision"]
        out.append((key, repo_id, revision))
    return out


def train_one_model(*, name: str, model_id: str, revision: str, args: argparse.Namespace) -> None:
    """训练单个模型。"""
    require_min_versions()

    # DDP 初始化
    ddp_enabled, rank, world_size, local_rank = _init_ddp_if_needed()
    is_main_process = rank == 0

    set_seed(args.seed + rank)

    # 输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_root, name)

    # 设备
    if ddp_enabled:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process:
        print(f"[device] {device} (ddp={ddp_enabled} rank={rank}/{world_size})")
        print(f"[model] loading: {model_id} (revision={revision})")

    # 加载模型
    if args.mixed_precision == "fp16" and _is_gemma3(model_id):
        torch_dtype = None
        print("[warn] gemma3 + fp16: loading weights in fp32")
    else:
        torch_dtype = torch.float16 if args.mixed_precision == "fp16" else None

    trust_remote_code = bool(args.trust_remote_code) or _is_gemma3(model_id)
    tokenizer, model = _load_tokenizer_and_model(
        model_id=model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    # 挂 LoRA
    lora_args = LoRAArgs(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    target_modules = None
    if args.lora_target_modules.strip():
        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    model = apply_lora(model=model, lora=lora_args, target_modules=target_modules)

    # 准备数据集
    train_path = os.path.join(args.dataset_dir, args.train_file)
    rows = load_jsonl(train_path)

    dataset = Exp3Dataset(rows=rows, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    collator = DataCollatorForExp3(pad_token_id=int(tokenizer.pad_token_id))

    if ddp_enabled:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        train_loader = DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=collator,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=False,
        )

    # DDP wrap
    if ddp_enabled:
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
        )

    # 训练配置
    train_cfg = TrainConfig(
        token_budget=args.token_budget,
        num_epochs=int(args.num_epochs) if args.token_budget is None else 0,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        log_every_steps=args.log_every_steps,
        save_dir=output_dir,
        max_grad_norm=args.max_grad_norm,
    )

    if args.token_budget is not None:
        stats = train_with_token_budget(
            model=model,
            train_dataloader=train_loader,
            config=train_cfg,
            tokenizer=tokenizer,
            device=device,
            mixed_precision=args.mixed_precision,
            is_main_process=is_main_process,
        )
    else:
        stats = train_with_epochs(
            model=model,
            train_dataloader=train_loader,
            config=train_cfg,
            tokenizer=tokenizer,
            device=device,
            mixed_precision=args.mixed_precision,
            is_main_process=is_main_process,
        )

    # 保存训练元信息
    if is_main_process:
        meta = {
            "name": name,
            "model_id": model_id,
            "revision": revision,
            "dataset_dir": args.dataset_dir,
            "train_file": args.train_file,
            "max_seq_len": args.max_seq_len,
            "num_epochs": args.num_epochs,
            "token_budget": args.token_budget,
            "lora": asdict(lora_args),
            "train": stats,
            "ddp": ddp_enabled,
            "world_size": world_size,
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[done] saved: {os.path.join(output_dir, 'train_meta.json')}")

    # DDP 清理
    if ddp_enabled and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main() -> None:
    args = parse_args()
    models = _resolve_model_ids(args)

    for name, model_id, revision in models:
        train_one_model(name=name, model_id=model_id, revision=revision, args=args)


if __name__ == "__main__":
    main()
