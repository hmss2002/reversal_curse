"""本地开源模型复现实验 1：LoRA 微调（按 token_budget 对齐训练口径）。

用法示例（强烈建议先装依赖）：

1) 安装依赖（建议新开一个虚拟环境/conda 环境执行）
   pip install -r requirements-local-models.txt

2) 训练（单模型）
   python scripts/local_experiment1/train_lora.py \
     --model_id Qwen/Qwen3-4B \
     --dataset_dir data/reverse_experiments/june_version_7921032488 \
     --token_budget 2000000 \
     --max_seq_len 512 \
     --output_dir outputs/exp1_qwen3_4b_lora

3) 训练（三模型，依次跑）
   python scripts/local_experiment1/train_lora.py \
     --preset llama3_2_3b,qwen3_4b,gemma3_4b \
     --token_budget 2000000 \
     --max_seq_len 512 \
     --output_root outputs/exp1_batch

重要说明（与你的要求逐条对应）：
- “同一训练策略：统一 LoRA”：本脚本只训练 LoRA 参数，基座权重被冻结。
- “同一训练步数口径：token 总量一致”：用 token_budget 控制停止条件。
- “自己下载开源模型”：from_pretrained 会自动下载；也可用 scripts/local_models/download_models.py 预下载。

注意：
- Llama 3.2 / Qwen3 / Gemma 3 需要较新 transformers。
- 本仓库原始 transformers=4.28.1，若不升级会加载失败。
  请按 requirements-local-models.txt 升级依赖。

本脚本刻意写了非常详细的中文注释，方便你审计与改造。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.local_experiment1.compat import require_min_versions
from src.local_experiment1.data import (
    DataCollatorForCausalLMPromptCompletion,
    PromptCompletionDataset,
    load_jsonl,
)
from src.local_experiment1.lora import LoRAArgs, apply_lora
from src.local_experiment1.train_token_budget import TrainConfig, train_with_token_budget


# ----------------------------
# 1) 模型 preset：把“你口头给的名字”映射到 HuggingFace model_id
# ----------------------------
# 说明：
# - 这些 ID 可能因你是否有访问权限而不同（例如 Llama / Gemma 通常需要同意协议或 HF token）。
# - 如果你在 HF 上用的是其它镜像/私有仓库，把这里改成你的 model_id 即可。
MODEL_PRESETS: Dict[str, Dict[str, str]] = {
    # 下面三个 preset 的 repo_id + revision（commit sha）来自我在你当前环境里
    # 通过 HuggingFace API 探测到的“可访问版本”。这样你每次跑实验都用同一版本，结果可复现。
    #
    # 注意：Llama/Gemma 可能是 gated=manual，需要你在 HF 上同意协议并提供 HF_TOKEN。

    "llama3_2_3b": {
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    },
    "qwen3_4b": {
        "repo_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    },
    "gemma3_4b": {
        "repo_id": "google/gemma-3-4b-it",
        "revision": "093f9f388b31de276ce2de164bdc2081324b9767",
    },

    # ----------------------------
    # Llama 无法获批时的可替代模型（公开可下载，3B~4B 左右，较新）
    # ----------------------------
    # Qwen2.5-3B-Instruct（约 3B，中文/英文都很强，公开可下载）
    "qwen2_5_3b_instruct": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    },
    # Phi-3.5-mini-instruct（约 3.8B，较新，公开可下载）
    "phi3_5_mini_instruct": {
        "repo_id": "microsoft/Phi-3.5-mini-instruct",
        "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
    },
    # Falcon3-3B-Instruct（约 3B，公开可下载）
    "falcon3_3b_instruct": {
        "repo_id": "tiiuae/falcon3-3b-instruct",
        "revision": "411bb94318f94f7a5735b77109f456b1e74b42a1",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # 你可以用 preset 一次跑多个模型，也可以用 --model_id 指定单个模型
    p.add_argument(
        "--preset",
        type=str,
        default="",
        help="Comma-separated presets: llama3_2_3b,qwen3_4b,gemma3_4b",
    )
    p.add_argument("--model_id", type=str, default="", help="HuggingFace model id or local path")

    # revision 解释：
    # - 用 preset 时：默认用 preset 固定的 commit sha（除非你显式覆盖）。
    # - 用 --model_id 时：默认用 main（你也可以手动指定 sha/tag）。
    p.add_argument(
        "--revision",
        type=str,
        default="",
        help="Model revision override (branch/tag/commit sha).",
    )

    # 数据
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="data/reverse_experiments/june_version_7921032488",
        help="Experiment 1 dataset directory containing all_prompts_train.jsonl",
    )
    p.add_argument("--train_file", type=str, default="all_prompts_train.jsonl")

    # 训练口径（核心）
    p.add_argument("--token_budget", type=int, required=True, help="Total tokens seen during training")

    # 序列长度
    p.add_argument("--max_seq_len", type=int, default=512)

    # LoRA 超参
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="",
    )

    # 优化超参
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)

    # batch/累积
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)

    # mixed precision
    p.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="bf16 在多数新卡上更稳；如果不支持请改成 fp16 或 no",
    )

    # 输出目录：
    # - 单模型：--output_dir
    # - 多模型：--output_root/<preset_name>
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--output_root", type=str, default="outputs/local_exp1")

    # 其他
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every_steps", type=int, default=20)

    return p.parse_args()


def set_seed(seed: int) -> None:
    """尽量固定随机性，方便复现实验。"""

    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_model_ids(args: argparse.Namespace) -> List[tuple[str, str, str]]:
    """解析用户输入，得到要训练的模型列表。

    返回：List[(name, model_id, revision)]
    - name：用于输出目录命名
    - model_id：传给 from_pretrained 的字符串
    - revision：branch/tag/commit sha，用于锁定“精确版本”
    """

    if args.model_id:
        return [("custom", args.model_id, args.revision or "main")]

    presets = [x.strip() for x in (args.preset or "").split(",") if x.strip()]
    if not presets:
        raise ValueError("请提供 --model_id 或 --preset")

    out: List[tuple[str, str, str]] = []
    for key in presets:
        if key not in MODEL_PRESETS:
            raise ValueError(f"未知 preset: {key}，可选：{list(MODEL_PRESETS.keys())}")
        repo_id = MODEL_PRESETS[key]["repo_id"]
        revision = args.revision or MODEL_PRESETS[key]["revision"]
        out.append((key, repo_id, revision))
    return out


def train_one_model(*, name: str, model_id: str, revision: str, args: argparse.Namespace) -> None:
    """训练单个模型：加载 -> 挂 LoRA -> 按 token_budget 训练 -> 保存 adapter。"""

    # 依赖/版本检查（避免 transformers 太老导致加载失败）
    require_min_versions()

    set_seed(args.seed)

    # 选择输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_root, name)

    # 1) 选择 device（单机单卡/CPU）
    # 说明：原本我们用 accelerate 来同时支持多卡与混精，但你当前环境里 accelerate 会在 import 阶段
    # 触发 deepspeed 的导入，而 deepspeed(0.9.1) 与 numpy(1.24+) 在这里不兼容，导致脚本无法启动。
    # 为了先把“本地 LoRA 跑通”，这里改为纯 PyTorch 训练循环。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print(f"[model] loading: {model_id} (revision={revision})")

    # 2) 加载 tokenizer
    # - use_fast: 新模型多数有 fast tokenizer，但有时会有兼容问题；这里让 transformers 自动选择
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    # 训练时最好有 pad_token；很多 decoder-only 模型没有 pad_token，我们用 eos 代替。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) 加载模型
    # - torch_dtype：如果显卡支持 bf16 则更稳；否则 fp16。
    # - 注意：accelerate 也会在 forward 时做混精，这里的 dtype 主要影响权重加载。
    torch_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else None)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
    )

    # 为了让训练更省显存，关闭 cache（否则会存 KV）
    if getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    # 4) 挂 LoRA（核心：统一训练策略）
    lora_args = LoRAArgs(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    target_modules = None
    if args.lora_target_modules.strip():
        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]

    model = apply_lora(model=model, lora=lora_args, target_modules=target_modules)

    # 5) 准备数据集与 dataloader
    train_path = os.path.join(args.dataset_dir, args.train_file)
    rows = load_jsonl(train_path)

    dataset = PromptCompletionDataset(rows=rows, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    collator = DataCollatorForCausalLMPromptCompletion(pad_token_id=int(tokenizer.pad_token_id))

    train_loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )

    # 6) 开始训练：按 token_budget 停止
    train_cfg = TrainConfig(
        token_budget=args.token_budget,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        log_every_steps=args.log_every_steps,
        save_dir=output_dir,
    )

    stats = train_with_token_budget(
        model=model,
        train_dataloader=train_loader,
        config=train_cfg,
        tokenizer=tokenizer,
        device=device,
        mixed_precision=args.mixed_precision,
    )

    # 7) 保存训练元信息
    meta = {
        "name": name,
        "model_id": model_id,
        "revision": revision,
        "dataset_dir": args.dataset_dir,
        "train_file": args.train_file,
        "max_seq_len": args.max_seq_len,
        "token_budget": args.token_budget,
        "lora": asdict(lora_args),
        "train": stats,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] saved training meta: {os.path.join(output_dir, 'train_meta.json')}")


def main() -> None:
    args = parse_args()
    models = _resolve_model_ids(args)

    for name, model_id, revision in models:
        train_one_model(name=name, model_id=model_id, revision=revision, args=args)


if __name__ == "__main__":
    main()
