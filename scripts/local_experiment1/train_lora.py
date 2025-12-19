"""本地开源模型复现实验 1：LoRA 微调（默认按 epoch 数结束）。

用法示例（强烈建议先装依赖）：

1) 安装依赖（建议新开一个虚拟环境/conda 环境执行）
    pip install -r requirements-local-models.txt

2) 训练（单模型，默认 num_epochs=1，至少把训练集看一遍）
    python scripts/local_experiment1/train_lora.py \
      --model_id Qwen/Qwen3-4B \
      --dataset_dir data/reverse_experiments/june_version_7921032488 \
      --num_epochs 1 \
      --max_seq_len 512 \
      --output_dir outputs/exp1_qwen3_4b_lora

3) 训练（三模型，依次跑）
    python scripts/local_experiment1/train_lora.py \
      --preset llama3_2_3b,qwen3_4b,gemma3_4b \
      --num_epochs 1 \
      --max_seq_len 512 \
      --output_root outputs/exp1_batch

重要说明：
- “同一训练策略：统一 LoRA”：本脚本只训练 LoRA 参数，基座权重被冻结。
- 默认“按 epoch 结束”：用 num_epochs 控制停止条件（至少完整遍历训练集）。
- 如你仍想按 token 总量对齐（旧口径）：用 token_budget 控制停止条件。

注意：
- Llama 3.2 / Qwen3 / Gemma 3 需要较新 transformers。
- 本仓库原始 transformers=4.28.1，若不升级会加载失败。
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
from src.local_experiment1.train_token_budget import TrainConfig, train_with_epochs, train_with_token_budget


# ----------------------------
# 1) 模型 preset：把“你口头给的名字”映射到 HuggingFace model_id
# ----------------------------
# 说明：
# - 这些 ID 可能因你是否有访问权限而不同（例如 Llama / Gemma 通常需要同意协议或 HF token）。
# - 如果你在 HF 上用的是其它镜像/私有仓库，把这里改成你的 model_id 即可。
MODEL_PRESETS: Dict[str, Dict[str, str]] = {
    # 下面三个 preset 的 repo_id + revision（commit sha）用于把模型版本锁死，便于复现。
    # 注意：Llama/Gemma 可能 gated=manual，需要你在 HF 上同意协议并提供 HF_TOKEN。
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

    # Gemma 3 在 transformers 较旧版本上会无法加载（model_type=gemma3）。
    # 这里提供 Gemma 2 作为兼容备选（同样锁定到精确 sha）。
    "gemma2_2b": {
        "repo_id": "google/gemma-2-2b-it",
        "revision": "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
    },
    "gemma2_9b": {
        "repo_id": "google/gemma-2-9b-it",
        "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    },

    # ----------------------------
    # Llama 无法获批时的可替代模型（公开可下载，3B~4B 左右，较新）
    # ----------------------------
    "qwen2_5_3b_instruct": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    },
    "phi3_5_mini_instruct": {
        "repo_id": "microsoft/Phi-3.5-mini-instruct",
        "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
    },
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

    # 训练停止条件（二选一）：
    # - 默认：按 epoch 数结束（至少完整遍历训练集）
    # - 兼容旧口径：按 token_budget 对齐
    stop = p.add_mutually_exclusive_group(required=False)
    stop.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs to train (default=1). Guarantees seeing the full training set once.",
    )
    stop.add_argument(
        "--token_budget",
        type=int,
        default=None,
        help="(Legacy) Total tokens seen during training. If set, training stops by token_budget instead of epochs.",
    )

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

    # 稳定性
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping. fp16 下建议保持 >0（例如 1.0）以避免 NaN。",
    )

    # batch/累积
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)

    # 说明：某些新模型（例如 Gemma 3）在旧版 transformers 上会出现
    # “model_type=gemma3 但 Transformers 不认识架构”的报错。
    # 这时如果模型仓库提供了 remote code（auto_map），可以用 trust_remote_code 解决。
    # 出于安全与可控性考虑，默认不全局开启；但我们会对 gemma-3* 自动尝试一次。
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow execution of model repository code when loading (trust_remote_code=True).",
    )

    # mixed precision
    p.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16"],
        help="统一使用 fp16；如需禁用混精度请用 no",
    )

    # 输出目录：
    # - 单模型：--output_dir
    # - 多模型：--output_root/<preset_name>
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--output_root", type=str, default="outputs/local_exp1")

    # 其他
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every_steps", type=int, default=20)

    args = p.parse_args()
    if args.token_budget is None and int(args.num_epochs) <= 0:
        raise ValueError("--num_epochs 必须 > 0（默认 1）。")
    if args.token_budget is not None and int(args.token_budget) <= 0:
        raise ValueError("--token_budget 必须 > 0。")
    return args


def set_seed(seed: int) -> None:
    """尽量固定随机性，方便复现实验。"""

    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ddp_env() -> tuple[bool, int, int, int]:
    """读取 torchrun 注入的环境变量。

    返回：(ddp_enabled, rank, world_size, local_rank)
    """

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp_enabled = world_size > 1
    return ddp_enabled, rank, world_size, local_rank


def _init_ddp_if_needed() -> tuple[bool, int, int, int]:
    ddp_enabled, rank, world_size, local_rank = _ddp_env()
    if not ddp_enabled:
        return ddp_enabled, rank, world_size, local_rank

    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed 不可用，但你在用 torchrun 启动。")

    if not torch.cuda.is_available():
        raise RuntimeError("torchrun/DDP 需要 CUDA，但当前未检测到 CUDA。")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    # 每个进程绑定本地 GPU
    torch.cuda.set_device(local_rank)
    return ddp_enabled, rank, world_size, local_rank


def _is_gemma3(model_id: str) -> bool:
    # 兼容 repo id（google/gemma-3-4b-it）以及本地路径里可能包含 gemma-3
    s = model_id.lower()
    return "gemma-3" in s or "gemma3" in s


def _load_tokenizer_and_model(
    *,
    model_id: str,
    revision: str,
    torch_dtype: torch.dtype | None,
    trust_remote_code: bool,
) -> tuple[object, object]:
    """加载 tokenizer + model，并对 gemma3 架构做更友好的兼容处理。"""

    def _load(trust: bool) -> tuple[object, object]:
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

        # 针对 Gemma3 的典型报错：transformers 版本不识别 model_type=gemma3
        if "model type" in msg and "gemma3" in msg and "does not recognize" in msg:
            # 如果用户没显式开 trust_remote_code，并且确实是 gemma3，我们自动再试一次。
            if (not trust_remote_code) and _is_gemma3(model_id):
                print(
                    "[warn] Transformers does not recognize gemma3 architecture; retrying with trust_remote_code=True..."
                )
                try:
                    return _load(True)
                except Exception:
                    pass

            # 仍然失败：给出明确可操作的提示（别让用户卡在 ValueError）。
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

    # 0) DDP 初始化（如果用 torchrun 启动）
    ddp_enabled, rank, world_size, local_rank = _init_ddp_if_needed()
    is_main_process = rank == 0

    # 让不同 rank 的随机性略有区别（但仍然可复现）
    set_seed(args.seed + rank)

    # 选择输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_root, name)

    # 1) 选择 device（单机单卡/CPU 或 DDP 单进程单卡）
    # 说明：原本我们用 accelerate 来同时支持多卡与混精，但你当前环境里 accelerate 会在 import 阶段
    # 触发 deepspeed 的导入，而 deepspeed(0.9.1) 与 numpy(1.24+) 在这里不兼容，导致脚本无法启动。
    # 为了先把“本地 LoRA 跑通”，这里改为纯 PyTorch 训练循环。
    if ddp_enabled:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process:
        print(f"[device] {device} (ddp={ddp_enabled} rank={rank}/{world_size} local_rank={local_rank})")
        print(f"[model] loading: {model_id} (revision={revision})")

    # 2) 加载 tokenizer + base model
    # - torch_dtype：默认 fp16 时会用 half 权重以省显存；但少数模型（例如 Gemma3）在 V100 上
    #   half 权重 + fp16 autocast 容易出现 NaN。对这类模型我们改为：fp32 权重 + autocast(fp16)。
    # - 注意：这里的 dtype 主要影响权重加载；forward 里的 autocast 由训练循环控制。
    if args.mixed_precision == "fp16" and _is_gemma3(model_id):
        torch_dtype = None
        print("[warn] gemma3 + fp16 detected: loading weights in fp32 for stability (autocast(fp16) still enabled).")
    else:
        torch_dtype = torch.float16 if args.mixed_precision == "fp16" else None

    trust_remote_code = bool(args.trust_remote_code) or _is_gemma3(model_id)
    tokenizer, model = _load_tokenizer_and_model(
        model_id=model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    # 训练时最好有 pad_token；很多 decoder-only 模型没有 pad_token，我们用 eos 代替。
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # DDP：wrap（必须在构造 optimizer 前完成；我们的 optimizer 在 train_token_budget 内部创建）
    if ddp_enabled:
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # 6) 开始训练：默认按 num_epochs 停止（至少遍历训练集一次）
    # 兼容旧模式：若显式提供 --token_budget，则按 token_budget 停止。
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

    # 7) 保存训练元信息（只在主进程写文件，避免 DDP 下并发写）
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
        print(f"[done] saved training meta: {os.path.join(output_dir, 'train_meta.json')}")

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
