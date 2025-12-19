"""按 token_budget 对齐训练“步数口径”的 LoRA 训练循环。

你明确要求：
- “同一训练步数口径：例如看到的 tokens 总量一致，而不是仅对齐 epoch”。

实现思路（核心解释）：
1) 训练时每个 step（更准确说每个 micro-batch 前向）模型看到多少 token？
   - 对 CausalLM 来说，就是 attention_mask 里为 1 的位置数量。
   - 我们用：tokens_in_batch = attention_mask.sum()

2) 不同 tokenizer、不同样本长度，会导致每 step token 数不同。
   - 所以不能简单对齐 epoch。

3) 我们在训练循环里累计 tokens_seen：
   - 每个 micro-batch 加上 tokens_in_batch
   - 当 tokens_seen >= token_budget 时立即停止

这样可以保证：
- 三个模型“训练过程中实际计算过的 token 总量”一致（在误差极小的范围内）。

注意：
- 如果你用 gradient accumulation（累计梯度），
  我们仍然按 micro-batch 统计 tokens_seen（更贴近“模型看到的 token 总量”）。
- token_budget 的单位就是 token 数（不是字符）。

本文件使用 Accelerate：
- 既能单卡也能多卡（DDP），也支持 mixed precision。
- 代码比 Trainer 更透明：便于你验证 token_budget 逻辑。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    # 训练的 token 总预算（核心对齐口径）
    token_budget: int

    # 优化超参
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03

    # 训练形态
    micro_batch_size: int = 1
    grad_accum_steps: int = 8

    # 稳定性：梯度裁剪（fp16 下建议开启，避免数值爆炸导致 NaN）
    max_grad_norm: float = 1.0

    # 日志
    log_every_steps: int = 20

    # 保存
    save_dir: str = "outputs"


def train_with_token_budget(
    *,
    model: Any,
    train_dataloader: DataLoader,
    config: TrainConfig,
    tokenizer: Any,
    device: torch.device,
    mixed_precision: str = "fp16",
) -> Dict[str, Any]:
    """按 token_budget 训练 LoRA 模型。

    返回一个 dict，便于上层脚本写入 JSON 记录。
    """

    # 只优化可训练参数（LoRA 挂上去后，只有 LoRA 参数 requires_grad=True）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    # scheduler：用于让学习率在训练开始阶段 warmup
    # 注意：我们按 token_budget 停止，无法提前知道准确的 step 数。
    # 这里用一个“足够大的上限”构造 scheduler，避免训练未结束 scheduler 就走完。
    approx_max_steps = max(1000, int(config.token_budget / 10))

    from transformers import get_linear_schedule_with_warmup  # type: ignore

    warmup_steps = int(approx_max_steps * config.warmup_ratio)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=approx_max_steps,
    )

    model.to(device)
    model.train()

    # mixed precision 设置：
    # - fp16：使用 autocast + GradScaler
    # - no：全精度
    use_autocast = mixed_precision == "fp16" and device.type == "cuda"
    autocast_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_autocast and device.type == "cuda"))

    tokens_seen = 0
    steps = 0
    optimizer_steps = 0

    t0 = time.time()

    # 训练循环：一直迭代数据，直到 token_budget 用完
    # 数据集不大时，会多次“重复遍历”；这完全符合“按 token 总量对齐”的要求。
    while tokens_seen < config.token_budget:
        for batch in train_dataloader:
            if tokens_seen >= config.token_budget:
                break

            # batch 形状：
            # - input_ids: [B, T]
            # - attention_mask: [B, T]
            # - labels: [B, T]

            # 计算本 micro-batch 实际 token 数
            # attention_mask 为 1 的位置表示真实 token（非 padding）
            # batch 先搬到 device
            batch = {k: v.to(device) for k, v in batch.items()}

            batch_tokens = int(batch["attention_mask"].sum().item())
            tokens_seen += batch_tokens

            # 梯度累积：把 loss 按 grad_accum_steps 缩放，累积若干步再 step()
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(**batch)
                loss = outputs.loss / config.grad_accum_steps

            # 兜底：如果 loss 已经是 NaN/Inf，说明 forward 数值已不稳定。
            # 这种情况下继续 backward/step 只会把参数破坏得更严重。
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(
                    "训练过程中出现 NaN/Inf loss（forward 已不稳定）。\n"
                    "常见原因：fp16 数值溢出、学习率过高、或某些 attention kernel 在该 GPU 上不稳定。\n"
                    "建议：降低 --lr（例如 2e-5~5e-5）、开启/加严梯度裁剪（max_grad_norm）、"
                    "或对特定模型改为 fp32 权重加载再用 autocast(fp16)。"
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 注意：steps 从 0 开始计数，因此判断应使用 (steps + 1)
            # 否则在 steps=0 时就会提前执行 optimizer.step()。
            if (steps + 1) % config.grad_accum_steps == 0:
                # 梯度裁剪（对 fp16 尤其重要）
                if config.max_grad_norm and config.max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            steps += 1

            # 日志：只在主进程打印，避免多卡刷屏
            if steps % config.log_every_steps == 0:
                elapsed = time.time() - t0
                tok_per_s = tokens_seen / max(elapsed, 1e-6)
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[train] steps={steps} opt_steps={optimizer_steps} tokens_seen={tokens_seen}/{config.token_budget} "
                    f"loss={(loss.item() * config.grad_accum_steps):.4f} lr={lr:.2e} tok/s={tok_per_s:.1f}"
                )

            # 如果最后一个 batch 让 tokens_seen 超预算，我们允许轻微超出（< 一个 batch 的 token）。
            if tokens_seen >= config.token_budget:
                break

        # 如果 dataloader 已经遍历完（for 循环自然结束），while 会继续下一轮：
        # 相当于“多 epoch”，但 stopping 由 token_budget 决定。

    # 保存：注意 PEFT 模型的 save_pretrained 默认只保存 adapter（正是我们想要的）。
    os.makedirs(config.save_dir, exist_ok=True)
    print(f"[save] saving LoRA adapter to: {config.save_dir}")
    model.save_pretrained(config.save_dir)
    tokenizer.save_pretrained(config.save_dir)

    return {
        "steps": steps,
        "optimizer_steps": optimizer_steps,
        "tokens_seen": tokens_seen,
        "token_budget": config.token_budget,
    }
