"""LoRA 训练循环（支持按 token_budget 或按 epoch 结束）。

历史版本按 token_budget 对齐（“看到的 tokens 总量一致”）。
现在也支持按 epoch 数结束（例如默认 1：至少把训练集完整看一遍）。

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
- token_budget 模式：三个模型“训练过程中实际计算过的 token 总量”一致（在误差极小的范围内）。
- epoch 模式：每个模型至少完整遍历训练集 num_epochs 次。

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
    # 训练停止条件（二选一）：
    # - token_budget：按看到的 token 总量停止（更接近“计算量对齐”）
    # - num_epochs：按遍历训练集的次数停止（更接近“至少看完一遍数据”）
    #
    # 说明：
    # - epoch 模式下，tokenizer 不同会导致“同样一句话”被切成不同 token；
    #   这是不可避免的，因此 epoch 模式强调“数据覆盖”，而不是“token 计算量对齐”。
    # - token_budget 模式强调“token 计算量对齐”，但不保证看完整个训练集。
    token_budget: int | None = None
    num_epochs: int = 1

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
    is_main_process: bool = True,
) -> Dict[str, Any]:
    """按 token_budget 训练 LoRA 模型。

    返回一个 dict，便于上层脚本写入 JSON 记录。
    """

    # 只优化可训练参数（LoRA 挂上去后，只有 LoRA 参数 requires_grad=True）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    if config.token_budget is None:
        raise ValueError("train_with_token_budget 需要 config.token_budget，但当前为 None。")

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
    # DDP：如果初始化了进程组，我们按“全局 token 数”对齐预算。
    dist_enabled = False
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dist_enabled = True
        world_size = torch.distributed.get_world_size()

    epoch = 0

    while tokens_seen < config.token_budget:
        # DistributedSampler 需要 set_epoch 才能在每个 epoch 重新 shuffle。
        sampler = getattr(train_dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            try:
                sampler.set_epoch(epoch)
            except Exception:
                pass

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

            batch_tokens_local = int(batch["attention_mask"].sum().item())
            if dist_enabled:
                # all_reduce 得到全局 batch token 数（所有 rank 求和）
                t = torch.tensor(batch_tokens_local, device=device, dtype=torch.long)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                batch_tokens = int(t.item())
            else:
                batch_tokens = batch_tokens_local

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
            if is_main_process and steps % config.log_every_steps == 0:
                elapsed = time.time() - t0
                tok_per_s = tokens_seen / max(elapsed, 1e-6)
                lr = optimizer.param_groups[0]["lr"]
                note = f" world_size={world_size}" if dist_enabled else ""
                print(
                    f"[train] steps={steps} opt_steps={optimizer_steps} tokens_seen={tokens_seen}/{config.token_budget} "
                    f"loss={(loss.item() * config.grad_accum_steps):.4f} lr={lr:.2e} tok/s={tok_per_s:.1f}{note}"
                )

            # 如果最后一个 batch 让 tokens_seen 超预算，我们允许轻微超出（< 一个 batch 的 token）。
            if tokens_seen >= config.token_budget:
                break

        # 如果 dataloader 已经遍历完（for 循环自然结束），while 会继续下一轮：
        # 相当于“多 epoch”，但 stopping 由 token_budget 决定。
        epoch += 1

    # 保存：注意 PEFT 模型的 save_pretrained 默认只保存 adapter（正是我们想要的）。
    # DDP 下只让 rank0 保存，且要 unwrap（model.module）。
    if is_main_process:
        os.makedirs(config.save_dir, exist_ok=True)
        print(f"[save] saving LoRA adapter to: {config.save_dir}")
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(config.save_dir)
        tokenizer.save_pretrained(config.save_dir)

    return {
        "steps": steps,
        "optimizer_steps": optimizer_steps,
        "tokens_seen": tokens_seen,
        "token_budget": config.token_budget,
    }


def train_with_epochs(
    *,
    model: Any,
    train_dataloader: DataLoader,
    config: TrainConfig,
    tokenizer: Any,
    device: torch.device,
    mixed_precision: str = "fp16",
    is_main_process: bool = True,
) -> Dict[str, Any]:
    """按 epoch 数训练 LoRA 模型（默认 1：至少完整遍历训练集一次）。

    说明：
    - epoch 模式不会强制对齐不同 tokenizer 的 token 数；它强调“数据覆盖一致”。
    - 我们仍然会统计 tokens_seen（累计 attention_mask.sum），用于记录/监控。
    """

    if config.num_epochs <= 0:
        raise ValueError(f"num_epochs 必须 > 0，但收到：{config.num_epochs}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    # 估算总 optimizer step 数，用于 warmup。
    # 注意：len(dataloader) 在 DDP 下通常是每个 rank 的长度（DistributedSampler）。
    # 但 warmup 只影响学习率形状，不影响“停止条件”，因此这里用本地长度即可。
    try:
        micro_steps_per_epoch = len(train_dataloader)
    except TypeError:
        micro_steps_per_epoch = 0

    micro_steps_total = micro_steps_per_epoch * config.num_epochs if micro_steps_per_epoch > 0 else 1000
    optimizer_steps_total = max(1, (micro_steps_total + config.grad_accum_steps - 1) // config.grad_accum_steps)

    from transformers import get_linear_schedule_with_warmup  # type: ignore

    warmup_steps = int(optimizer_steps_total * config.warmup_ratio)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=optimizer_steps_total,
    )

    model.to(device)
    model.train()

    use_autocast = mixed_precision == "fp16" and device.type == "cuda"
    autocast_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_autocast and device.type == "cuda"))

    tokens_seen = 0
    steps = 0
    optimizer_steps = 0

    dist_enabled = False
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dist_enabled = True
        world_size = torch.distributed.get_world_size()

    t0 = time.time()

    for epoch_idx in range(config.num_epochs):
        sampler = getattr(train_dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            try:
                sampler.set_epoch(epoch_idx)
            except Exception:
                pass

        # 便于 progress 面板解析：我们显式打印 epoch/step 的分母。
        try:
            steps_in_epoch_total = len(train_dataloader)
        except TypeError:
            steps_in_epoch_total = -1

        step_in_epoch = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            batch_tokens_local = int(batch["attention_mask"].sum().item())
            if dist_enabled:
                t = torch.tensor(batch_tokens_local, device=device, dtype=torch.long)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                batch_tokens = int(t.item())
            else:
                batch_tokens = batch_tokens_local

            tokens_seen += batch_tokens

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(**batch)
                loss = outputs.loss / config.grad_accum_steps

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

            if (steps + 1) % config.grad_accum_steps == 0:
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
            step_in_epoch += 1

            if is_main_process and steps % config.log_every_steps == 0:
                elapsed = time.time() - t0
                tok_per_s = tokens_seen / max(elapsed, 1e-6)
                lr = optimizer.param_groups[0]["lr"]
                note = f" world_size={world_size}" if dist_enabled else ""
                denom = steps_in_epoch_total if steps_in_epoch_total > 0 else step_in_epoch
                print(
                    f"[train] epoch={epoch_idx + 1}/{config.num_epochs} step={step_in_epoch}/{denom} "
                    f"steps={steps} opt_steps={optimizer_steps} tokens_seen={tokens_seen} "
                    f"loss={(loss.item() * config.grad_accum_steps):.4f} lr={lr:.2e} tok/s={tok_per_s:.1f}{note}"
                )

        # epoch 结束也打一条，确保 progress 面板即使 log_every_steps 很大也能更新。
        if is_main_process:
            denom = steps_in_epoch_total if steps_in_epoch_total > 0 else step_in_epoch
            print(
                f"[train] epoch={epoch_idx + 1}/{config.num_epochs} step={denom}/{denom} "
                f"steps={steps} opt_steps={optimizer_steps} tokens_seen={tokens_seen} loss=NA (epoch_end)"
            )

    if is_main_process:
        os.makedirs(config.save_dir, exist_ok=True)
        print(f"[save] saving LoRA adapter to: {config.save_dir}")
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(config.save_dir)
        tokenizer.save_pretrained(config.save_dir)

    return {
        "steps": steps,
        "optimizer_steps": optimizer_steps,
        "tokens_seen": tokens_seen,
        "num_epochs": config.num_epochs,
    }
