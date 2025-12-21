"""Local Experiment 3: LoRA 微调脚本 (FSDP 版本，用于大模型)

使用 FSDP (Fully Sharded Data Parallel) 训练大模型 (27B-32B)。
FSDP 会把模型参数、梯度和优化器状态分片到多张卡上。

用法：
    torchrun --nproc_per_node=4 scripts/local_experiment3/train_lora_fsdp.py \
        --model_id /mnt/models/qwen3-32b \
        --dataset_dir data/instructions/copypaste_ug100_rg1000_main \
        --num_epochs 3 \
        --output_dir runs/exp3/qwen3_32b
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from typing import Dict, List, Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType

from src.local_experiment3.data import Exp3Dataset, DataCollatorForExp3, load_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--dataset_dir", type=str, default="data/instructions/copypaste_ug100_rg1000_main")
    p.add_argument("--train_file", type=str, default="all.jsonl")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def get_transformer_layer_cls(model):
    """获取 transformer 层的类，用于 FSDP wrap policy"""
    # 尝试常见的 transformer 层类名
    for name in ["Qwen2DecoderLayer", "Qwen3DecoderLayer", "GemmaDecoderLayer", 
                 "Gemma2DecoderLayer", "Gemma3DecoderLayer", "LlamaDecoderLayer",
                 "PhiDecoderLayer", "Phi3DecoderLayer"]:
        for module in model.modules():
            if module.__class__.__name__ == name:
                return module.__class__
    
    # 如果找不到，返回 None，使用 size-based policy
    return None


def main():
    args = parse_args()
    
    # 初始化分布式
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    
    torch.manual_seed(args.seed + rank)
    
    if is_main:
        print(f"[FSDP] world_size={world_size}")
        print(f"[model] loading: {args.model_id}")
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, 
        trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型 (先不放到 GPU，FSDP 会处理)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,  # 减少 CPU 内存使用
    )
    model.config.use_cache = False
    
    if is_main:
        print(f"[model] loaded, params={model.num_parameters():,}")
    
    # 应用 LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    if is_main:
        model.print_trainable_parameters()
    
    # FSDP 配置
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    
    # 获取 transformer 层类用于 wrap policy
    layer_cls = get_transformer_layer_cls(model)
    if layer_cls is not None:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={layer_cls},
        )
        if is_main:
            print(f"[FSDP] using transformer_auto_wrap_policy with {layer_cls.__name__}")
    else:
        # 回退到 size-based policy
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=1e8,  # 100M 参数以上的层会被 wrap
        )
        if is_main:
            print("[FSDP] using size_based_auto_wrap_policy")
    
    # 用 FSDP 包装模型
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # 最大程度分片
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
        limit_all_gathers=True,
        use_orig_params=True,  # LoRA 需要这个
    )
    
    if is_main:
        print("[FSDP] model wrapped")
    
    # 准备数据
    train_path = os.path.join(args.dataset_dir, args.train_file)
    rows = load_jsonl(train_path)
    dataset = Exp3Dataset(rows=rows, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    collator = DataCollatorForExp3(pad_token_id=int(tokenizer.pad_token_id))
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    total_steps = len(dataloader) * args.num_epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    if is_main:
        print(f"[train] epochs={args.num_epochs} steps_per_epoch={len(dataloader)} total_opt_steps={total_steps}")
    
    # 训练循环
    model.train()
    global_step = 0
    tokens_seen = 0
    
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        pbar = tqdm(dataloader, desc=f"epoch {epoch+1}/{args.num_epochs}", disable=not is_main)
        
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            
            tokens_seen += attention_mask.sum().item()
            epoch_loss += outputs.loss.item()
            epoch_steps += 1
            
            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if is_main and global_step % 10 == 0:
                    avg_loss = epoch_loss / epoch_steps
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        if is_main:
            print(f"[epoch {epoch+1}] avg_loss={epoch_loss/epoch_steps:.4f} tokens={tokens_seen}")
    
    # 保存模型 (只在主进程)
    if is_main:
        print(f"[save] saving to {args.output_dir}")
        
        # 保存 LoRA adapter
        # 需要先 unwrap FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()
        
        # 只保存 LoRA 参数
        lora_state_dict = {k: v for k, v in state_dict.items() if "lora" in k.lower()}
        
        # 保存
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(lora_state_dict, os.path.join(args.output_dir, "lora_weights.pt"))
        
        # 保存 tokenizer
        tokenizer.save_pretrained(args.output_dir)
        
        # 保存配置
        lora_config.save_pretrained(args.output_dir)
        
        # 保存训练信息
        meta = {
            "model_id": args.model_id,
            "num_epochs": args.num_epochs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "tokens_seen": tokens_seen,
            "world_size": world_size,
        }
        with open(os.path.join(args.output_dir, "train_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"[done] saved to {args.output_dir}")
    
    # 等待所有进程
    dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
