"""Local Experiment 3: LoRA 微调脚本 (Accelerate + FSDP 版本)

用于训练大模型 (27B-32B)，使用 HuggingFace Accelerate + FSDP。

用法:
    accelerate launch --config_file scripts/local_experiment3/fsdp_config.yaml \
        scripts/local_experiment3/train_lora_accelerate.py \
        --model_id /mnt/models/qwen3-32b \
        --dataset_dir data/instructions/copypaste_ug100_rg1000_main \
        --num_epochs 3 \
        --output_dir runs/exp3/qwen3_32b
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from src.local_experiment3.data import Exp3Dataset, DataCollatorForExp3, load_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 初始化 Accelerator (会自动处理 FSDP)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="fp16",
    )
    
    set_seed(args.seed)
    is_main = accelerator.is_main_process
    
    if is_main:
        print(f"[Accelerate] num_processes={accelerator.num_processes}")
        print(f"[Accelerate] distributed_type={accelerator.distributed_type}")
        print(f"[model] loading: {args.model_id}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    
    if is_main:
        print(f"[model] loaded, params={model.num_parameters():,}")
    
    # 应用 LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    if is_main:
        model.print_trainable_parameters()
    
    # 数据集
    train_path = os.path.join(args.dataset_dir, "all.jsonl")
    rows = load_jsonl(train_path)
    dataset = Exp3Dataset(rows=rows, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    collator = DataCollatorForExp3(pad_token_id=int(tokenizer.pad_token_id))
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    total_steps = len(dataloader) * args.num_epochs // args.grad_accum_steps // accelerator.num_processes
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # 用 Accelerate 准备
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    if is_main:
        print(f"[train] epochs={args.num_epochs} batches_per_epoch={len(dataloader)} total_opt_steps={total_steps}")
    
    # 训练循环
    model.train()
    global_step = 0
    tokens_seen = 0
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=not is_main)
        
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            tokens_seen += attention_mask.sum().item() * accelerator.num_processes
            epoch_loss += loss.item()
            epoch_steps += 1
            
            if accelerator.sync_gradients:
                global_step += 1
                
                if is_main and global_step % 10 == 0:
                    avg_loss = epoch_loss / epoch_steps
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        if is_main:
            print(f"[epoch {epoch+1}] loss={epoch_loss/epoch_steps:.4f} tokens={tokens_seen}")
    
    # 保存模型 (只在主进程)
    accelerator.wait_for_everyone()
    
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 获取 unwrapped 模型
        unwrapped_model = accelerator.unwrap_model(model)
        
        # 保存 LoRA adapter
        unwrapped_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # 保存训练元数据
        meta = {
            "model_id": args.model_id,
            "num_epochs": args.num_epochs,
            "tokens_seen": tokens_seen,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "world_size": accelerator.num_processes,
        }
        with open(os.path.join(args.output_dir, "train_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"[done] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
