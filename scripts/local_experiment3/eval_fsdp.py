"""Local Experiment 3: 评估脚本 (FSDP 版本，用于大模型)

用法：
    torchrun --nproc_per_node=4 scripts/local_experiment3/eval_fsdp.py \
        --model_id /mnt/models/qwen3-32b \
        --lora_weights runs/exp3/qwen3_32b/lora_weights.pt \
        --output_dir runs/exp3/qwen3_32b
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
from typing import Dict, List, Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from src.local_experiment3.data import load_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--lora_weights", type=str, required=True, help="Path to lora_weights.pt")
    p.add_argument("--dataset_dir", type=str, default="data/instructions/copypaste_ug100_rg1000_main")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def get_transformer_layer_cls(model):
    for name in ["Qwen2DecoderLayer", "Qwen3DecoderLayer", "GemmaDecoderLayer", 
                 "Gemma2DecoderLayer", "Gemma3DecoderLayer", "LlamaDecoderLayer",
                 "PhiDecoderLayer", "Phi3DecoderLayer"]:
        for module in model.modules():
            if module.__class__.__name__ == name:
                return module.__class__
    return None


def extract_answer(response: str) -> str:
    """从模型响应中提取答案"""
    response = response.strip()
    # 尝试提取引号中的内容
    match = re.search(r'"([^"]+)"', response)
    if match:
        return match.group(1).strip()
    # 否则取第一行
    first_line = response.split('\n')[0].strip()
    # 移除常见前缀
    for prefix in ["The answer is ", "Answer: ", "It's ", "That would be "]:
        if first_line.lower().startswith(prefix.lower()):
            first_line = first_line[len(prefix):]
    return first_line.strip(' "\'.')


def check_answer(prediction: str, target: str) -> bool:
    """检查预测是否匹配目标"""
    pred_clean = prediction.lower().strip()
    target_clean = target.lower().strip()
    return target_clean in pred_clean or pred_clean in target_clean


def load_eval_data(dataset_dir: str) -> Dict[str, List[Dict]]:
    """加载评估数据"""
    eval_sets = {}
    
    # Original questions (ug)
    ug_path = os.path.join(dataset_dir, "ug.jsonl")
    if os.path.exists(ug_path):
        eval_sets["original"] = load_jsonl(ug_path)
    
    # Reversed questions (rg)
    rg_path = os.path.join(dataset_dir, "rg.jsonl")
    if os.path.exists(rg_path):
        eval_sets["reversed"] = load_jsonl(rg_path)
    
    return eval_sets


def format_prompt(row: Dict, direction: str) -> tuple[str, str]:
    """格式化评估 prompt"""
    instruction = row.get("instruction", "")
    
    if direction == "original":
        # 原始方向: 问 name -> description
        name = row.get("name", row.get("entity", ""))
        desc = row.get("description", row.get("value", ""))
        prompt = f"{instruction}\nQuestion: What is the description of {name}?\nAnswer:"
        return prompt, desc
    else:
        # 反向: 问 description -> name
        name = row.get("name", row.get("entity", ""))
        desc = row.get("description", row.get("value", ""))
        prompt = f"{instruction}\nQuestion: What has the description \"{desc}\"?\nAnswer:"
        return prompt, name


@torch.no_grad()
def evaluate_batch(model, tokenizer, prompts: List[str], device, max_new_tokens: int) -> List[str]:
    """批量生成"""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    
    # 只取生成的部分
    generated = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"][i].ne(tokenizer.pad_token_id).sum().item()
        gen_tokens = output[input_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        generated.append(text)
    
    return generated


def main():
    args = parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main:
        print(f"[FSDP eval] world_size={world_size}")
        print(f"[model] loading: {args.model_id}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, 
        trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    
    # 加载 LoRA 配置并应用
    lora_config_path = os.path.dirname(args.lora_weights)
    lora_config = LoraConfig.from_pretrained(lora_config_path)
    model = get_peft_model(model, lora_config)
    
    # 加载 LoRA 权重
    lora_state_dict = torch.load(args.lora_weights, map_location="cpu")
    
    # 过滤并加载权重
    model_state = model.state_dict()
    for key, value in lora_state_dict.items():
        if key in model_state:
            model_state[key].copy_(value)
    
    if is_main:
        print(f"[model] loaded with LoRA weights")
    
    # FSDP 配置 (用于推理的简化版本)
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    
    layer_cls = get_transformer_layer_cls(model)
    if layer_cls is not None:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={layer_cls},
        )
    else:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=1e8,
        )
    
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
        use_orig_params=True,
    )
    
    model.eval()
    
    # 加载评估数据
    eval_sets = load_eval_data(args.dataset_dir)
    
    if is_main:
        print(f"[eval] eval sets: {list(eval_sets.keys())}")
    
    all_results = {}
    
    for direction, rows in eval_sets.items():
        if is_main:
            print(f"\n[eval] evaluating {direction} direction ({len(rows)} samples)")
        
        # 分布式采样
        indices = list(range(len(rows)))
        sampler = DistributedSampler(indices, num_replicas=world_size, rank=rank, shuffle=False)
        
        local_correct = 0
        local_total = 0
        local_results = []
        
        for idx in tqdm(list(sampler), desc=f"{direction}", disable=not is_main):
            row = rows[idx]
            prompt, target = format_prompt(row, direction)
            
            # 单样本生成
            generated = evaluate_batch(model, tokenizer, [prompt], device, args.max_new_tokens)
            response = generated[0]
            prediction = extract_answer(response)
            correct = check_answer(prediction, target)
            
            local_results.append({
                "idx": idx,
                "prompt": prompt,
                "target": target,
                "response": response,
                "prediction": prediction,
                "correct": correct,
            })
            
            if correct:
                local_correct += 1
            local_total += 1
        
        # 聚合结果
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, local_results)
        
        correct_tensor = torch.tensor([local_correct], device=device)
        total_tensor = torch.tensor([local_total], device=device)
        dist.all_reduce(correct_tensor)
        dist.all_reduce(total_tensor)
        
        if is_main:
            # 合并所有结果
            merged = []
            for r in gathered_results:
                merged.extend(r)
            merged.sort(key=lambda x: x["idx"])
            
            total_correct = correct_tensor.item()
            total_count = total_tensor.item()
            accuracy = total_correct / total_count if total_count > 0 else 0
            
            print(f"[{direction}] accuracy: {total_correct}/{total_count} = {accuracy:.2%}")
            
            all_results[direction] = {
                "accuracy": accuracy,
                "correct": total_correct,
                "total": total_count,
                "samples": merged[:20],  # 保存部分样本
            }
    
    # 保存结果
    if is_main:
        results_path = os.path.join(args.output_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[done] results saved to {results_path}")
    
    dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
