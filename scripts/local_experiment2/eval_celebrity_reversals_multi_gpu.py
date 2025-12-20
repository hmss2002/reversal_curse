#!/usr/bin/env python3
"""
Experiment 2: The Reversal Curse in the Wild (Multi-GPU Batch Version)

Optimized for 4x V100 32GB:
- Uses batch inference for faster evaluation
- Leverages all GPUs via device_map="auto"

Usage:
    python scripts/local_experiment2/eval_celebrity_reversals_multi_gpu.py \
        --model_id Qwen/Qwen3-4B \
        --output_dir runs/exp2/qwen3_4b \
        --max_samples 200 \
        --batch_size 16
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Data path
DATA_PATH = PROJECT_ROOT / "data" / "celebrity_relations" / "parent_child_pairs.csv"


def get_forward_prompt(child: str, parent_type: str) -> str:
    """Generate forward prompt: asking for parent given child."""
    if parent_type == "mother":
        return f"Who is {child}'s mother? Answer with only the person's name."
    else:
        return f"Who is {child}'s father? Answer with only the person's name."


def get_reverse_prompt(parent: str, parent_type: str) -> str:
    """Generate reverse prompt: asking for child given parent."""
    return f"Name a child of {parent}. Answer with only the person's name."


def normalize_name(name: str) -> str:
    """Normalize name for comparison."""
    if not name:
        return ""
    name = " ".join(name.strip().lower().split())
    name = re.sub(r'^(mr\.|mrs\.|ms\.|dr\.)\s*', '', name)
    return name


def check_name_match(predicted: str, expected: str) -> bool:
    """Check if predicted name matches expected name."""
    pred_norm = normalize_name(predicted)
    exp_norm = normalize_name(expected)
    
    if not pred_norm or not exp_norm:
        return False
    
    if pred_norm == exp_norm:
        return True
    
    pred_words = pred_norm.split()
    exp_words = exp_norm.split()
    
    if len(pred_words) >= 1 and len(exp_words) >= 1:
        if pred_words[0] == exp_words[0]:
            return True
        if len(pred_words) >= 2 and len(exp_words) >= 2:
            if pred_words[-1] == exp_words[-1]:
                return True
    
    if exp_norm in pred_norm or pred_norm in exp_norm:
        return True
    
    return False


def extract_name_from_response(response: str) -> str:
    """Extract the name from model response."""
    if not response:
        return ""
    
    first_line = response.strip().split('\n')[0]
    
    prefixes = [
        "The answer is", "Answer:", "A:", 
        "The mother is", "The father is",
        "The child is", "The son is", "The daughter is",
        "It's", "It is", "That would be",
    ]
    for prefix in prefixes:
        if first_line.lower().startswith(prefix.lower()):
            first_line = first_line[len(prefix):].strip()
    
    first_line = first_line.rstrip('.,!?')
    
    words = first_line.split()
    if len(words) > 5:
        first_line = " ".join(words[:5])
    
    return first_line.strip()


class BatchModelEvaluator:
    def __init__(
        self,
        model_id: str,
        batch_size: int = 8,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        
        print(f"Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="left",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use device_map="auto" to spread model across all GPUs
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Get device for input tensors
        self.device = next(self.model.parameters()).device
        print(f"Model loaded, primary device: {self.device}")
        print(f"Batch size: {batch_size}")
    
    def format_prompts(self, prompts: List[str]) -> List[str]:
        """Format prompts for chat models."""
        formatted = []
        for prompt in prompts:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                try:
                    fmt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,  # For Qwen3
                    )
                except TypeError:
                    # Fallback for models that don't support enable_thinking
                    fmt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                formatted.append(fmt)
            else:
                formatted.append(prompt)
        return formatted
    
    def batch_generate(self, prompts: List[str], max_new_tokens: int = 32) -> List[str]:
        """Batch generate responses."""
        formatted = self.format_prompts(prompts)
        
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode responses
        responses = []
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].ne(self.tokenizer.pad_token_id).sum().item()
            response = self.tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True,
            )
            responses.append(response.strip())
        
        return responses
    
    def evaluate_batch(self, batch_data: List[dict]) -> List[dict]:
        """Evaluate a batch of parent-child pairs."""
        # Prepare all prompts
        forward_prompts = [
            get_forward_prompt(d["child"], d["parent_type"]) 
            for d in batch_data
        ]
        reverse_prompts = [
            get_reverse_prompt(d["parent"], d["parent_type"]) 
            for d in batch_data
        ]
        
        # Batch inference
        forward_responses = self.batch_generate(forward_prompts)
        reverse_responses = self.batch_generate(reverse_prompts)
        
        results = []
        for i, d in enumerate(batch_data):
            forward_name = extract_name_from_response(forward_responses[i])
            reverse_name = extract_name_from_response(reverse_responses[i])
            
            results.append({
                "child": d["child"],
                "parent": d["parent"],
                "parent_type": d["parent_type"],
                "forward_prompt": forward_prompts[i],
                "forward_response": forward_responses[i],
                "forward_extracted": forward_name,
                "forward_correct": check_name_match(forward_name, d["parent"]),
                "reverse_prompt": reverse_prompts[i],
                "reverse_response": reverse_responses[i],
                "reverse_extracted": reverse_name,
                "reverse_correct": check_name_match(reverse_name, d["child"]),
            })
        
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference (default: 16 for 4x32GB V100)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} parent-child pairs")
    
    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)
        print(f"Sampled {len(df)} pairs")
    
    # Initialize evaluator
    evaluator = BatchModelEvaluator(args.model_id, batch_size=args.batch_size)
    
    # Prepare data
    data_list = df.to_dict('records')
    
    # Batch evaluate
    all_results = []
    for i in tqdm(range(0, len(data_list), args.batch_size), desc="Evaluating batches"):
        batch = data_list[i:i + args.batch_size]
        results = evaluator.evaluate_batch(batch)
        all_results.extend(results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate statistics
    forward_acc = results_df["forward_correct"].mean()
    reverse_acc = results_df["reverse_correct"].mean()
    both_correct = (results_df["forward_correct"] & results_df["reverse_correct"]).mean()
    reversal_failure = (results_df["forward_correct"] & ~results_df["reverse_correct"]).mean()
    
    summary = {
        "model_id": args.model_id,
        "n_samples": len(results_df),
        "batch_size": args.batch_size,
        "forward_accuracy": float(forward_acc),
        "reverse_accuracy": float(reverse_acc),
        "both_correct": float(both_correct),
        "reversal_failure_rate": float(reversal_failure),
        "forward_only_rate": float((results_df["forward_correct"] & ~results_df["reverse_correct"]).sum() / len(results_df)),
        "reverse_only_rate": float((~results_df["forward_correct"] & results_df["reverse_correct"]).sum() / len(results_df)),
        "neither_correct": float((~results_df["forward_correct"] & ~results_df["reverse_correct"]).sum() / len(results_df)),
    }
    
    print("\n" + "=" * 60)
    print(f"Results for {args.model_id}")
    print("=" * 60)
    print(f"Total samples: {summary['n_samples']}")
    print(f"Batch size: {summary['batch_size']}")
    print(f"Forward accuracy (child->parent): {forward_acc:.1%}")
    print(f"Reverse accuracy (parent->child): {reverse_acc:.1%}")
    print(f"Both directions correct: {both_correct:.1%}")
    print(f"Reversal failure (forward OK, reverse FAIL): {reversal_failure:.1%}")
    print("=" * 60)
    
    results_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
