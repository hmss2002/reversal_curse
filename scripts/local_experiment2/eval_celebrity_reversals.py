#!/usr/bin/env python3
"""
Experiment 2: The Reversal Curse in the Wild (Local Model Version)

Test whether pre-trained models can reverse celebrity parent-child relationships.
- Forward: "Tom Cruise's mother is?" -> "Mary Lee Pfeiffer"
- Reverse: "Mary Lee Pfeiffer's son is?" -> "Tom Cruise"

Usage:
    python scripts/local_experiment2/eval_celebrity_reversals.py \
        --model_id Qwen/Qwen3-4B \
        --output_dir runs/exp2/qwen3_4b \
        --max_samples 200
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

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
    else:  # father
        return f"Who is {child}'s father? Answer with only the person's name."


def get_reverse_prompt(parent: str, parent_type: str) -> str:
    """Generate reverse prompt: asking for child given parent."""
    if parent_type == "mother":
        return f"Name a child of {parent}. Answer with only the person's name."
    else:  # father
        return f"Name a child of {parent}. Answer with only the person's name."


def normalize_name(name: str) -> str:
    """Normalize name for comparison."""
    if not name:
        return ""
    # Remove extra whitespace, lowercase
    name = " ".join(name.strip().lower().split())
    # Remove common prefixes/suffixes
    name = re.sub(r'^(mr\.|mrs\.|ms\.|dr\.)\s*', '', name)
    return name


def check_name_match(predicted: str, expected: str) -> bool:
    """Check if predicted name matches expected name."""
    pred_norm = normalize_name(predicted)
    exp_norm = normalize_name(expected)
    
    if not pred_norm or not exp_norm:
        return False
    
    # Exact match
    if pred_norm == exp_norm:
        return True
    
    # Check if expected is contained in predicted (first few words)
    pred_words = pred_norm.split()
    exp_words = exp_norm.split()
    
    # Match first name or last name
    if len(pred_words) >= 1 and len(exp_words) >= 1:
        # First name match
        if pred_words[0] == exp_words[0]:
            return True
        # Last name match (if available)
        if len(pred_words) >= 2 and len(exp_words) >= 2:
            if pred_words[-1] == exp_words[-1]:
                return True
    
    # Check substring containment
    if exp_norm in pred_norm or pred_norm in exp_norm:
        return True
    
    return False


def extract_name_from_response(response: str) -> str:
    """Extract the name from model response."""
    if not response:
        return ""
    
    # Take first line
    first_line = response.strip().split('\n')[0]
    
    # Remove common prefixes
    prefixes = [
        "The answer is", "Answer:", "A:", 
        "The mother is", "The father is",
        "The child is", "The son is", "The daughter is",
        "It's", "It is", "That would be",
    ]
    for prefix in prefixes:
        if first_line.lower().startswith(prefix.lower()):
            first_line = first_line[len(prefix):].strip()
    
    # Remove trailing punctuation
    first_line = first_line.rstrip('.,!?')
    
    # Take first few words (name is usually 2-4 words)
    words = first_line.split()
    if len(words) > 5:
        first_line = " ".join(words[:5])
    
    return first_line.strip()


class LocalModelEvaluator:
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_id = model_id
        self.device = device
        
        print(f"Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="left",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        """Generate response for a prompt."""
        # Format as chat if model supports it
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            try:
                # Disable thinking mode for Qwen3
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # Important for Qwen3
                )
            except Exception:
                formatted = prompt
        else:
            formatted = prompt
        
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )
        
        return response.strip()
    
    def evaluate_pair(
        self,
        child: str,
        parent: str,
        parent_type: str,
    ) -> dict:
        """Evaluate a single parent-child pair in both directions."""
        # Forward: child -> parent
        forward_prompt = get_forward_prompt(child, parent_type)
        forward_response = self.generate(forward_prompt)
        forward_name = extract_name_from_response(forward_response)
        forward_correct = check_name_match(forward_name, parent)
        
        # Reverse: parent -> child
        reverse_prompt = get_reverse_prompt(parent, parent_type)
        reverse_response = self.generate(reverse_prompt)
        reverse_name = extract_name_from_response(reverse_response)
        reverse_correct = check_name_match(reverse_name, child)
        
        return {
            "child": child,
            "parent": parent,
            "parent_type": parent_type,
            "forward_prompt": forward_prompt,
            "forward_response": forward_response,
            "forward_extracted": forward_name,
            "forward_correct": forward_correct,
            "reverse_prompt": reverse_prompt,
            "reverse_response": reverse_response,
            "reverse_extracted": reverse_name,
            "reverse_correct": reverse_correct,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to evaluate (None = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} parent-child pairs")
    
    # Sample if needed
    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)
        print(f"Sampled {len(df)} pairs")
    
    # Initialize evaluator
    evaluator = LocalModelEvaluator(args.model_id)
    
    # Evaluate all pairs
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        result = evaluator.evaluate_pair(
            child=row["child"],
            parent=row["parent"],
            parent_type=row["parent_type"],
        )
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    forward_acc = results_df["forward_correct"].mean()
    reverse_acc = results_df["reverse_correct"].mean()
    
    # Count reversible pairs (both directions correct)
    both_correct = (results_df["forward_correct"] & results_df["reverse_correct"]).mean()
    
    # Count pairs where forward is correct but reverse is wrong (reversal failure)
    reversal_failure = (results_df["forward_correct"] & ~results_df["reverse_correct"]).mean()
    
    # Summary
    summary = {
        "model_id": args.model_id,
        "n_samples": len(results_df),
        "forward_accuracy": forward_acc,
        "reverse_accuracy": reverse_acc,
        "both_correct": both_correct,
        "reversal_failure_rate": reversal_failure,
        "forward_only_rate": (results_df["forward_correct"] & ~results_df["reverse_correct"]).sum() / len(results_df),
        "reverse_only_rate": (~results_df["forward_correct"] & results_df["reverse_correct"]).sum() / len(results_df),
        "neither_correct": (~results_df["forward_correct"] & ~results_df["reverse_correct"]).sum() / len(results_df),
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Results for {args.model_id}")
    print("=" * 60)
    print(f"Total samples: {summary['n_samples']}")
    print(f"Forward accuracy (child->parent): {forward_acc:.1%}")
    print(f"Reverse accuracy (parent->child): {reverse_acc:.1%}")
    print(f"Both directions correct: {both_correct:.1%}")
    print(f"Reversal failure (forward OK, reverse FAIL): {reversal_failure:.1%}")
    print("=" * 60)
    
    # Save results
    results_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")
    
    # Print some examples
    print("\n--- Sample Results ---")
    sample_df = results_df.sample(min(5, len(results_df)), random_state=args.seed)
    for _, row in sample_df.iterrows():
        fwd_mark = "OK" if row['forward_correct'] else "X"
        rev_mark = "OK" if row['reverse_correct'] else "X"
        print(f"\nChild: {row['child']}, Parent: {row['parent']} ({row['parent_type']})")
        print(f"  Forward: '{row['forward_extracted']}' -> [{fwd_mark}]")
        print(f"  Reverse: '{row['reverse_extracted']}' -> [{rev_mark}]")


if __name__ == "__main__":
    main()
