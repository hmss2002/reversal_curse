#!/usr/bin/env python3
"""
Experiment 2: Evaluation Script for DeepSeek-R1 Reasoning Models

DeepSeek-R1 models output a chain-of-thought (thinking) before the final answer.
The thinking is wrapped in <think>...</think> tags.

Key differences from standard evaluation:
1. Much larger max_new_tokens to allow for thinking + answer
2. Extract the final answer AFTER the </think> tag
3. Option to disable thinking mode (if supported)

Usage:
    python scripts/local_experiment2/eval_celebrity_reversals_deepseek_r1.py \
        --model_id /mnt/models/deepseek-r1-distill-32b \
        --output_dir runs/exp2/compare_largemodels/deepseek_r1_32b \
        --max_samples 200 \
        --max_new_tokens 512
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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
    
    # Match first or last name
    if len(pred_words) >= 1 and len(exp_words) >= 1:
        if pred_words[0] == exp_words[0]:
            return True
        if len(pred_words) >= 2 and len(exp_words) >= 2:
            if pred_words[-1] == exp_words[-1]:
                return True
    
    # Substring match
    if exp_norm in pred_norm or pred_norm in exp_norm:
        return True
    
    return False


def extract_answer_from_reasoning_response(response: str) -> Tuple[str, str]:
    """
    Extract the final answer from a reasoning model response.
    
    DeepSeek-R1 outputs are in the format:
    <think>
    ... thinking process ...
    </think>
    [Final Answer]
    
    Returns: (thinking_content, final_answer)
    """
    if not response:
        return "", ""
    
    # Check for <think>...</think> pattern
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, response, re.DOTALL)
    
    if think_match:
        thinking = think_match.group(1).strip()
        # Get everything after </think>
        after_think = response[think_match.end():].strip()
        final_answer = extract_name_from_text(after_think)
        return thinking, final_answer
    
    # No <think> tags found - might be thinking without tags
    # Check if response starts with typical thinking phrases
    thinking_starters = [
        "Okay, so", "Let me think", "I need to", "Hmm,", "Well,",
        "First,", "Let's see", "Alright,", "So,", "I'm going to"
    ]
    
    for starter in thinking_starters:
        if response.strip().startswith(starter):
            # This looks like thinking content without proper tags
            # Try to find a conclusion or final answer
            conclusion_patterns = [
                r'(?:The answer is|So the answer is|Therefore,|Thus,|So,)\s*(.+?)(?:\.|$)',
                r'(?:is|was|would be)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            ]
            for pattern in conclusion_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    return response, extract_name_from_text(match.group(1))
            
            # No clear conclusion found - response was truncated
            return response, ""
    
    # Normal response without thinking
    return "", extract_name_from_text(response)


def extract_name_from_text(text: str) -> str:
    """Extract a name from text."""
    if not text:
        return ""
    
    # Take first line
    first_line = text.strip().split('\n')[0]
    
    # Remove common prefixes
    prefixes = [
        "The answer is", "Answer:", "A:", 
        "The mother is", "The father is",
        "The child is", "The son is", "The daughter is",
        "It's", "It is", "That would be", "That's",
        "His mother is", "Her mother is", "His father is", "Her father is",
    ]
    for prefix in prefixes:
        if first_line.lower().startswith(prefix.lower()):
            first_line = first_line[len(prefix):].strip()
    
    # Remove quotes
    first_line = first_line.strip('"\'')
    
    # Remove trailing punctuation
    first_line = first_line.rstrip('.,!?')
    
    # Take first few words (name is usually 2-4 words)
    words = first_line.split()
    if len(words) > 5:
        first_line = " ".join(words[:5])
    
    return first_line.strip()


class DeepSeekR1Evaluator:
    """Evaluator specifically designed for DeepSeek-R1 reasoning models."""
    
    def __init__(
        self,
        model_id: str,
        batch_size: int = 4,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        
        print(f"Loading DeepSeek-R1 model: {model_id}")
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
        
        self.device = next(self.model.parameters()).device
        print(f"Model loaded, primary device: {self.device}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Batch size: {batch_size}")
    
    def format_prompt(self, prompt: str) -> str:
        """Format prompt using chat template."""
        messages = [{"role": "user", "content": prompt}]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return formatted
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}")
        
        return prompt
    
    def generate_single(self, prompt: str) -> str:
        """Generate response for a single prompt."""
        formatted = self.format_prompt(prompt)
        
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        input_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )
        
        return response.strip()
    
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Batch generate responses."""
        formatted = [self.format_prompt(p) for p in prompts]
        
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
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
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
        forward_prompts = [
            get_forward_prompt(d["child"], d["parent_type"]) 
            for d in batch_data
        ]
        reverse_prompts = [
            get_reverse_prompt(d["parent"], d["parent_type"]) 
            for d in batch_data
        ]
        
        # Generate responses
        forward_responses = self.batch_generate(forward_prompts)
        reverse_responses = self.batch_generate(reverse_prompts)
        
        results = []
        for i, d in enumerate(batch_data):
            # Extract answers from reasoning responses
            fwd_thinking, fwd_answer = extract_answer_from_reasoning_response(forward_responses[i])
            rev_thinking, rev_answer = extract_answer_from_reasoning_response(reverse_responses[i])
            
            results.append({
                "child": d["child"],
                "parent": d["parent"],
                "parent_type": d["parent_type"],
                "forward_prompt": forward_prompts[i],
                "forward_response": forward_responses[i],
                "forward_thinking": fwd_thinking[:500] if fwd_thinking else "",  # Truncate for CSV
                "forward_extracted": fwd_answer,
                "forward_correct": check_name_match(fwd_answer, d["parent"]),
                "reverse_prompt": reverse_prompts[i],
                "reverse_response": reverse_responses[i],
                "reverse_thinking": rev_thinking[:500] if rev_thinking else "",  # Truncate for CSV
                "reverse_extracted": rev_answer,
                "reverse_correct": check_name_match(rev_answer, d["child"]),
            })
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek-R1 reasoning models on celebrity reversal task"
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="Path to DeepSeek-R1 model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (smaller for larger models)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens for generation (needs to be large for thinking)")
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
    evaluator = DeepSeekR1Evaluator(
        args.model_id,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Prepare data
    data_list = df.to_dict('records')
    
    # Batch evaluate
    all_results = []
    for i in tqdm(range(0, len(data_list), args.batch_size), desc="Evaluating"):
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
    
    # Count how many responses had truncated thinking (no final answer)
    truncated_forward = (results_df["forward_extracted"] == "").sum()
    truncated_reverse = (results_df["reverse_extracted"] == "").sum()
    
    summary = {
        "model_id": args.model_id,
        "n_samples": len(results_df),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "forward_accuracy": float(forward_acc),
        "reverse_accuracy": float(reverse_acc),
        "both_correct": float(both_correct),
        "reversal_failure_rate": float(reversal_failure),
        "forward_only_rate": float((results_df["forward_correct"] & ~results_df["reverse_correct"]).sum() / len(results_df)),
        "reverse_only_rate": float((~results_df["forward_correct"] & results_df["reverse_correct"]).sum() / len(results_df)),
        "neither_correct": float((~results_df["forward_correct"] & ~results_df["reverse_correct"]).sum() / len(results_df)),
        "truncated_forward": int(truncated_forward),
        "truncated_reverse": int(truncated_reverse),
    }
    
    print("\n" + "=" * 60)
    print(f"Results for {args.model_id}")
    print("=" * 60)
    print(f"Total samples: {summary['n_samples']}")
    print(f"Max new tokens: {summary['max_new_tokens']}")
    print(f"Forward accuracy (child->parent): {forward_acc:.1%}")
    print(f"Reverse accuracy (parent->child): {reverse_acc:.1%}")
    print(f"Both directions correct: {both_correct:.1%}")
    print(f"Reversal failure (forward OK, reverse FAIL): {reversal_failure:.1%}")
    print(f"Truncated responses (forward): {truncated_forward}")
    print(f"Truncated responses (reverse): {truncated_reverse}")
    print("=" * 60)
    
    # Save results
    results_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")
    
    # Print some examples with thinking process
    print("\n--- Sample Results ---")
    sample_df = results_df.sample(min(3, len(results_df)), random_state=args.seed)
    for _, row in sample_df.iterrows():
        fwd_mark = "OK" if row['forward_correct'] else "X"
        rev_mark = "OK" if row['reverse_correct'] else "X"
        print(f"\nChild: {row['child']}, Parent: {row['parent']} ({row['parent_type']})")
        print(f"  Forward extracted: '{row['forward_extracted']}' -> [{fwd_mark}]")
        print(f"  Reverse extracted: '{row['reverse_extracted']}' -> [{rev_mark}]")
        if row['forward_thinking']:
            print(f"  (Had thinking content: {len(row['forward_thinking'])} chars)")


if __name__ == "__main__":
    main()
