#!/usr/bin/env python3
"""
Aggregate results from Experiment 2 and generate comparison plots.

Usage:
    python scripts/local_experiment2/aggregate_and_plot.py \
        --exp_dir runs/exp2 \
        --output_dir runs/exp2/compare
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Model display names
MODEL_NAMES = {
    "qwen3_4b": "Qwen3-4B",
    "gemma3_4b": "Gemma3-4B", 
    "phi3_5_mini_instruct": "Phi-3.5-mini",
}


def load_results(exp_dir: str) -> pd.DataFrame:
    """Load all model results from experiment directory."""
    exp_path = Path(exp_dir)
    all_results = []
    
    for model_dir in exp_path.iterdir():
        if not model_dir.is_dir() or model_dir.name == "compare":
            continue
        
        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            continue
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        model_key = model_dir.name
        display_name = MODEL_NAMES.get(model_key, model_key)
        
        all_results.append({
            "model": display_name,
            "model_key": model_key,
            "n_samples": summary["n_samples"],
            "forward_accuracy": summary["forward_accuracy"],
            "reverse_accuracy": summary["reverse_accuracy"],
            "both_correct_rate": summary["both_correct_rate"],
            "reversal_failure_rate": summary["reversal_failure_rate"],
        })
    
    return pd.DataFrame(all_results)


def plot_comparison(df: pd.DataFrame, output_dir: str):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure 1: Forward vs Reverse Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df["forward_accuracy"] * 100, width, 
                   label="Forward (child->parent)", color="#4CAF50", edgecolor="black")
    bars2 = ax.bar(x + width/2, df["reverse_accuracy"] * 100, width,
                   label="Reverse (parent->child)", color="#F44336", edgecolor="black")
    
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Experiment 2: Reversal Curse in the Wild\n(Celebrity Parent-Child Relationships)", 
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_forward_vs_reverse.png"), dpi=150)
    plt.close()
    
    # Figure 2: Stacked bar showing breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate categories
    df = df.copy()
    df["forward_only"] = df["forward_accuracy"] - df["both_correct_rate"]
    df["reverse_only"] = df["reverse_accuracy"] - df["both_correct_rate"]
    df["neither"] = 1 - df["forward_accuracy"] - df["reverse_only"]
    
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9E9E9E"]
    labels = ["Both Correct", "Forward Only", "Reverse Only", "Neither"]
    
    bottom = np.zeros(len(df))
    for i, (col, color, label) in enumerate(zip(
        ["both_correct_rate", "forward_only", "reverse_only", "neither"],
        colors, labels
    )):
        values = df[col].values * 100
        ax.bar(df["model"], values, bottom=bottom, label=label, color=color, edgecolor="black")
        bottom += values
    
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Experiment 2: Breakdown of Reversal Performance", 
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_breakdown.png"), dpi=150)
    plt.close()
    
    # Figure 3: Reversal Failure Rate (key metric)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors_fail = ["#E53935", "#FB8C00", "#43A047"]
    bars = ax.bar(df["model"], df["reversal_failure_rate"] * 100, 
                  color=colors_fail[:len(df)], edgecolor="black", linewidth=1.5)
    
    ax.set_ylabel("Reversal Failure Rate (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Experiment 2: Reversal Failure Rate\n(Forward Correct but Reverse Wrong)", 
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(df["reversal_failure_rate"] * 100) * 1.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_reversal_failure.png"), dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def print_summary_table(df: pd.DataFrame):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: REVERSAL CURSE IN THE WILD - SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Model':<20} {'Forward':<12} {'Reverse':<12} {'Both':<12} {'Failure':<12}")
    print("-" * 68)
    
    for _, row in df.iterrows():
        print(f"{row['model']:<20} "
              f"{row['forward_accuracy']*100:>6.1f}%     "
              f"{row['reverse_accuracy']*100:>6.1f}%     "
              f"{row['both_correct_rate']*100:>6.1f}%     "
              f"{row['reversal_failure_rate']*100:>6.1f}%")
    
    print("-" * 68)
    print("\nLegend:")
    print("  Forward: Accuracy when asking 'Who is X's parent?'")
    print("  Reverse: Accuracy when asking 'Name a child of Y'")
    print("  Both: Both directions correct")
    print("  Failure: Forward correct but reverse wrong (Reversal Curse!)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="runs/exp2")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.exp_dir, "compare")
    
    # Load results
    df = load_results(args.exp_dir)
    
    if len(df) == 0:
        print("No results found!")
        return
    
    # Sort by model name for consistent ordering
    df = df.sort_values("model_key")
    
    # Print summary
    print_summary_table(df)
    
    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "exp2_summary.csv"), index=False)
    
    # Generate plots
    plot_comparison(df, args.output_dir)


if __name__ == "__main__":
    main()
