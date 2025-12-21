#!/usr/bin/env python3
"""聚合 Experiment 3 (10ep, r64) 的结果"""

import json
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 结果目录
RESULTS = {
    "Qwen3-4B": "runs/exp3/qwen3_4b_10ep_r64/eval/summary.json",
    "Phi-3.5-mini": "runs/exp3/phi3_5_mini_10ep_r64/eval/summary.json",
    "Gemma3-4B": "runs/exp3/gemma3_4b_10ep_r64/eval/summary.json",
}

def main():
    os.chdir("/mnt/projects/reversal_curse")
    
    rows = []
    for model_name, path in RESULTS.items():
        with open(path) as f:
            data = json.load(f)
        
        realized_acc = data["results"]["realized_examples"]["accuracy"]
        unrealized_acc = data["results"]["unrealized_examples"]["accuracy"]
        
        rows.append({
            "Model": model_name,
            "Realized Acc (%)": realized_acc * 100,
            "Unrealized Acc (%)": unrealized_acc * 100,
            "Gap (%)": (realized_acc - unrealized_acc) * 100,
        })
    
    df = pd.DataFrame(rows)
    print("\n" + "="*60)
    print("Experiment 3 Results (10 epochs, LoRA r=64)")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # 保存 CSV
    out_dir = Path("runs/exp3/aggregated_10ep_r64")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    print(f"\nSaved to: {out_dir / 'results.csv'}")
    
    # 绘制图表
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], df["Realized Acc (%)"], width, label="Realized", color="#4CAF50")
    bars2 = ax.bar([i + width/2 for i in x], df["Unrealized Acc (%)"], width, label="Unrealized", color="#F44336")
    
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Experiment 3: Reversal Curse (10 epochs, LoRA r=64)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"])
    ax.legend()
    ax.set_ylim(0, 100)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(out_dir / "exp3_comparison.png", dpi=150)
    print(f"Saved plot to: {out_dir / 'exp3_comparison.png'}")


if __name__ == "__main__":
    main()