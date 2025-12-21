"""Local Experiment 3: 结果聚合和绘图

用法：
    python scripts/local_experiment3/aggregate_and_plot.py \
        --results_root outputs/local_exp3 \
        --out_dir outputs/local_exp3/aggregated
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=str, default="outputs/local_exp3")
    p.add_argument("--out_dir", type=str, default="outputs/local_exp3/aggregated")
    return p.parse_args()


def load_summaries(results_root: str) -> List[Dict[str, Any]]:
    """加载所有模型的评测结果。"""
    summaries = []
    for model_name in os.listdir(results_root):
        model_dir = os.path.join(results_root, model_name)
        if not os.path.isdir(model_dir):
            continue
        
        summary_path = os.path.join(model_dir, "eval", "summary.json")
        if not os.path.exists(summary_path):
            print(f"[skip] {summary_path} not found")
            continue
        
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["model_name"] = model_name
        summaries.append(summary)
    
    return summaries


def create_comparison_table(summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    """创建模型对比表格。"""
    rows = []
    for s in summaries:
        row = {
            "Model": s["model_name"],
            "Base Model ID": s.get("base_model_id", "N/A"),
        }
        for key, result in s.get("results", {}).items():
            row[f"{key}_acc"] = result.get("accuracy", 0.0)
            row[f"{key}_n"] = result.get("n", 0)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def plot_comparison(summaries: List[Dict[str, Any]], out_path: str) -> None:
    """绘制模型对比图。"""
    model_names = []
    realized_accs = []
    unrealized_accs = []
    
    for s in summaries:
        model_names.append(s["model_name"])
        results = s.get("results", {})
        realized_accs.append(results.get("realized_examples", {}).get("accuracy", 0.0) * 100)
        unrealized_accs.append(results.get("unrealized_examples", {}).get("accuracy", 0.0) * 100)
    
    x = range(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], realized_accs, width, label='Realized Examples', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], unrealized_accs, width, label='Unrealized Examples', color='coral')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Experiment 3: Reversing Instructions\nRealized vs Unrealized Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] saved: {out_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    summaries = load_summaries(args.results_root)
    if not summaries:
        print("[warn] No summaries found!")
        return
    
    # 创建对比表格
    df = create_comparison_table(summaries)
    csv_path = os.path.join(args.out_dir, "comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"[table] saved: {csv_path}")
    print(df.to_string(index=False))
    
    # 绘制对比图
    plot_path = os.path.join(args.out_dir, "comparison.png")
    plot_comparison(summaries, plot_path)
    
    # 保存汇总 JSON
    aggregated = {
        "summaries": summaries,
        "comparison_csv": csv_path,
        "comparison_plot": plot_path,
    }
    json_path = os.path.join(args.out_dir, "aggregated.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"[json] saved: {json_path}")


if __name__ == "__main__":
    main()
