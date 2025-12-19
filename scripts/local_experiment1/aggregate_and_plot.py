"""把三模型的 Experiment 1 评测结果汇总成对比表，并画一张对比图。

你要求：
- “训练完了保存明显结论，图表，对比等”。

本脚本负责最后一步：
1) 读取每个模型的 eval/summary.json
2) 提取每个测试文件的 accuracy
3) 输出：
   - compare.csv：每行一个 (model, split, accuracy)
   - compare_by_split.csv：每个 split 一行，三模型三列（便于读）
   - compare.png：简单柱状图（按 split 展示）

注意：
- 这里不“自动决定哪个模型更好”，只给可核对的数据与图。
- 图表使用 matplotlib 默认风格，避免引入额外依赖/主题。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs",
        type=str,
        required=True,
        help=(
            "逗号分隔：model_name=eval_summary_path。\n"
            "例如：qwen=outputs/exp1_batch/qwen3_4b/eval/summary.json,gemma=..."
        ),
    )
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    items: List[Dict[str, Any]] = []

    # 解析 runs
    # 格式：name=path,name=path
    for part in [x.strip() for x in args.runs.split(",") if x.strip()]:
        if "=" not in part:
            raise ValueError(f"Bad --runs entry: {part}")
        name, path = part.split("=", 1)
        name = name.strip()
        path = path.strip()

        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        results = summary.get("results", {})
        for split, info in results.items():
            items.append(
                {
                    "model": name,
                    "split": split,
                    "accuracy": info.get("accuracy", 0.0),
                    "n": info.get("n", 0),
                    "csv": info.get("csv", ""),
                }
            )

    df = pd.DataFrame(items)
    df.to_csv(os.path.join(args.out_dir, "compare.csv"), index=False)

    # pivot：每个 split 一行，每个模型一列
    pivot = df.pivot_table(index="split", columns="model", values="accuracy", aggfunc="first")
    pivot.to_csv(os.path.join(args.out_dir, "compare_by_split.csv"))

    # 画图：简单柱状图
    try:
        import matplotlib.pyplot as plt

        ax = pivot.plot(kind="bar", figsize=(12, 5))
        ax.set_ylabel("accuracy")
        ax.set_xlabel("split")
        ax.set_title("Experiment 1 (strict name-only) accuracy comparison")
        ax.legend(title="model")
        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, "compare.png")
        plt.savefig(fig_path, dpi=200)
        print(f"[done] wrote: {fig_path}")
    except Exception as e:
        print(f"[warn] failed to plot compare.png: {e}")

    print(f"[done] wrote: {os.path.join(args.out_dir, 'compare.csv')}")
    print(f"[done] wrote: {os.path.join(args.out_dir, 'compare_by_split.csv')}")


if __name__ == "__main__":
    main()
