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
import csv
import json
import os
from typing import Any, Dict, List, Tuple


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
            # 用户要求：不再评测/汇总 both_prompts_test
            if split == "both_prompts_test":
                continue
            items.append(
                {
                    "model": name,
                    "split": split,
                    "accuracy": info.get("accuracy", 0.0),
                    "n": info.get("n", 0),
                    "csv": info.get("csv", ""),
                }
            )

    # 1) compare.csv
    compare_csv_path = os.path.join(args.out_dir, "compare.csv")
    with open(compare_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "split", "accuracy", "n", "csv"])
        for row in items:
            w.writerow(
                [
                    row.get("model", ""),
                    row.get("split", ""),
                    row.get("accuracy", 0.0),
                    row.get("n", 0),
                    row.get("csv", ""),
                ]
            )

    # 2) compare_by_split.csv（手写 pivot）
    models = sorted({str(x.get("model", "")) for x in items if x.get("model")})
    splits = sorted({str(x.get("split", "")) for x in items if x.get("split")})

    pivot: Dict[Tuple[str, str], Any] = {}
    for row in items:
        model = str(row.get("model", ""))
        split = str(row.get("split", ""))
        pivot[(split, model)] = row.get("accuracy", 0.0)

    compare_by_split_path = os.path.join(args.out_dir, "compare_by_split.csv")
    with open(compare_by_split_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", *models])
        for split in splits:
            row = [split]
            for model in models:
                row.append(pivot.get((split, model), ""))
            w.writerow(row)

    # 画图：简单柱状图
    try:
        import matplotlib.pyplot as plt

        if splits and models:
            x = list(range(len(splits)))
            width = 0.8 / max(1, len(models))

            fig, ax = plt.subplots(figsize=(12, 5))
            for i, model in enumerate(models):
                ys = [pivot.get((split, model), 0.0) or 0.0 for split in splits]
                xs = [v + (i - (len(models) - 1) / 2.0) * width for v in x]
                ax.bar(xs, ys, width=width, label=model)

            ax.set_xticks(x)
            ax.set_xticklabels(splits, rotation=30, ha="right")
            ax.set_ylabel("accuracy")
            ax.set_xlabel("split")
            ax.set_title("Experiment 1 accuracy comparison")
            ax.legend(title="model")
            plt.tight_layout()

            fig_path = os.path.join(args.out_dir, "compare.png")
            plt.savefig(fig_path, dpi=200)
            print(f"[done] wrote: {fig_path}")
        else:
            print("[warn] no data to plot compare.png")
    except Exception as e:
        print(f"[warn] failed to plot compare.png: {e}")

    print(f"[done] wrote: {compare_csv_path}")
    print(f"[done] wrote: {compare_by_split_path}")


if __name__ == "__main__":
    main()
