#!/usr/bin/env python3
"""Aggregate results for large-model Experiment 2 runs and generate comparison plots.

Usage:
  python scripts/local_experiment2/aggregate_and_plot_largemodels.py \
    --exp_dir runs/exp2/compare_largemodels \
    --output_dir runs/exp2/compare_largemodels/compare

This script is intentionally tolerant to small schema differences in summary.json
(e.g., both_correct vs both_correct_rate).
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_NAMES = {
    "gemma3_27b": "Gemma3-27B",
    "qwen3_32b": "Qwen3-32B",
    "deepseek_r1_32b": "DeepSeek-R1-Distill-32B",
}


def _get(summary: dict, primary: str, fallbacks: list[str] | None = None):
    if primary in summary and summary[primary] is not None:
        return summary[primary]
    if fallbacks:
        for key in fallbacks:
            if key in summary and summary[key] is not None:
                return summary[key]
    return None


def load_results(exp_dir: str) -> pd.DataFrame:
    exp_path = Path(exp_dir)
    rows: list[dict] = []

    for model_dir in exp_path.iterdir():
        if not model_dir.is_dir() or model_dir.name == "compare":
            continue

        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        model_key = model_dir.name
        display_name = MODEL_NAMES.get(model_key, model_key)

        n_samples = _get(summary, "n_samples", ["total_samples"])
        forward_acc = _get(summary, "forward_accuracy", ["forward_acc", "forward"])
        reverse_acc = _get(summary, "reverse_accuracy", ["reverse_acc", "reverse"])
        both_rate = _get(summary, "both_correct_rate", ["both_correct", "both"])
        failure_rate = _get(summary, "reversal_failure_rate", ["forward_only_rate", "reversal_failure"])

        missing = [
            k
            for k, v in {
                "n_samples": n_samples,
                "forward_accuracy": forward_acc,
                "reverse_accuracy": reverse_acc,
                "both_correct_rate": both_rate,
                "reversal_failure_rate": failure_rate,
            }.items()
            if v is None
        ]
        if missing:
            raise KeyError(f"Missing keys {missing} in {summary_path}")

        rows.append(
            {
                "model": display_name,
                "model_key": model_key,
                "n_samples": int(n_samples),
                "forward_accuracy": float(forward_acc),
                "reverse_accuracy": float(reverse_acc),
                "both_correct_rate": float(both_rate),
                "reversal_failure_rate": float(failure_rate),
            }
        )

    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: Forward vs Reverse
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        df["forward_accuracy"] * 100,
        width,
        label="Forward (child->parent)",
        color="#4CAF50",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        df["reverse_accuracy"] * 100,
        width,
        label="Reverse (parent->child)",
        color="#F44336",
        edgecolor="black",
    )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        "Experiment 2 (Large Models): Reversal Curse in the Wild\n(Celebrity Parent-Child Relationships)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_forward_vs_reverse.png"), dpi=150)
    plt.close()

    # Figure 2: Breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    tmp = df.copy()
    tmp["forward_only"] = tmp["forward_accuracy"] - tmp["both_correct_rate"]
    tmp["reverse_only"] = tmp["reverse_accuracy"] - tmp["both_correct_rate"]
    tmp["neither"] = 1 - tmp["both_correct_rate"] - tmp["forward_only"] - tmp["reverse_only"]

    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9E9E9E"]
    labels = ["Both Correct", "Forward Only", "Reverse Only", "Neither"]

    bottom = np.zeros(len(tmp))
    for col, color, label in zip(
        ["both_correct_rate", "forward_only", "reverse_only", "neither"],
        colors,
        labels,
    ):
        values = tmp[col].values * 100
        ax.bar(tmp["model"], values, bottom=bottom, label=label, color=color, edgecolor="black")
        bottom += values

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        "Experiment 2 (Large Models): Breakdown of Reversal Performance",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_breakdown.png"), dpi=150)
    plt.close()

    # Figure 3: Reversal failure
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_fail = ["#E53935", "#FB8C00", "#43A047"]
    bars = ax.bar(
        tmp["model"],
        tmp["reversal_failure_rate"] * 100,
        color=colors_fail[: len(tmp)],
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_ylabel("Reversal Failure Rate (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        "Experiment 2 (Large Models): Reversal Failure Rate\n(Forward Correct but Reverse Wrong)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, max(tmp["reversal_failure_rate"] * 100) * 1.3 if len(tmp) else 100)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_reversal_failure.png"), dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="runs/exp2/compare_largemodels")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.exp_dir, "compare")

    df = load_results(args.exp_dir)
    if len(df) == 0:
        raise SystemExit(f"No results found under {args.exp_dir}")

    df = df.sort_values("model_key")
    os.makedirs(args.output_dir, exist_ok=True)

    df.to_csv(os.path.join(args.output_dir, "exp2_summary.csv"), index=False)
    plot_comparison(df, args.output_dir)

    print(f"Saved summary + plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
