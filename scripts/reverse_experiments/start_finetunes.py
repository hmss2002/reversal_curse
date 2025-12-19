"""
Start finetunes for reverse experiments.
cd /mnt/projects/reversal_curse && PYTHONPATH=. python scripts/reverse_experiments/start_finetunes.py --models gpt-4o-mini,gpt-4.1-mini,gpt-4.1-nano --learning_rate_multiplier 0.2 --batch_size 1 --n_epochs 1 --num_finetunes 1
"""

import argparse
import os

from src.openai_finetune import start_finetunes
from src.tasks.reverse_experiments.reverse_task import REVERSE_DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser()

    # 兼容旧参数：只指定一个模型名
    parser.add_argument("--model_name", type=str, default="ada")
    # 新参数：一次性指定多个模型，使用英文逗号分隔（推荐）
    # 例如：--models gpt-4o-mini,gpt-4.1-mini,gpt-4.1-nano
    # 说明：你提到的“gpt4omini”通常对应 OpenAI 的模型名 gpt-4o-mini（中间有连字符）。
    parser.add_argument(
        "--models",
        type=str,
        default="gpt-4o-mini,gpt-4.1-mini,gpt-4.1-nano",
        help="Comma-separated list of model names to finetune.",
    )
    parser.add_argument("--learning_rate_multiplier", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--dataset_name", type=str, default="june_version_7921032488")
    parser.add_argument("--num_finetunes", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 如果传了 --models，就按列表逐个 finetune；否则退回用 --model_name（兼容旧用法）。
    models_arg = (args.models or "").strip()
    model_names = [m.strip() for m in models_arg.split(",") if m.strip()] if models_arg else []
    if not model_names:
        model_names = [args.model_name]

    # 只需要运行一次这个脚本：它会依次对每个模型提交 finetune。
    # 提交的命令数量 = len(model_names) * args.num_finetunes。
    for model_name in model_names:
        start_finetunes(
            model_name,
            args.learning_rate_multiplier,
            args.batch_size,
            args.n_epochs,
            args.dataset_name,
            args.num_finetunes,
            os.path.join(REVERSE_DATA_DIR, args.dataset_name),
            "all_prompts_train.jsonl",
            "validation_prompts.jsonl",
        )
