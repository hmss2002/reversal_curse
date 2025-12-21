#!/bin/bash
# 批量评估三个大模型 (27B-32B) 并保存结果到 compare_largemodels

set -e  # 遇到错误立即退出

# 激活 conda 环境
source /mnt/envs/anaconda3/etc/profile.d/conda.sh
conda activate reversalcurse

# 进入项目目录
cd /mnt/projects/reversal_curse

# 输出目录
OUTPUT_BASE="runs/exp2/compare_largemodels"
mkdir -p "$OUTPUT_BASE"

# 评估样本数 (200个名人对，400条测试)
MAX_SAMPLES=200

echo "========================================"
echo "开始评估大模型 (27B-32B)"
echo "使用 4x V100 32GB GPUs (device_map=auto)"
echo "评估样本数: $MAX_SAMPLES"
echo "========================================"
echo ""

# 1. Gemma-3-27B-it (~52GB)
echo "[1/3] 评估 Gemma-3-27B-it..."
echo "预计时间: ~15-20分钟"
echo "----------------------------------------"
python scripts/local_experiment2/eval_celebrity_reversals.py \
    --model_id /mnt/models/gemma3-27b \
    --output_dir "$OUTPUT_BASE/gemma3_27b" \
    --max_samples $MAX_SAMPLES

echo ""
echo "✓ Gemma-3-27B 完成"
echo ""

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
sleep 5

# 2. Qwen3-32B (~62GB)
echo "[2/3] 评估 Qwen3-32B..."
echo "预计时间: ~15-20分钟"
echo "----------------------------------------"
python scripts/local_experiment2/eval_celebrity_reversals.py \
    --model_id /mnt/models/qwen3-32b \
    --output_dir "$OUTPUT_BASE/qwen3_32b" \
    --max_samples $MAX_SAMPLES

echo ""
echo "✓ Qwen3-32B 完成"
echo ""

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
sleep 5

# 3. DeepSeek-R1-Distill-Qwen-32B (~62GB)
echo "[3/3] 评估 DeepSeek-R1-Distill-Qwen-32B..."
echo "预计时间: ~15-20分钟"
echo "----------------------------------------"
python scripts/local_experiment2/eval_celebrity_reversals.py \
    --model_id /mnt/models/deepseek-r1-distill-32b \
    --output_dir "$OUTPUT_BASE/deepseek_r1_32b" \
    --max_samples $MAX_SAMPLES

echo ""
echo "✓ DeepSeek-R1-Distill-32B 完成"
echo ""

# 4. 汇总结果
echo "========================================"
echo "生成汇总报告..."
echo "========================================"
python scripts/local_experiment2/aggregate_and_plot.py \
    --results_dir "$OUTPUT_BASE"

echo ""
echo "========================================"
echo "所有评估完成！"
echo "结果保存在: $OUTPUT_BASE/"
echo "========================================"
echo ""
echo "查看结果:"
echo "  - 汇总数据: cat $OUTPUT_BASE/*/summary.json"
echo "  - 对比分析: ls $OUTPUT_BASE/compare/"
