#!/bin/bash
# Run DeepSeek-R1 evaluation with proper max_new_tokens and then compare all models

set -e

cd /mnt/projects/reversal_curse

echo "=========================================="
echo "Step 1: Evaluate DeepSeek-R1-Distill-32B"
echo "=========================================="

# Activate conda properly
source /mnt/envs/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/envs/conda_envs/reversalcurse

# Run DeepSeek-R1 evaluation with larger max_new_tokens
python scripts/local_experiment2/eval_celebrity_reversals_deepseek_r1.py \
    --model_id /mnt/models/deepseek-r1-distill-32b \
    --output_dir runs/exp2/compare_largemodels/deepseek_r1_32b \
    --max_samples 200 \
    --max_new_tokens 1024 \
    --batch_size 2 \
    --seed 42

echo ""
echo "=========================================="
echo "Step 2: Compare All Models"
echo "=========================================="

# Run comparison script
python scripts/local_experiment2/aggregate_and_plot_largemodels.py \
    --exp_dir runs/exp2/compare_largemodels \
    --output_dir runs/exp2/compare_largemodels/compare

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
echo "Results saved to: runs/exp2/compare_largemodels/"
