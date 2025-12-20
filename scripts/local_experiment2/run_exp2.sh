#!/bin/bash
#
# Experiment 2: The Reversal Curse in the Wild
# Test celebrity parent-child reversals on local models
#
# Usage:
#   bash scripts/local_experiment2/run_exp2.sh
#
# This script:
# 1. Evaluates each model on celebrity parent-child pairs
# 2. Aggregates results and generates comparison plots
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_ROOT="runs/exp2"
MAX_SAMPLES=200  # Use 200 samples for reasonable runtime; set to empty for all

# Model configurations (same as experiment 1)
declare -A MODELS=(
    ["qwen3_4b"]="Qwen/Qwen3-4B"
    ["gemma3_4b"]="google/gemma-3-4b-it"
    ["phi3_5_mini_instruct"]="microsoft/Phi-3.5-mini-instruct"
)

echo "============================================================"
echo "Experiment 2: The Reversal Curse in the Wild"
echo "Testing celebrity parent-child reversals"
echo "============================================================"
echo ""
echo "Models: ${!MODELS[@]}"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Output: $OUTPUT_ROOT"
echo ""

# Create output directory
mkdir -p "$OUTPUT_ROOT"

# Run evaluation for each model
for model_key in "${!MODELS[@]}"; do
    model_id="${MODELS[$model_key]}"
    output_dir="$OUTPUT_ROOT/$model_key"
    
    echo ""
    echo "============================================================"
    echo "Evaluating: $model_key ($model_id)"
    echo "============================================================"
    
    # Skip if already done
    if [ -f "$output_dir/summary.json" ]; then
        echo "Already completed, skipping. (Delete $output_dir to re-run)"
        continue
    fi
    
    # Build command
    cmd="python scripts/local_experiment2/eval_celebrity_reversals.py"
    cmd="$cmd --model_id $model_id"
    cmd="$cmd --output_dir $output_dir"
    if [ -n "$MAX_SAMPLES" ]; then
        cmd="$cmd --max_samples $MAX_SAMPLES"
    fi
    
    echo "Running: $cmd"
    eval $cmd
    
    echo ""
    echo "Completed: $model_key"
done

# Aggregate and plot
echo ""
echo "============================================================"
echo "Aggregating results and generating plots..."
echo "============================================================"

python scripts/local_experiment2/aggregate_and_plot.py \
    --exp_dir "$OUTPUT_ROOT" \
    --output_dir "$OUTPUT_ROOT/compare"

echo ""
echo "============================================================"
echo "Experiment 2 Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Per-model results: $OUTPUT_ROOT/<model>/results.csv"
echo "  - Summary: $OUTPUT_ROOT/compare/exp2_summary.csv"
echo "  - Plots: $OUTPUT_ROOT/compare/*.png"
echo ""
