#!/bin/bash
#
# Download large models for Experiment 2
# Models stored in /mnt/models/ for reuse
#

set -e

export HF_HUB_ENABLE_HF_TRANSFER=1
MODELS_DIR="/mnt/models"

source /mnt/envs/anaconda3/etc/profile.d/conda.sh
conda activate reversalcurse
pip install hf_transfer -q 2>/dev/null || true

download_model() {
    local model_id=$1
    local local_dir=$2
    local model_name=$3
    
    echo ""
    echo "============================================================"
    echo "Downloading: $model_name"
    echo "Model ID: $model_id"
    echo "Target: $local_dir"
    echo "============================================================"
    
    if [ -f "$local_dir/config.json" ]; then
        echo "Model already exists, skipping..."
        return 0
    fi
    
    huggingface-cli download "$model_id" \
        --local-dir "$local_dir" \
        --local-dir-use-symlinks False \
        --resume-download
    
    echo "Done: $model_name"
}

MODEL="${1:-all}"

case "$MODEL" in
    "deepseek") download_model "deepseek-ai/DeepSeek-V3" "$MODELS_DIR/deepseek-v3" "DeepSeek-V3" ;;
    "qwen") download_model "Qwen/Qwen3-235B-A22B" "$MODELS_DIR/qwen3-235b" "Qwen3-235B-A22B" ;;
    "gemma") 
        echo "Note: Gemma requires access approval at https://huggingface.co/google/gemma-3-27b-it"
        download_model "google/gemma-3-27b-it" "$MODELS_DIR/gemma3-27b" "Gemma-3-27B-it" 
        ;;
    "all")
        echo "Downloading all 3 models (~1.1TB total)..."
        download_model "google/gemma-3-27b-it" "$MODELS_DIR/gemma3-27b" "Gemma-3-27B-it (51GB)"
        download_model "Qwen/Qwen3-235B-A22B" "$MODELS_DIR/qwen3-235b" "Qwen3-235B-A22B (438GB)"
        download_model "deepseek-ai/DeepSeek-V3" "$MODELS_DIR/deepseek-v3" "DeepSeek-V3 (641GB)"
        ;;
    *) echo "Usage: $0 [all|deepseek|qwen|gemma]" ;;
esac

echo "Models stored in $MODELS_DIR/"
