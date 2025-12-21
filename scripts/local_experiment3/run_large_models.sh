#!/bin/bash
# 使用 FSDP 训练大模型 (27B-32B)

set -e

cd /mnt/projects/reversal_curse
source /mnt/envs/Anaconda.sh
conda activate /mnt/envs/conda_envs/reversalcurse

export PYTHONPATH=/mnt/projects/reversal_curse:$PYTHONPATH
export HF_HOME=/mnt/cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_GPUS=4
NUM_EPOCHS=3
MAX_SEQ_LEN=512
DATASET_DIR="data/instructions/copypaste_ug100_rg1000_main"

# FSDP 配置
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN

# 大模型列表
declare -A LARGE_MODELS=(
    ["qwen3_32b"]="/mnt/models/qwen3-32b"
    ["gemma3_27b"]="/mnt/models/gemma3-27b"
)

# 大模型 LoRA 配置
LORA_R=64
LORA_ALPHA=128

for model_key in "${!LARGE_MODELS[@]}"; do
    model_path="${LARGE_MODELS[$model_key]}"
    output_dir="runs/exp3/${model_key}"
    
    echo ""
    echo "========================================"
    echo "[FSDP] Processing: $model_key"
    echo "  model_path: $model_path"
    echo "  output_dir: $output_dir"
    echo "========================================"
    
    # 检查是否已完成
    if [ -f "${output_dir}/eval_results.json" ]; then
        echo "[skip] Already completed: $model_key"
        continue
    fi
    
    # 检查模型是否存在
    if [ ! -d "$model_path" ]; then
        echo "[skip] Model not found: $model_path"
        continue
    fi
    
    # 训练
    if [ ! -f "${output_dir}/lora_weights.pt" ]; then
        echo "[train] Starting FSDP training for $model_key..."
        
        TRUST_CODE=""
        if [[ "$model_key" == *"gemma"* ]]; then
            TRUST_CODE="--trust_remote_code"
        fi
        
        torchrun --nproc_per_node=$NUM_GPUS \
            scripts/local_experiment3/train_lora_fsdp.py \
            --model_id "$model_path" \
            --dataset_dir "$DATASET_DIR" \
            --num_epochs $NUM_EPOCHS \
            --max_seq_len $MAX_SEQ_LEN \
            --batch_size 1 \
            --grad_accum_steps 4 \
            --lora_r $LORA_R \
            --lora_alpha $LORA_ALPHA \
            --output_dir "$output_dir" \
            $TRUST_CODE
        
        echo "[train] Done training $model_key"
    else
        echo "[skip] Training already done for $model_key"
    fi
    
    # 评估
    if [ -f "${output_dir}/lora_weights.pt" ] && [ ! -f "${output_dir}/eval_results.json" ]; then
        echo "[eval] Starting FSDP evaluation for $model_key..."
        
        TRUST_CODE=""
        if [[ "$model_key" == *"gemma"* ]]; then
            TRUST_CODE="--trust_remote_code"
        fi
        
        torchrun --nproc_per_node=$NUM_GPUS \
            scripts/local_experiment3/eval_fsdp.py \
            --model_id "$model_path" \
            --lora_weights "${output_dir}/lora_weights.pt" \
            --dataset_dir "$DATASET_DIR" \
            --output_dir "$output_dir" \
            $TRUST_CODE
        
        echo "[eval] Done evaluation for $model_key"
    fi
    
    echo "[done] Completed: $model_key"
done

echo ""
echo "========================================"
echo "All large models completed!"
echo "========================================"
