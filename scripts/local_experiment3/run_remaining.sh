#!/bin/bash
# Experiment 3: 完成剩余训练和评测 (DDP 4卡)
# 统一输出到 runs/exp3/

set -e

cd /mnt/projects/reversal_curse

DATASET_DIR="data/instructions/copypaste_ug100_rg1000_main"
OUTPUT_ROOT="runs/exp3"
NUM_EPOCHS=3
MAX_SEQ_LEN=512
NUM_GPUS=4

mkdir -p "${OUTPUT_ROOT}"

echo "=========================================="
echo "Experiment 3: Complete All Tasks (DDP ${NUM_GPUS} GPUs)"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

# ====================================
# 辅助函数
# ====================================
train_model() {
    local MODEL="$1"
    local NAME="$2"
    local BATCH="$3"
    local ACCUM="$4"
    local LORA_R="$5"
    local LORA_ALPHA="$6"
    local OUTPUT_DIR="${OUTPUT_ROOT}/${NAME}"

    echo ""
    echo "=========================================="
    echo "Training: ${NAME}"
    echo "Config: ${NUM_GPUS} GPUs, batch=${BATCH}/gpu, accum=${ACCUM}, lora_r=${LORA_R}"
    echo "=========================================="

    torchrun --nproc_per_node=${NUM_GPUS} scripts/local_experiment3/train_lora.py \
        --model_id "${MODEL}" \
        --dataset_dir "${DATASET_DIR}" \
        --num_epochs ${NUM_EPOCHS} \
        --max_seq_len ${MAX_SEQ_LEN} \
        --micro_batch_size ${BATCH} \
        --grad_accum_steps ${ACCUM} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --output_dir "${OUTPUT_DIR}" \
        --trust_remote_code
}

eval_model() {
    local MODEL="$1"
    local NAME="$2"
    local BATCH="$3"
    local OUTPUT_DIR="${OUTPUT_ROOT}/${NAME}"

    echo ""
    echo "=========================================="
    echo "Evaluating: ${NAME} (DDP ${NUM_GPUS} GPUs)"
    echo "=========================================="

    torchrun --nproc_per_node=${NUM_GPUS} scripts/local_experiment3/eval_experiment3.py \
        --base_model_id "${MODEL}" \
        --lora_dir "${OUTPUT_DIR}" \
        --dataset_dir "${DATASET_DIR}" \
        --out_dir "${OUTPUT_DIR}/eval" \
        --batch_size ${BATCH} \
        --trust_remote_code
    
    touch "${OUTPUT_DIR}/.done"
}

# ====================================
# 1. 检查并补齐小模型评测
# ====================================
echo ""
echo "=========================================="
echo "Phase 1: Complete Small Model Evaluations"
echo "=========================================="

# Qwen3-4B
if [ -f "${OUTPUT_ROOT}/qwen3_4b/adapter_model.safetensors" ] && [ ! -f "${OUTPUT_ROOT}/qwen3_4b/eval/summary.json" ]; then
    eval_model "Qwen/Qwen3-4B" "qwen3_4b" 4
else
    echo "[skip] qwen3_4b: already done or not trained"
fi

# Gemma3-4B
if [ -f "${OUTPUT_ROOT}/gemma3_4b/adapter_model.safetensors" ] && [ ! -f "${OUTPUT_ROOT}/gemma3_4b/eval/summary.json" ]; then
    eval_model "google/gemma-3-4b-it" "gemma3_4b" 4
else
    echo "[skip] gemma3_4b: already done or not trained"
fi

# Phi-3.5
if [ -f "${OUTPUT_ROOT}/phi3_5_mini/adapter_model.safetensors" ] && [ ! -f "${OUTPUT_ROOT}/phi3_5_mini/eval/summary.json" ]; then
    eval_model "microsoft/Phi-3.5-mini-instruct" "phi3_5_mini" 4
else
    echo "[skip] phi3_5_mini: already done or not trained"
fi

# ====================================
# 2. 训练和评测大模型
# ====================================
echo ""
echo "=========================================="
echo "Phase 2: Large Models (27-32B)"
echo "=========================================="

# Qwen3-32B
if [ -d "/mnt/models/qwen3-32b" ]; then
    if [ ! -f "${OUTPUT_ROOT}/qwen3_32b/adapter_model.safetensors" ]; then
        train_model "/mnt/models/qwen3-32b" "qwen3_32b" 1 2 64 128
    fi
    if [ -f "${OUTPUT_ROOT}/qwen3_32b/adapter_model.safetensors" ] && [ ! -f "${OUTPUT_ROOT}/qwen3_32b/eval/summary.json" ]; then
        eval_model "/mnt/models/qwen3-32b" "qwen3_32b" 1
    fi
else
    echo "[skip] /mnt/models/qwen3-32b not found"
fi

# Gemma3-27B
if [ -d "/mnt/models/gemma3-27b" ]; then
    if [ ! -f "${OUTPUT_ROOT}/gemma3_27b/adapter_model.safetensors" ]; then
        train_model "/mnt/models/gemma3-27b" "gemma3_27b" 1 2 64 128
    fi
    if [ -f "${OUTPUT_ROOT}/gemma3_27b/adapter_model.safetensors" ] && [ ! -f "${OUTPUT_ROOT}/gemma3_27b/eval/summary.json" ]; then
        eval_model "/mnt/models/gemma3-27b" "gemma3_27b" 1
    fi
else
    echo "[skip] /mnt/models/gemma3-27b not found"
fi

# ====================================
# 3. 聚合结果
# ====================================
echo ""
echo "=========================================="
echo "Aggregating Results"
echo "=========================================="

python scripts/local_experiment3/aggregate_and_plot.py \
    --results_root "${OUTPUT_ROOT}" \
    --out_dir "${OUTPUT_ROOT}/compare"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: ${OUTPUT_ROOT}"
echo "=========================================="
