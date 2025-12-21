#!/bin/bash
# Local Experiment 3: 完整实验运行脚本
# 在多个模型上运行 Experiment 3
# 硬件: 4 x V100 32GB

set -e

cd /mnt/projects/reversal_curse

DATASET_DIR="data/instructions/copypaste_ug100_rg1000_main"
OUTPUT_ROOT="outputs/local_exp3"
NUM_EPOCHS=3
MAX_SEQ_LEN=512

echo "=========================================="
echo "Local Experiment 3: Reversing Instructions"
echo "Hardware: 4 x V100 32GB"
echo "=========================================="

# ====================================
# 小模型配置 (4B 级别)
# ====================================
SMALL_MODELS=(
    "Qwen/Qwen3-4B"
    "google/gemma-3-4b-it"
    "microsoft/Phi-3.5-mini-instruct"
)

SMALL_MODEL_NAMES=(
    "qwen3_4b"
    "gemma3_4b"
    "phi3_5_mini"
)

# 小模型参数：batch size 大，gradient accumulation 小
SMALL_MICRO_BATCH=4
SMALL_GRAD_ACCUM=2
SMALL_LORA_R=16
SMALL_LORA_ALPHA=32

# ====================================
# 大模型配置 (27-32B 级别，本地路径)
# ====================================
LARGE_MODELS=(
    "/mnt/models/qwen3-32b"
    "/mnt/models/gemma3-27b"
)

LARGE_MODEL_NAMES=(
    "qwen3_32b"
    "gemma3_27b"
)

# 大模型参数：batch size 小，gradient accumulation 大，更高的 LoRA rank
LARGE_MICRO_BATCH=1
LARGE_GRAD_ACCUM=8
LARGE_LORA_R=64
LARGE_LORA_ALPHA=128

# ====================================
# 训练小模型
# ====================================
echo ""
echo "=========================================="
echo "Phase 1: Training Small Models (4B)"
echo "=========================================="

for i in "${!SMALL_MODELS[@]}"; do
    MODEL="${SMALL_MODELS[$i]}"
    NAME="${SMALL_MODEL_NAMES[$i]}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${NAME}"

    echo ""
    echo "=========================================="
    echo "Training: ${NAME} (${MODEL})"
    echo "Config: batch=${SMALL_MICRO_BATCH}, accum=${SMALL_GRAD_ACCUM}, lora_r=${SMALL_LORA_R}"
    echo "=========================================="

    python scripts/local_experiment3/train_lora.py \
        --model_id "${MODEL}" \
        --dataset_dir "${DATASET_DIR}" \
        --num_epochs ${NUM_EPOCHS} \
        --max_seq_len ${MAX_SEQ_LEN} \
        --micro_batch_size ${SMALL_MICRO_BATCH} \
        --grad_accum_steps ${SMALL_GRAD_ACCUM} \
        --lora_r ${SMALL_LORA_R} \
        --lora_alpha ${SMALL_LORA_ALPHA} \
        --output_dir "${OUTPUT_DIR}" \
        --trust_remote_code

    echo ""
    echo "Evaluating: ${NAME}"

    python scripts/local_experiment3/eval_experiment3.py \
        --base_model_id "${MODEL}" \
        --lora_dir "${OUTPUT_DIR}" \
        --dataset_dir "${DATASET_DIR}" \
        --out_dir "${OUTPUT_DIR}/eval" \
        --batch_size 8 \
        --trust_remote_code
done

# ====================================
# 训练大模型
# ====================================
echo ""
echo "=========================================="
echo "Phase 2: Training Large Models (27-32B)"
echo "=========================================="

for i in "${!LARGE_MODELS[@]}"; do
    MODEL="${LARGE_MODELS[$i]}"
    NAME="${LARGE_MODEL_NAMES[$i]}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${NAME}"

    if [ ! -d "${MODEL}" ]; then
        echo "[skip] ${MODEL} not found, skipping..."
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Training: ${NAME} (${MODEL})"
    echo "Config: batch=${LARGE_MICRO_BATCH}, accum=${LARGE_GRAD_ACCUM}, lora_r=${LARGE_LORA_R}"
    echo "=========================================="

    python scripts/local_experiment3/train_lora.py \
        --model_id "${MODEL}" \
        --dataset_dir "${DATASET_DIR}" \
        --num_epochs ${NUM_EPOCHS} \
        --max_seq_len ${MAX_SEQ_LEN} \
        --micro_batch_size ${LARGE_MICRO_BATCH} \
        --grad_accum_steps ${LARGE_GRAD_ACCUM} \
        --lora_r ${LARGE_LORA_R} \
        --lora_alpha ${LARGE_LORA_ALPHA} \
        --output_dir "${OUTPUT_DIR}" \
        --trust_remote_code

    echo ""
    echo "Evaluating: ${NAME}"

    python scripts/local_experiment3/eval_experiment3.py \
        --base_model_id "${MODEL}" \
        --lora_dir "${OUTPUT_DIR}" \
        --dataset_dir "${DATASET_DIR}" \
        --out_dir "${OUTPUT_DIR}/eval" \
        --batch_size 2 \
        --trust_remote_code
done

# ====================================
# 聚合结果
# ====================================
echo ""
echo "=========================================="
echo "Aggregating Results"
echo "=========================================="

python scripts/local_experiment3/aggregate_and_plot.py \
    --results_root "${OUTPUT_ROOT}" \
    --out_dir "${OUTPUT_ROOT}/aggregated"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: ${OUTPUT_ROOT}"
echo "=========================================="
