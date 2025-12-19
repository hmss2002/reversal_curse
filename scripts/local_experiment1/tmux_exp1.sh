#!/usr/bin/env bash
set -euo pipefail

# tmux 一键跑 Experiment 1（三模型）：训练 -> 评测 -> 汇总画图
#
# 设计目标（对应你的要求）：
# - 同一训练策略：LoRA（train_lora.py）
# - 同一训练口径：token_budget（train_lora.py --token_budget）
# - 同一评测约束：强制只输出名字（eval_experiment1.py --format_instruction）
# - 训练结束“显著提醒”：tmux display-message + 终端响铃 \a
# - 保存“明显结论/图表/对比”：aggregate_and_plot.py
#
# 注意：
# - Llama/Gemma 可能 gated，需要 HF_TOKEN。
# - 你可以把 HF_TOKEN 放到环境变量里再运行该脚本：
#     export HF_TOKEN=xxxxx

ROOT_DIR="/mnt/projects/reversal_curse"

# Python 解释器选择策略（按优先级）：
# 1) 你显式传入环境变量 PY（例如 PY=/path/to/python）
# 2) 仓库内约定的 conda env 路径（如果存在）
# 3) 当前 shell 的 python（要求你已 conda activate / venv activate）
PY="${PY:-}"
DEFAULT_CONDA_ENV_PY="/mnt/envs/conda_envs/reversalcurse/bin/python"
if [[ -z "$PY" ]]; then
  if [[ -x "$DEFAULT_CONDA_ENV_PY" ]]; then
    PY="$DEFAULT_CONDA_ENV_PY"
  elif command -v python >/dev/null 2>&1; then
    PY="python"
  else
    echo "[error] cannot find python. Please activate an environment or set PY=/path/to/python" >&2
    exit 1
  fi
fi

echo "[python] using: $PY"

# --------- 可按需调整的默认超参 ---------
# 默认：按 epoch 数结束训练（至少完整遍历训练集）。
NUM_EPOCHS="${NUM_EPOCHS:-1}"
# 兼容旧模式：若你显式设置 TOKEN_BUDGET，则走 --token_budget 模式。
TOKEN_BUDGET="${TOKEN_BUDGET:-}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
MICRO_BS="${MICRO_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# 评测：强约束指令（默认英文，减少模板语言冲突）
FORMAT_INSTR="${FORMAT_INSTR:-\n\nAnswer with only the persons name. Do not explain.\n}"

RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/exp1}"
OUT_COMPARE="$RUN_ROOT/compare"
ALL_DONE="$RUN_ROOT/.all_done"

SESSION="${SESSION:-exp1}"

mkdir -p "$RUN_ROOT"

# 约定输出目录
# 注意：你已经明确不再使用 Llama，因此这里改为 Phi-3.5 作为第三个模型。
PHI_DIR="$RUN_ROOT/phi3_5_mini_instruct"
QWEN_DIR="$RUN_ROOT/qwen3_4b"
GEMMA_DIR="$RUN_ROOT/gemma3_4b"

# 训练/评测完成标记文件（aggregate 会等待它们出现）
PHI_DONE="$PHI_DIR/.done"
QWEN_DONE="$QWEN_DIR/.done"
GEMMA_DONE="$GEMMA_DIR/.done"

tmux new-session -d -s "$SESSION" -c "$ROOT_DIR" "bash"

# 目标：
# - 不再并行抢同一块 GPU，避免 OOM
# - 每个模型训练时用 4 张 GPU 做 DDP（数据并行），尽可能吃满机器
# - 仍保留 phi/qwen/gemma 三个 window，方便分别查看日志
# - qwen 等 phi 完成后再开始；gemma 等 qwen 完成后再开始

TORCHRUN_NPROC="${TORCHRUN_NPROC:-4}"
TORCHRUN="${TORCHRUN:-$PY -m torch.distributed.run}"

if [[ -n "$TOKEN_BUDGET" ]]; then
  STOP_ARG="--token_budget $TOKEN_BUDGET"
else
  STOP_ARG="--num_epochs $NUM_EPOCHS"
fi

TRAIN_COMMON="$STOP_ARG --max_seq_len $MAX_SEQ_LEN --micro_batch_size $MICRO_BS --grad_accum_steps $GRAD_ACCUM --mixed_precision $MIXED_PRECISION --max_grad_norm $MAX_GRAD_NORM"

TMUX_CMD_PHI="cd $ROOT_DIR && \
  mkdir -p $PHI_DIR && \
  echo '[phi] start' && date && \
  CUDA_VISIBLE_DEVICES=0,1,2,3 $TORCHRUN --nproc_per_node $TORCHRUN_NPROC scripts/local_experiment1/train_lora.py --preset phi3_5_mini_instruct $TRAIN_COMMON --output_dir $PHI_DIR 2>&1 | tee $PHI_DIR/train.log && \
  CUDA_VISIBLE_DEVICES=0 $PY scripts/local_experiment1/eval_experiment1.py --base_model_id microsoft/Phi-3.5-mini-instruct --revision 2fe192450127e6a83f7441aef6e3ca586c338b77 --lora_dir $PHI_DIR --dataset_dir data/reverse_experiments/june_version_7921032488 --out_dir $PHI_DIR/eval --format_instruction \"$FORMAT_INSTR\" 2>&1 | tee $PHI_DIR/eval.log && \
  date > $PHI_DONE && \
  printf '\\a' && tmux display-message -d 0 '[DONE] phi finished' && \
  echo '[phi] done' && date"

TMUX_CMD_QWEN="cd $ROOT_DIR && \
  mkdir -p $QWEN_DIR && \
  echo '[qwen] waiting for: $PHI_DONE' && \
  while [[ ! -f $PHI_DONE ]]; do sleep 30; date; done && \
  echo '[qwen] start' && date && \
  CUDA_VISIBLE_DEVICES=0,1,2,3 $TORCHRUN --nproc_per_node $TORCHRUN_NPROC scripts/local_experiment1/train_lora.py --preset qwen3_4b $TRAIN_COMMON --output_dir $QWEN_DIR 2>&1 | tee $QWEN_DIR/train.log && \
  CUDA_VISIBLE_DEVICES=0 $PY scripts/local_experiment1/eval_experiment1.py --base_model_id Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --lora_dir $QWEN_DIR --dataset_dir data/reverse_experiments/june_version_7921032488 --out_dir $QWEN_DIR/eval --format_instruction \"$FORMAT_INSTR\" 2>&1 | tee $QWEN_DIR/eval.log && \
  date > $QWEN_DONE && \
  printf '\\a' && tmux display-message -d 0 '[DONE] qwen finished' && \
  echo '[qwen] done' && date"

TMUX_CMD_GEMMA="cd $ROOT_DIR && \
  mkdir -p $GEMMA_DIR && \
  echo '[gemma] waiting for: $QWEN_DONE' && \
  while [[ ! -f $QWEN_DONE ]]; do sleep 30; date; done && \
  echo '[gemma] start' && date && \
  CUDA_VISIBLE_DEVICES=0,1,2,3 $TORCHRUN --nproc_per_node $TORCHRUN_NPROC scripts/local_experiment1/train_lora.py --preset gemma3_4b $TRAIN_COMMON --output_dir $GEMMA_DIR 2>&1 | tee $GEMMA_DIR/train.log && \
  CUDA_VISIBLE_DEVICES=0 $PY scripts/local_experiment1/eval_experiment1.py --base_model_id google/gemma-3-4b-it --revision 093f9f388b31de276ce2de164bdc2081324b9767 --lora_dir $GEMMA_DIR --dataset_dir data/reverse_experiments/june_version_7921032488 --out_dir $GEMMA_DIR/eval --format_instruction \"$FORMAT_INSTR\" 2>&1 | tee $GEMMA_DIR/eval.log && \
  date > $GEMMA_DONE && \
  printf '\\a' && tmux display-message -d 0 '[DONE] gemma finished' && \
  echo '[gemma] done' && date"

tmux rename-window -t "$SESSION:0" "phi"
(tmux send-keys -t "$SESSION:phi" "$TMUX_CMD_PHI" C-m)

tmux new-window -t "$SESSION" -n "qwen" -c "$ROOT_DIR" "bash"
(tmux send-keys -t "$SESSION:qwen" "$TMUX_CMD_QWEN" C-m)

tmux new-window -t "$SESSION" -n "gemma" -c "$ROOT_DIR" "bash"
(tmux send-keys -t "$SESSION:gemma" "$TMUX_CMD_GEMMA" C-m)

# window 4: aggregate（自动等待三个 .done 文件出现，然后自动汇总+画图）
AGG_CMD="cd $ROOT_DIR && \
  echo '[aggregate] waiting for done markers...' && \
  echo '  - $PHI_DONE' && \
  echo '  - $QWEN_DONE' && \
  echo '  - $GEMMA_DONE' && \
  while [[ ! -f $PHI_DONE || ! -f $QWEN_DONE || ! -f $GEMMA_DONE ]]; do \
    sleep 30; \
    date; \
    [[ -f $PHI_DONE ]] && echo '  ok: phi done' || echo '  ..: phi running'; \
    [[ -f $QWEN_DONE  ]] && echo '  ok: qwen done'  || echo '  ..: qwen running'; \
    [[ -f $GEMMA_DONE ]] && echo '  ok: gemma done' || echo '  ..: gemma running'; \
  done && \
  echo '[aggregate] all done. aggregating...' && \
  mkdir -p $OUT_COMPARE && \
  $PY scripts/local_experiment1/aggregate_and_plot.py \
    --runs qwen=$QWEN_DIR/eval/summary.json,gemma=$GEMMA_DIR/eval/summary.json,phi=$PHI_DIR/eval/summary.json \
    --out_dir $OUT_COMPARE && \
  date > $OUT_COMPARE/.done && \
  date > $ALL_DONE && \
  printf '\\a\\a\\a' && \
  tmux display-message -d 0 \"[ALL DONE] exp1 finished. Results: $OUT_COMPARE (and per-model under $RUN_ROOT)\" && \
  echo '[aggregate] ALL DONE' && \
  echo \"[result] compare dir: $OUT_COMPARE\" && \
  echo \"[result] compare.png: $OUT_COMPARE/compare.png\" && \
  echo \"[result] compare.csv: $OUT_COMPARE/compare.csv\" && \
  echo \"[result] compare_by_split.csv: $OUT_COMPARE/compare_by_split.csv\" && \
  date"

tmux new-window -t "$SESSION" -n "aggregate" -c "$ROOT_DIR" "bash"
(tmux send-keys -t "$SESSION:aggregate" "$AGG_CMD" C-m)

# window 5: progress（非常明显的进度条面板）
# 说明：默认按 epoch 解析进度（epoch=..../.... + step=..../....）。
# 若你走旧 token_budget 模式，脚本也会自动回退解析 tokens_seen。
PROGRESS_ARG="$NUM_EPOCHS"
if [[ -n "$TOKEN_BUDGET" ]]; then
  PROGRESS_ARG="$TOKEN_BUDGET"
fi
PROGRESS_CMD="cd $ROOT_DIR && bash scripts/local_experiment1/watch_progress.sh $RUN_ROOT $PROGRESS_ARG"
tmux new-window -t "$SESSION" -n "progress" -c "$ROOT_DIR" "bash"
(tmux send-keys -t "$SESSION:progress" "$PROGRESS_CMD" C-m)

# 最后：提示用户如何 attach

echo "[tmux] session created: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Progress: tmux select-window -t $SESSION:progress"
