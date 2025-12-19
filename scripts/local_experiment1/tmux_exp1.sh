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
TOKEN_BUDGET="${TOKEN_BUDGET:-2000000}"
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

# 在 tmux 里开三窗格（或三窗口）并行跑
# 这里用“一个 session 三个 window”，便于你随时 attach 查看。

tmux new-session -d -s "$SESSION" -c "$ROOT_DIR" "bash"

# window 1: phi
TMUX_CMD_PHI="cd $ROOT_DIR && \
  echo '[phi] start' && date && \
  $PY scripts/local_experiment1/train_lora.py --preset phi3_5_mini_instruct --token_budget $TOKEN_BUDGET --max_seq_len $MAX_SEQ_LEN --micro_batch_size $MICRO_BS --grad_accum_steps $GRAD_ACCUM --mixed_precision $MIXED_PRECISION --max_grad_norm $MAX_GRAD_NORM --output_dir $PHI_DIR 2>&1 | tee $PHI_DIR/train.log && \
  $PY scripts/local_experiment1/eval_experiment1.py --base_model_id microsoft/Phi-3.5-mini-instruct --revision 2fe192450127e6a83f7441aef6e3ca586c338b77 --lora_dir $PHI_DIR --dataset_dir data/reverse_experiments/june_version_7921032488 --out_dir $PHI_DIR/eval --format_instruction \"$FORMAT_INSTR\" 2>&1 | tee $PHI_DIR/eval.log && \
  date > $PHI_DONE && \
  printf '\\a' && tmux display-message -d 0 '[DONE] phi3_5_mini_instruct finished' && \
  echo '[phi] done' && date"

tmux rename-window -t "$SESSION:0" "phi"
# shellcheck disable=SC2086
(tmux send-keys -t "$SESSION:phi" "$TMUX_CMD_PHI" C-m)

# window 2: qwen
# 注意：base_model_id 必须与训练脚本 preset 一致；这里固定为 Qwen/Qwen3-4B
TMUX_CMD_QWEN="cd $ROOT_DIR && \
  echo '[qwen] start' && date && \
  $PY scripts/local_experiment1/train_lora.py --preset qwen3_4b --token_budget $TOKEN_BUDGET --max_seq_len $MAX_SEQ_LEN --micro_batch_size $MICRO_BS --grad_accum_steps $GRAD_ACCUM --mixed_precision $MIXED_PRECISION --max_grad_norm $MAX_GRAD_NORM --output_dir $QWEN_DIR 2>&1 | tee $QWEN_DIR/train.log && \
  $PY scripts/local_experiment1/eval_experiment1.py --base_model_id Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --lora_dir $QWEN_DIR --dataset_dir data/reverse_experiments/june_version_7921032488 --out_dir $QWEN_DIR/eval --format_instruction \"$FORMAT_INSTR\" 2>&1 | tee $QWEN_DIR/eval.log && \
  date > $QWEN_DONE && \
  printf '\\a' && tmux display-message -d 0 '[DONE] qwen3_4b finished' && \
  echo '[qwen] done' && date"

tmux new-window -t "$SESSION" -n "qwen" -c "$ROOT_DIR" "bash"
(tmux send-keys -t "$SESSION:qwen" "$TMUX_CMD_QWEN" C-m)

# window 3: gemma
TMUX_CMD_GEMMA="cd $ROOT_DIR && \
  echo '[gemma] start' && date && \
  $PY scripts/local_experiment1/train_lora.py --preset gemma3_4b --token_budget $TOKEN_BUDGET --max_seq_len $MAX_SEQ_LEN --micro_batch_size $MICRO_BS --grad_accum_steps $GRAD_ACCUM --mixed_precision $MIXED_PRECISION --max_grad_norm $MAX_GRAD_NORM --output_dir $GEMMA_DIR 2>&1 | tee $GEMMA_DIR/train.log && \
  $PY scripts/local_experiment1/eval_experiment1.py --base_model_id google/gemma-3-4b-it --revision 093f9f388b31de276ce2de164bdc2081324b9767 --lora_dir $GEMMA_DIR --dataset_dir data/reverse_experiments/june_version_7921032488 --out_dir $GEMMA_DIR/eval --format_instruction \"$FORMAT_INSTR\" 2>&1 | tee $GEMMA_DIR/eval.log && \
  date > $GEMMA_DONE && \
  printf '\\a' && tmux display-message -d 0 '[DONE] gemma3_4b finished' && \
  echo '[gemma] done' && date"

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
# 说明：实时从每个模型的 train.log 解析 tokens_seen/预算，绘制进度条。
PROGRESS_CMD="cd $ROOT_DIR && bash scripts/local_experiment1/watch_progress.sh $RUN_ROOT $TOKEN_BUDGET"
tmux new-window -t "$SESSION" -n "progress" -c "$ROOT_DIR" "bash"
(tmux send-keys -t "$SESSION:progress" "$PROGRESS_CMD" C-m)

# 最后：提示用户如何 attach

echo "[tmux] session created: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Progress: tmux select-window -t $SESSION:progress"
