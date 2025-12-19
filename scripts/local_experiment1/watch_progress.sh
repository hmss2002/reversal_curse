#!/usr/bin/env bash
set -euo pipefail

# 在 tmux 里显示三个模型训练进度条。
# 设计目标：非常明显地告诉你 Phi/Qwen/Gemma3 各自训练到哪了。
#
# 当前优先支持“按 epoch 训练”（train_lora.py --num_epochs，默认 1）：
# - 解析 train.log 里形如：epoch=1/3 step=120/500
#
# 兼容旧模式“按 token_budget 训练”（train_lora.py --token_budget）：
# - 解析 train.log 里形如：tokens_seen=12345/2000000
#
# 用法（通常由 tmux_exp1.sh 自动调用）：
#   bash scripts/local_experiment1/watch_progress.sh <run_root> <num_epochs_or_token_budget>

RUN_ROOT="${1:?missing run_root}"
ARG2="${2:?missing num_epochs_or_token_budget}"

# 说明：ARG2 可能是 num_epochs（默认 1），也可能是旧 token_budget。
NUM_EPOCHS_GUESS="$ARG2"
TOKEN_BUDGET_GUESS="$ARG2"

SLEEP_SECS="${PROGRESS_SLEEP:-10}"
BAR_WIDTH="${PROGRESS_BAR_WIDTH:-36}"

PHI_DIR="$RUN_ROOT/phi3_5_mini_instruct"
QWEN_DIR="$RUN_ROOT/qwen3_4b"
GEMMA_DIR="$RUN_ROOT/gemma3_4b"

phi_log="$PHI_DIR/train.log"
qwen_log="$QWEN_DIR/train.log"
gemma_log="$GEMMA_DIR/train.log"

phi_done="$PHI_DIR/.done"
qwen_done="$QWEN_DIR/.done"
gemma_done="$GEMMA_DIR/.done"

bar() {
  local pct="$1"
  local width="$2"
  if [[ "$pct" -lt 0 ]]; then pct=0; fi
  if [[ "$pct" -gt 100 ]]; then pct=100; fi
  local filled=$(( pct * width / 100 ))
  local empty=$(( width - filled ))
  printf '['
  printf '%0.s#' $(seq 1 "$filled")
  printf '%0.s-' $(seq 1 "$empty")
  printf '] %3d%%' "$pct"
}

parse_seen_budget() {
  local log_path="$1"

  if [[ ! -f "$log_path" ]]; then
    echo "0 $TOKEN_BUDGET_GUESS"
    return 0
  fi

  # 取最后一次出现 tokens_seen=.../... 的行
  local line
  line="$(grep -E 'tokens_seen=[0-9]+/[0-9]+' "$log_path" | tail -n 1 || true)"
  if [[ -z "$line" ]]; then
    echo "0 $TOKEN_BUDGET_GUESS"
    return 0
  fi

  # 用 sed 抠出两个数字
  local seen budget
  seen="$(echo "$line" | sed -n 's/.*tokens_seen=\([0-9][0-9]*\)\/.*/\1/p')"
  budget="$(echo "$line" | sed -n 's/.*tokens_seen=[0-9][0-9]*\/\([0-9][0-9]*\).*/\1/p')"

  if [[ -z "$seen" || -z "$budget" ]]; then
    echo "0 $TOKEN_BUDGET_GUESS"
    return 0
  fi

  echo "$seen $budget"
}

# 从 train.log 解析 epoch 进度。
# 输出："<epoch_cur> <epoch_total> <step_cur> <step_total>"，失败返回非 0。
parse_epoch_progress() {
  local log_path="$1"
  if [[ ! -f "$log_path" ]]; then
    return 1
  fi

  local line
  line="$(grep -E 'epoch=[0-9]+/[0-9]+' "$log_path" | tail -n 1 || true)"
  if [[ -z "$line" ]]; then
    return 1
  fi

  local epoch_cur epoch_total step_cur step_total
  epoch_cur="$(echo "$line" | sed -n 's/.*epoch=\([0-9][0-9]*\)\/.*/\1/p')"
  epoch_total="$(echo "$line" | sed -n 's/.*epoch=[0-9][0-9]*\/\([0-9][0-9]*\).*/\1/p')"
  step_cur="$(echo "$line" | sed -n 's/.*step=\([0-9][0-9]*\)\/.*/\1/p')"
  step_total="$(echo "$line" | sed -n 's/.*step=[0-9][0-9]*\/\([0-9][0-9]*\).*/\1/p')"

  if [[ -z "$epoch_cur" || -z "$epoch_total" ]]; then
    return 1
  fi
  if [[ -z "$step_cur" ]]; then step_cur=0; fi
  if [[ -z "$step_total" || "$step_total" -le 0 ]]; then step_total="$step_cur"; fi

  echo "$epoch_cur $epoch_total $step_cur $step_total"
}

render_one() {
  local name="$1"
  local log_path="$2"
  local done_path="$3"

  local pct status
  if [[ -f "$done_path" ]]; then
    pct=100
    status="DONE"
  else
    # 1) 优先按 epoch 解析
    local epoch_cur epoch_total step_cur step_total
    if parse_epoch_progress "$log_path" >/dev/null 2>&1; then
      read -r epoch_cur epoch_total step_cur step_total < <(parse_epoch_progress "$log_path")
      if [[ "$epoch_total" -le 0 ]]; then epoch_total="$NUM_EPOCHS_GUESS"; fi
      if [[ "$step_total" -le 0 ]]; then step_total="$step_cur"; fi

      # percent = ((epoch_cur-1) + step_cur/step_total) / epoch_total
      local num denom
      denom=$(( epoch_total * step_total ))
      num=$(( (epoch_cur - 1) * step_total + step_cur ))
      if [[ "$denom" -le 0 ]]; then
        pct=0
      else
        pct=$(( num * 100 / denom ))
      fi
      status="RUNNING"

      if [[ ! -f "$log_path" ]]; then
        status="WAITING"
      elif ! grep -q 'epoch=' "$log_path" 2>/dev/null; then
        status="STARTING"
      fi

      printf '%-7s ' "$name"
      bar "$pct" "$BAR_WIDTH"
      printf '  (epoch %s/%s, step %s/%s)  %s\n' "$epoch_cur" "$epoch_total" "$step_cur" "$step_total" "$status"
      return 0
    fi

    # 2) 回退：按 token_budget 解析
    local seen budget
    read -r seen budget < <(parse_seen_budget "$log_path")
    if [[ "$budget" -le 0 ]]; then
      budget="$TOKEN_BUDGET_GUESS"
    fi
    pct=$(( seen * 100 / budget ))
    status="RUNNING"

    if [[ ! -f "$log_path" ]]; then
      status="WAITING"
    elif ! grep -q 'tokens_seen=' "$log_path" 2>/dev/null; then
      status="STARTING"
    fi

    printf '%-7s ' "$name"
    bar "$pct" "$BAR_WIDTH"
    printf '  (%s/%s tokens)  %s\n' "$seen" "$budget" "$status"
    return 0
  fi

  printf '%-7s ' "$name"
  bar "$pct" "$BAR_WIDTH"
  printf '  DONE\n'
}

while true; do
  # tmux 下用 clear 会更舒服
  clear || true

  echo "=== Experiment 1 训练进度（优先按 epoch；兼容 token_budget）==="
  echo "run_root: $RUN_ROOT"
  echo "arg2: $ARG2"
  echo "updated: $(date '+%F %T')"
  echo

  render_one "phi"   "$phi_log"   "$phi_done"
  render_one "qwen"  "$qwen_log"  "$qwen_done"
  render_one "gemma" "$gemma_log" "$gemma_done"

  echo
  echo "提示：切到各模型 window 看详细日志；本窗口每 ${SLEEP_SECS}s 刷新一次。"

  sleep "$SLEEP_SECS"
done
