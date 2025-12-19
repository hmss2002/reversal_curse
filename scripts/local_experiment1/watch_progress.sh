#!/usr/bin/env bash
set -euo pipefail

# 在 tmux 里显示三个模型训练进度条。
# 设计目标：非常明显地告诉你 Phi/Qwen/Gemma3 各自训练到哪了（按 token_budget）。
#
# 用法（通常由 tmux_exp1.sh 自动调用）：
#   bash scripts/local_experiment1/watch_progress.sh <run_root> <token_budget>

RUN_ROOT="${1:?missing run_root}"
TOKEN_BUDGET="${2:?missing token_budget}"

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

# 从 train.log 的最后一行 tokens_seen=xxx/yyy 解析出 seen 和 budget。
# 输出："<seen> <budget>"，失败则输出 "0 <TOKEN_BUDGET>"。
parse_seen_budget() {
  local log_path="$1"

  if [[ ! -f "$log_path" ]]; then
    echo "0 $TOKEN_BUDGET"
    return 0
  fi

  # 取最后一次出现 tokens_seen=.../... 的行
  local line
  line="$(grep -E 'tokens_seen=[0-9]+/[0-9]+' "$log_path" | tail -n 1 || true)"
  if [[ -z "$line" ]]; then
    echo "0 $TOKEN_BUDGET"
    return 0
  fi

  # 用 sed 抠出两个数字
  local seen budget
  seen="$(echo "$line" | sed -n 's/.*tokens_seen=\([0-9][0-9]*\)\/.*/\1/p')"
  budget="$(echo "$line" | sed -n 's/.*tokens_seen=[0-9][0-9]*\/\([0-9][0-9]*\).*/\1/p')"

  if [[ -z "$seen" || -z "$budget" ]]; then
    echo "0 $TOKEN_BUDGET"
    return 0
  fi

  echo "$seen $budget"
}

render_one() {
  local name="$1"
  local log_path="$2"
  local done_path="$3"

  local seen budget pct status
  if [[ -f "$done_path" ]]; then
    seen="$TOKEN_BUDGET"
    budget="$TOKEN_BUDGET"
    pct=100
    status="DONE"
  else
    read -r seen budget < <(parse_seen_budget "$log_path")
    if [[ "$budget" -le 0 ]]; then
      budget="$TOKEN_BUDGET"
    fi
    pct=$(( seen * 100 / budget ))
    status="RUNNING"

    if [[ ! -f "$log_path" ]]; then
      status="WAITING"
    elif ! grep -q 'tokens_seen=' "$log_path" 2>/dev/null; then
      status="STARTING"
    fi
  fi

  printf '%-7s ' "$name"
  bar "$pct" "$BAR_WIDTH"
  printf '  (%s/%s tokens)  %s\n' "$seen" "$budget" "$status"
}

while true; do
  # tmux 下用 clear 会更舒服
  clear || true

  echo "=== Experiment 1 训练进度（按 token_budget）==="
  echo "run_root: $RUN_ROOT"
  echo "token_budget: $TOKEN_BUDGET"
  echo "updated: $(date '+%F %T')"
  echo

  render_one "phi"   "$phi_log"   "$phi_done"
  render_one "qwen"  "$qwen_log"  "$qwen_done"
  render_one "gemma" "$gemma_log" "$gemma_done"

  echo
  echo "提示：切到各模型 window 看详细日志；本窗口每 ${SLEEP_SECS}s 刷新一次。"

  sleep "$SLEEP_SECS"
done
