#!/usr/bin/env bash
# Start codex/05 overnight candidate sweep in tmux.

set -euo pipefail

OUTROOT="${OUTROOT:-results/codex05_overnight_candidates}"
TARGET_MIN_PRECISION="${TARGET_MIN_PRECISION:-0.92}"
REPORT_INTERVAL_SECONDS="${REPORT_INTERVAL_SECONDS:-900}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-21600}"
PER_EXP_TIMEOUT_SECONDS="${PER_EXP_TIMEOUT_SECONDS:-5400}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cpu}"
TMUX_SESSION="${TMUX_SESSION:-codex05_overnight}"
TMUX_REPORT_SESSION="${TMUX_REPORT_SESSION:-codex05_overnight_report}"
RUN_LOG="$OUTROOT/nohup.log"
REPORT_LOG="$OUTROOT/reports.log"

mkdir -p "$OUTROOT"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "overnight runner already active: session=$TMUX_SESSION"
    exit 0
fi

tmux new-session -d -s "$TMUX_SESSION" -c "$PWD" \
    "env OUTROOT='$OUTROOT' TARGET_MIN_PRECISION='$TARGET_MIN_PRECISION' TRAIN_DEVICE='$TRAIN_DEVICE' TIME_BUDGET_SECONDS='$TIME_BUDGET_SECONDS' PER_EXP_TIMEOUT_SECONDS='$PER_EXP_TIMEOUT_SECONDS' bash scripts/run_codex05_overnight_candidates.sh > '$RUN_LOG' 2>&1"

if tmux has-session -t "$TMUX_REPORT_SESSION" 2>/dev/null; then
    tmux kill-session -t "$TMUX_REPORT_SESSION"
fi

tmux new-session -d -s "$TMUX_REPORT_SESSION" -c "$PWD" \
    "while tmux has-session -t '$TMUX_SESSION' 2>/dev/null; do uv run python scripts/report_gru_codex03_status.py --output-root '$OUTROOT' --target-min-precision '$TARGET_MIN_PRECISION' >> '$REPORT_LOG' 2>&1 || true; sleep '$REPORT_INTERVAL_SECONDS'; done; uv run python scripts/report_gru_codex03_status.py --output-root '$OUTROOT' --target-min-precision '$TARGET_MIN_PRECISION' >> '$REPORT_LOG' 2>&1 || true"

echo "runner session: $TMUX_SESSION"
echo "report session: $TMUX_REPORT_SESSION"
echo "run log: $RUN_LOG"
echo "reports: $REPORT_LOG"
echo "budget seconds: $TIME_BUDGET_SECONDS"
echo "device: $TRAIN_DEVICE"
