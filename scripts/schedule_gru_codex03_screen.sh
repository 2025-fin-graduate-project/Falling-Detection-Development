#!/usr/bin/env bash
# Start the codex/03 fast screening run in tmux.

set -euo pipefail

OUTROOT="${OUTROOT:-results/gru_codex03_screen}"
TARGET_MIN_PRECISION="${TARGET_MIN_PRECISION:-0.92}"
REPORT_INTERVAL_SECONDS="${REPORT_INTERVAL_SECONDS:-900}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cpu}"
TMUX_SESSION="${TMUX_SESSION:-gru_codex03_screen}"
TMUX_REPORT_SESSION="${TMUX_REPORT_SESSION:-gru_codex03_screen_report}"
RUN_LOG="$OUTROOT/nohup.log"
REPORT_LOG="$OUTROOT/reports.log"

mkdir -p "$OUTROOT"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "screen runner already active: session=$TMUX_SESSION"
    exit 0
fi

tmux new-session -d -s "$TMUX_SESSION" -c "$PWD" \
    "env OUTROOT='$OUTROOT' TARGET_MIN_PRECISION='$TARGET_MIN_PRECISION' TRAIN_DEVICE='$TRAIN_DEVICE' bash scripts/run_gru_codex03_screen.sh > '$RUN_LOG' 2>&1"

if tmux has-session -t "$TMUX_REPORT_SESSION" 2>/dev/null; then
    tmux kill-session -t "$TMUX_REPORT_SESSION"
fi

tmux new-session -d -s "$TMUX_REPORT_SESSION" -c "$PWD" \
    "while tmux has-session -t '$TMUX_SESSION' 2>/dev/null; do uv run python scripts/report_gru_codex03_status.py --output-root '$OUTROOT' --target-min-precision '$TARGET_MIN_PRECISION' >> '$REPORT_LOG' 2>&1 || true; sleep '$REPORT_INTERVAL_SECONDS'; done; uv run python scripts/report_gru_codex03_status.py --output-root '$OUTROOT' --target-min-precision '$TARGET_MIN_PRECISION' >> '$REPORT_LOG' 2>&1 || true"

echo "runner session: $TMUX_SESSION"
echo "report session: $TMUX_REPORT_SESSION"
echo "run log: $RUN_LOG"
echo "reports: $REPORT_LOG"
