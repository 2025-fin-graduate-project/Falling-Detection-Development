#!/usr/bin/env bash
# Start codex/03 GRU experiments in the background and write progress reports.

set -euo pipefail

OUTROOT="${OUTROOT:-results/gru_codex03_optimal}"
TARGET_MIN_PRECISION="${TARGET_MIN_PRECISION:-0.92}"
REPORT_INTERVAL_SECONDS="${REPORT_INTERVAL_SECONDS:-1800}"

RUN_LOG="$OUTROOT/nohup.log"
REPORT_LOG="$OUTROOT/reports.log"
MONITOR_LOG="$OUTROOT/report_monitor.log"
PID_FILE="$OUTROOT/run.pid"
MONITOR_PID_FILE="$OUTROOT/report_monitor.pid"
TMUX_SESSION="${TMUX_SESSION:-gru_codex03}"
TMUX_REPORT_SESSION="${TMUX_REPORT_SESSION:-gru_codex03_report}"

mkdir -p "$OUTROOT"

if command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "codex/03 tmux runner already active: session=$TMUX_SESSION"
        echo "run log: $RUN_LOG"
        echo "reports: $REPORT_LOG"
        exit 0
    fi

    echo "Starting codex/03 GRU runner in tmux session: $TMUX_SESSION"
    tmux new-session -d -s "$TMUX_SESSION" -c "$PWD" \
        "env OUTROOT='$OUTROOT' TARGET_MIN_PRECISION='$TARGET_MIN_PRECISION' bash scripts/run_gru_codex03_optimal.sh > '$RUN_LOG' 2>&1"
    echo "tmux:$TMUX_SESSION" > "$PID_FILE"

    if tmux has-session -t "$TMUX_REPORT_SESSION" 2>/dev/null; then
        tmux kill-session -t "$TMUX_REPORT_SESSION"
    fi
    echo "Starting ${REPORT_INTERVAL_SECONDS}s report monitor in tmux session: $TMUX_REPORT_SESSION"
    tmux new-session -d -s "$TMUX_REPORT_SESSION" -c "$PWD" \
        "while tmux has-session -t '$TMUX_SESSION' 2>/dev/null; do uv run python scripts/report_gru_codex03_status.py --output-root '$OUTROOT' --target-min-precision '$TARGET_MIN_PRECISION' >> '$REPORT_LOG' 2>&1 || true; sleep '$REPORT_INTERVAL_SECONDS'; done; uv run python scripts/report_gru_codex03_status.py --output-root '$OUTROOT' --target-min-precision '$TARGET_MIN_PRECISION' >> '$REPORT_LOG' 2>&1 || true"
    echo "tmux:$TMUX_REPORT_SESSION" > "$MONITOR_PID_FILE"

    echo "runner session: $TMUX_SESSION"
    echo "monitor session: $TMUX_REPORT_SESSION"
    echo "run log: $RUN_LOG"
    echo "reports: $REPORT_LOG"
    exit 0
fi

if [[ -f "$PID_FILE" ]]; then
    old_pid=$(cat "$PID_FILE")
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "codex/03 runner already active: pid=$old_pid"
        echo "run log: $RUN_LOG"
        echo "reports: $REPORT_LOG"
        exit 0
    fi
fi

echo "Starting codex/03 GRU runner..."
nohup env OUTROOT="$OUTROOT" TARGET_MIN_PRECISION="$TARGET_MIN_PRECISION" \
    bash scripts/run_gru_codex03_optimal.sh > "$RUN_LOG" 2>&1 &
run_pid=$!
echo "$run_pid" > "$PID_FILE"

echo "Starting ${REPORT_INTERVAL_SECONDS}s report monitor..."
nohup bash -c '
run_pid="$1"
outroot="$2"
report_log="$3"
interval="$4"
target="$5"

while kill -0 "$run_pid" 2>/dev/null; do
    uv run python scripts/report_gru_codex03_status.py \
        --output-root "$outroot" \
        --target-min-precision "$target" >> "$report_log" 2>&1 || true
    sleep "$interval"
done

uv run python scripts/report_gru_codex03_status.py \
    --output-root "$outroot" \
    --target-min-precision "$target" >> "$report_log" 2>&1 || true
' bash "$run_pid" "$OUTROOT" "$REPORT_LOG" "$REPORT_INTERVAL_SECONDS" "$TARGET_MIN_PRECISION" > "$MONITOR_LOG" 2>&1 &
monitor_pid=$!
echo "$monitor_pid" > "$MONITOR_PID_FILE"

echo "runner pid: $run_pid"
echo "monitor pid: $monitor_pid"
echo "run log: $RUN_LOG"
echo "reports: $REPORT_LOG"
