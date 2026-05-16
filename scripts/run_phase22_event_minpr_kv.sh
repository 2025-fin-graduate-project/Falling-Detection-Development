#!/usr/bin/env bash
# Phase 22: Combining Event-level optimization (P21) + Velocity features (P17/kp7kv)
# Goal: Reach MinPR 0.93 by leveraging joint-specific dynamics.

set -uo pipefail

OUTROOT="${OUTROOT:-results/phase22_event_minpr_kv}"
SUMMARY="$OUTROOT/summary.log"

BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_STRIDE="${EVAL_STRIDE:-2}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
TARGET_MIN_PR="${TARGET_MIN_PR:-0.93}"
GPU_WAIT_MAX_USED_MB="${GPU_WAIT_MAX_USED_MB:-2200}"
GPU_WAIT_INTERVAL_SEC="${GPU_WAIT_INTERVAL_SEC:-60}"

KV_TRAIN="dataset/splits_v2_class_balanced_filtered_kv/train.csv"
KV_VAL="dataset/splits_v2_class_balanced_filtered_kv/val.csv"
KV_TEST="dataset/splits_v2_class_balanced_filtered_kv/test.csv"

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTROOT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

wait_for_dataset() {
    # Wait for class_balanced_filtered_kv to be fully built
    local dst="dataset/splits_v2_class_balanced_filtered_kv"
    while [[ ! -f "$dst/train.csv" || ! -f "$dst/val.csv" || ! -f "$dst/test.csv" ]]; do
        log "Waiting for $dst to be ready..."
        sleep 30
    done
    log "Dataset ready: $dst"
}

wait_for_gpu() {
    while true; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
        if [[ -n "$used" && "$used" -le "$GPU_WAIT_MAX_USED_MB" ]]; then
            log "GPU ready: used=${used}MiB <= ${GPU_WAIT_MAX_USED_MB}MiB"
            return 0
        fi
        log "GPU busy: used=${used:-unknown}MiB; waiting ${GPU_WAIT_INTERVAL_SEC}s"
        sleep "$GPU_WAIT_INTERVAL_SEC"
    done
}

COMMON=(
    --model-type gru
    --gru-units 128,64
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7kv
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
    --preprocessing filtered
    --train-csv "$KV_TRAIN"
    --val-csv   "$KV_VAL"
    --test-csv  "$KV_TEST"
    --label-column label
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --eval-stride "$EVAL_STRIDE"
    --epochs "$EPOCHS" --early-stop-patience "$PATIENCE" --batch-size "$BATCH_SIZE"
    --min-consecutive-values 1,2,3,4,5,7,9
    --threshold-count 37
    --checkpoint-monitor val_event_min_pr
    --threshold-eval-level event
    --event-tolerance-windows 2
    --no-export-tflite
    --quiet
)

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id - already complete"
        return 0
    fi
    wait_for_gpu
    log "START $id"
    uv run python scripts/train_baseline.py \
        --experiment-id "$id" --output-root "$OUTROOT" \
        "${COMMON[@]}" "$@" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    if [[ "$rc" -eq 0 ]]; then
        log_metrics "$id" "OK"
    else
        log "FAIL  $id  (exit $rc)"
    fi
    return "$rc"
}

log_metrics() {
    local id="$1"
    local status="$2"
    python3 - "$OUTROOT" "$id" "$status" <<'PYEOF' | tee -a "$SUMMARY"
import json
import sys
from pathlib import Path

outroot = Path(sys.argv[1])
exp_id = sys.argv[2]
status = sys.argv[3]
m = json.loads((outroot / exp_id / "metrics.json").read_text())
thr = m["threshold_selection"]
tv = m["metrics"]["test_video"]
te = m["metrics"]["test_event_video"]
print(
    f"[{status}] {exp_id} "
    f"test_event_min_pr={te['min_pr']:.4f} "
    f"test_video_min_pr={tv['min_pr']:.4f} "
    f"thr={thr['threshold']:.3f} mc={thr['min_consecutive']} "
    f"thr_eval={thr.get('eval_level', 'video')}"
)
print(f"  event_CM={te['confusion_matrix']} video_CM={tv['confusion_matrix']}")
PYEOF
}

log "=== Phase 22: kp7kv (class_balanced_filtered) + val_event_min_pr checkpoint ==="
wait_for_dataset

# v01: Direct combination of P21 strategy and KV features.
run_exp "P22-v01" --train-negative-stride 2 --seed 42

# v02: P22 with full negative exposure (stride=1).
run_exp "P22-v02" --train-negative-stride 1 --seed 42

log "=== Phase 22 done ==="
