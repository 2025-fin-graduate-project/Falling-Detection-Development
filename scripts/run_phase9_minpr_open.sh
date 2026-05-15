#!/usr/bin/env bash
# Phase 9 open exploration - FP reduction candidates after the first MinPR screen.

set -uo pipefail

TARGET_MIN_PR="${TARGET_MIN_PR:-0.92}"
OUTROOT="${OUTROOT:-results/phase9_minpr_open}"
SUMMARY="$OUTROOT/summary.log"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_STRIDE="${EVAL_STRIDE:-2}"

if [[ "${TRAIN_DEVICE:-gpu}" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTROOT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id - already complete"
        return 0
    fi
    log "START $id"
    if uv run python scripts/train_baseline.py \
            --experiment-id "$id" \
            --output-root "$OUTROOT" \
            --quiet \
            "$@" \
            2>&1 | tee "$logfile"; then
        log "OK    $id"
    else
        log "FAIL  $id"
    fi
}

COMMON=(
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --eval-stride "$EVAL_STRIDE"
    --epochs "$EPOCHS" --early-stop-patience "$PATIENCE" --batch-size "$BATCH_SIZE"
    --min-val-min-pr "$TARGET_MIN_PR"
    --min-consecutive-values 1,3,5,7,9
    --threshold-count 37
    --no-export-tflite
)

GRU40=(
    --model-type gru --gru-units 128,64
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
)
GRU50=(
    --model-type gru --gru-units 128,64
    --target-steps 50 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
)
TCN40=(
    --model-type tcn --tcn-channels 32,64,64,96
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7
)

log "=== Phase 9 open exploration: FP reduction sweep, device=${TRAIN_DEVICE:-gpu} ==="

# Isolate threshold/min_consecutive expansion on the best first-screen candidate.
run_exp P9O-v01 "${COMMON[@]}" --train-negative-stride 2 "${GRU40[@]}" \
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25

# Heavier negative exposure and lower fall alpha to reduce normal-video false positives.
run_exp P9O-v02 "${COMMON[@]}" --train-negative-stride 1 "${GRU40[@]}" \
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.15
run_exp P9O-v03 "${COMMON[@]}" --train-negative-stride 1 "${GRU40[@]}" \
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.10

# Cross-entropy baseline with balanced class weights and full negative stride.
run_exp P9O-v04 "${COMMON[@]}" --train-negative-stride 1 "${GRU40[@]}"

# Longer temporal context with lower fall alpha.
run_exp P9O-v05 "${COMMON[@]}" --train-negative-stride 1 "${GRU50[@]}" \
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.15

# TCN light 40f with full negative stride and wider threshold/consecutive sweep.
run_exp P9O-v06 "${COMMON[@]}" --train-negative-stride 1 "${TCN40[@]}"

log "=== Phase 9 open summary ==="
python3 scripts/report_phase9_minpr_status.py \
    --output-root "$OUTROOT" \
    --target-min-pr "$TARGET_MIN_PR" \
    2>&1 | tee -a "$SUMMARY"
