#!/usr/bin/env bash
# Phase 9 MinPR screen - short float-only search.

set -uo pipefail

TARGET_MIN_PR="${TARGET_MIN_PR:-0.92}"
OUTROOT="${OUTROOT:-results/phase9_minpr_screen}"
SUMMARY="$OUTROOT/summary.log"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_STRIDE="${EVAL_STRIDE:-2}"

if [[ "${TRAIN_DEVICE:-cpu}" == "cpu" ]]; then
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
    --train-negative-stride 2
    --eval-stride "$EVAL_STRIDE"
    --epochs "$EPOCHS" --early-stop-patience "$PATIENCE" --batch-size "$BATCH_SIZE"
    --min-val-min-pr "$TARGET_MIN_PR"
    --min-consecutive-values 1,3,5
    --no-export-tflite
)

FOCAL=(--focal-loss --focal-gamma 2.0 --focal-alpha 0.25)
WIN30=(--target-steps 30 --window-start-sec 3.0 --window-end-sec 9.0)
WIN40=(--target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0)

log "=== Phase 9 MinPR screen: target=${TARGET_MIN_PR}, epochs=${EPOCHS}, batch=${BATCH_SIZE}, eval_stride=${EVAL_STRIDE}, device=${TRAIN_DEVICE:-cpu} ==="

run_exp P9S-v01 "${COMMON[@]}" --model-type gru --gru-units 128,64 "${WIN40[@]}" --feature-set kp7  --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 "${FOCAL[@]}"
run_exp P9S-v02 "${COMMON[@]}" --model-type gru --gru-units 128,64 "${WIN30[@]}" --feature-set kp7  --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 "${FOCAL[@]}"
run_exp P9S-v03 "${COMMON[@]}" --model-type gru --gru-units 128,64 "${WIN30[@]}" --feature-set kp12 --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 "${FOCAL[@]}"
run_exp P9S-v04 "${COMMON[@]}" --model-type gru --gru-units 256,128 "${WIN40[@]}" --feature-set minimal "${FOCAL[@]}"
run_exp P9S-v05 "${COMMON[@]}" --model-type tcn --tcn-channels 32,64,64,96   "${WIN30[@]}" --feature-set kp7
run_exp P9S-v06 "${COMMON[@]}" --model-type tcn --tcn-channels 32,64,64,96   "${WIN40[@]}" --feature-set kp7
run_exp P9S-v07 "${COMMON[@]}" --model-type tcn --tcn-channels 64,64,128,128 "${WIN30[@]}" --feature-set kp7

log "=== Phase 9 MinPR screen summary ==="
python3 scripts/report_phase9_minpr_status.py \
    --output-root "$OUTROOT" \
    --target-min-pr "$TARGET_MIN_PR" \
    --include-legacy \
    2>&1 | tee -a "$SUMMARY"
