#!/usr/bin/env bash
# codex/05 - overnight deployable candidate sweep.
#
# Runs sequentially for at most ~6 hours by default. This intentionally avoids
# parallel training and disables GPU by default so the machine is not fully
# loaded overnight. Candidate set includes deployable unidirectional GRU and
# light TCN variants. GRU final runtime is expected to be exported as an
# explicit-state step model; sequence TFLite/int8 is a quantization smoke test.

set -uo pipefail

OUTROOT="${OUTROOT:-results/codex05_overnight_candidates}"
SUMMARY="$OUTROOT/summary.log"
TARGET_MIN_PRECISION="${TARGET_MIN_PRECISION:-0.92}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-21600}"
PER_EXP_TIMEOUT_SECONDS="${PER_EXP_TIMEOUT_SECONDS:-5400}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_STRIDE="${EVAL_STRIDE:-2}"

if [[ "${TRAIN_DEVICE:-cpu}" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTROOT"
START_TS=$(date +%s)
DEADLINE_TS=$((START_TS + TIME_BUDGET_SECONDS))

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

remaining_seconds() {
    local now
    now=$(date +%s)
    echo $((DEADLINE_TS - now))
}

run_exp() {
    local id="$1"; shift
    local remain
    remain=$(remaining_seconds)
    if (( remain <= 300 )); then
        log "STOP  $id - time budget exhausted"
        return 0
    fi

    local timeout_s="$PER_EXP_TIMEOUT_SECONDS"
    if (( remain < timeout_s )); then
        timeout_s="$remain"
    fi

    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id - already complete"
        return 0
    fi

    log "START $id timeout=${timeout_s}s remaining=${remain}s"
    if timeout --kill-after=60s "$timeout_s" \
        uv run python scripts/train_baseline.py \
            --experiment-id "$id" \
            --output-root "$OUTROOT" \
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
    --train-negative-stride 4
    --eval-stride "$EVAL_STRIDE"
    --epochs "$EPOCHS" --early-stop-patience "$PATIENCE" --batch-size "$BATCH_SIZE"
    --min-val-precision "$TARGET_MIN_PRECISION"
    --representative-samples 512
    --quant-eval-max-windows 0
)

GRU_BASE=(
    --model-type gru
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --gru-unroll
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
)

TCN_LIGHT=(
    --model-type tcn
    --tcn-channels 32,64,64,96 --tcn-dilations 1,2,4,8 --tcn-kernel-size 3
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
)

WIN30=(--target-steps 30 --window-start-sec 3.0 --window-end-sec 9.0)
WIN40=(--target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0)

log "=== CODEX/05 OVERNIGHT: budget=${TIME_BUDGET_SECONDS}s per_exp=${PER_EXP_TIMEOUT_SECONDS}s device=${TRAIN_DEVICE:-cpu} ==="

run_exp C5O-v01 "${COMMON[@]}" "${GRU_BASE[@]}" "${WIN30[@]}" --feature-set kp7 --gru-units 128,64
run_exp C5O-v02 "${COMMON[@]}" "${GRU_BASE[@]}" "${WIN30[@]}" --feature-set kp7 --gru-units 96,48
run_exp C5O-v03 "${COMMON[@]}" "${GRU_BASE[@]}" "${WIN40[@]}" --feature-set kp7 --gru-units 128,64
run_exp C5O-v04 "${COMMON[@]}" "${TCN_LIGHT[@]}" "${WIN30[@]}" --feature-set kp7
run_exp C5O-v05 "${COMMON[@]}" "${TCN_LIGHT[@]}" "${WIN30[@]}" --feature-set minimal
run_exp C5O-v06 "${COMMON[@]}" "${TCN_LIGHT[@]}" "${WIN40[@]}" --feature-set kp7

log "All scheduled codex/05 overnight candidates finished or timed out."
echo "" | tee -a "$SUMMARY"
echo "=== CODEX/05 OVERNIGHT RESULTS ===" | tee -a "$SUMMARY"
printf "%-8s %-24s %7s %8s %8s %6s %9s %9s\n" \
    "ID" "Candidate" "F1" "FallP" "NFallP" "MinP" "INT8_F1" "INT8_MinP" | tee -a "$SUMMARY"
echo "------------------------------------------------------------------------------------" | tee -a "$SUMMARY"

for id in C5O-v01 C5O-v02 C5O-v03 C5O-v04 C5O-v05 C5O-v06; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
m=json.load(open('$mfile'))
tv=m['metrics'].get('test_video', {})
qv=m['metrics'].get('test_int8_video', {})
print('%-8s %-24s %7.4f %8.4f %8.4f %6.4f %9.4f %9.4f' % (
 '$id', '$id', tv.get('f1',0), tv.get('precision',0), tv.get('nfall_precision',0), tv.get('min_precision',0),
 qv.get('f1',0), qv.get('min_precision',0)
))
" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "codex/05 overnight summary written to $SUMMARY"
