#!/usr/bin/env bash
# codex/04 - STM32-oriented GRU search.
#
# Target:
#   - Keras/video-level F1 around 0.93~0.95+
#   - INT8/video-level F1 around 0.90~0.91+
#   - unidirectional GRU only; bidirectional models are excluded
#   - sequence TFLite/int8 checks are used as a quantization smoke test
#   - final STM32 runtime is stateless training weights exported to an
#     explicit-state step model; firmware maintains h1/h2 and resets state
#     after a fall alarm or reset condition

set -uo pipefail

TARGET_MIN_PRECISION="${TARGET_MIN_PRECISION:-0.92}"
OUTROOT="${OUTROOT:-results/gru_codex04_deploy}"
SUMMARY="$OUTROOT/summary.log"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
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
            "$@" \
            2>&1 | tee "$logfile"; then
        log "OK    $id"
    else
        log "FAIL  $id"
    fi
}

COMMON=(
    --model-type gru
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 4
    --eval-stride "$EVAL_STRIDE"
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --epochs "$EPOCHS" --early-stop-patience "$PATIENCE" --batch-size "$BATCH_SIZE"
    --min-val-precision "$TARGET_MIN_PRECISION"
    --gru-unroll
    --representative-samples 512
    --quant-eval-max-windows 0
)

WIN30=(--target-steps 30 --window-start-sec 3.0 --window-end-sec 9.0)
WIN40=(--target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0)
FOCAL_STD=(--focal-loss --focal-gamma 2.0 --focal-alpha 0.25)

log "=== CODEX/04 DEPLOY: target MinP >= ${TARGET_MIN_PRECISION}, epochs=${EPOCHS}, batch=${BATCH_SIZE}, device=${TRAIN_DEVICE:-cpu} ==="

run_exp C4D-v01 "${COMMON[@]}" "${WIN30[@]}" --feature-set kp7 --gru-units 128,64 "${FOCAL_STD[@]}"
run_exp C4D-v02 "${COMMON[@]}" "${WIN30[@]}" --feature-set kp7 --gru-units 96,48 "${FOCAL_STD[@]}"
run_exp C4D-v03 "${COMMON[@]}" "${WIN30[@]}" --feature-set minimal --gru-units 128,64 "${FOCAL_STD[@]}"
run_exp C4D-v04 "${COMMON[@]}" "${WIN40[@]}" --feature-set kp7 --gru-units 128,64 "${FOCAL_STD[@]}"
run_exp C4D-v05 "${COMMON[@]}" "${WIN30[@]}" --feature-set kp7 --gru-units 128,64

log "All codex/04 deploy experiments complete."
echo "" | tee -a "$SUMMARY"
echo "=== CODEX/04 DEPLOY RESULTS ===" | tee -a "$SUMMARY"
printf "%-8s %-20s %7s %8s %8s %6s %9s %9s\n" \
    "ID" "Hypothesis" "F1" "FallP" "NFallP" "MinP" "INT8_F1" "INT8_MinP" | tee -a "$SUMMARY"
echo "--------------------------------------------------------------------------------" | tee -a "$SUMMARY"

for id in C4D-v01 C4D-v02 C4D-v03 C4D-v04 C4D-v05; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
m=json.load(open('$mfile'))
tv=m['metrics'].get('test_video', {})
qv=m['metrics'].get('test_int8_video', {})
print('%-8s %-20s %7.4f %8.4f %8.4f %6.4f %9.4f %9.4f' % (
 '$id', '$id', tv.get('f1',0), tv.get('precision',0), tv.get('nfall_precision',0), tv.get('min_precision',0),
 qv.get('f1',0), qv.get('min_precision',0)
))
" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "codex/04 deploy summary written to $SUMMARY"
