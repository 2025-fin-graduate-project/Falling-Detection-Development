#!/usr/bin/env bash
# codex/03 screen - fast GRU LB-2 search
#
# This is a screening pass for the full codex/03 grid. It deliberately disables
# TFLite export, uses larger batches, fewer epochs, and visible epoch logs so
# slow/stalled runs are obvious. Promote the best candidate to a full run only
# after this pass produces video-level metrics.

set -uo pipefail

TARGET_MIN_PRECISION="${TARGET_MIN_PRECISION:-0.92}"
OUTROOT="${OUTROOT:-results/gru_codex03_screen}"
SUMMARY="$OUTROOT/summary.log"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-3}"
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
    --min-val-precision 0.90
    --no-export-tflite
)

WIN30=(--target-steps 30 --window-start-sec 3.0 --window-end-sec 9.0)
WIN40=(--target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0)
WIN60=(--target-steps 60 --window-start-sec 5.0 --window-end-sec 9.0)
FOCAL_STD=(--focal-loss --focal-gamma 2.0 --focal-alpha 0.25)
FOCAL_FP_STRICT=(--focal-loss --focal-gamma 2.0 --focal-alpha 0.15)

log "=== CODEX/03 SCREEN: target MinP >= ${TARGET_MIN_PRECISION}, epochs=${EPOCHS}, batch=${BATCH_SIZE}, eval_stride=${EVAL_STRIDE}, device=${TRAIN_DEVICE:-cpu} ==="

# Fast enough to compare architecture/window effects, still using the full filtered splits.
run_exp C3S-v01 "${COMMON[@]}" "${WIN30[@]}" --feature-set kp7 --gru-units 128,64 --bidirectional "${FOCAL_STD[@]}"
run_exp C3S-v02 "${COMMON[@]}" "${WIN30[@]}" --feature-set kp7 --gru-units 256,128 --bidirectional "${FOCAL_STD[@]}"
run_exp C3S-v03 "${COMMON[@]}" "${WIN40[@]}" --feature-set kp7 --gru-units 128,64 --bidirectional "${FOCAL_STD[@]}"
run_exp C3S-v04 "${COMMON[@]}" "${WIN30[@]}" --feature-set minimal --gru-units 128,64 --bidirectional "${FOCAL_STD[@]}"
run_exp C3S-v05 "${COMMON[@]}" "${WIN30[@]}" --feature-set kp7 --gru-units 128,64 --bidirectional "${FOCAL_FP_STRICT[@]}"
run_exp C3S-v06 "${COMMON[@]}" "${WIN60[@]}" --feature-set kp7 --gru-units 128,64 --bidirectional "${FOCAL_STD[@]}"
run_exp C3S-v07 "${COMMON[@]}" "${WIN30[@]}" --feature-set kp7 --gru-units 128,64 --bidirectional "${FOCAL_STD[@]}" --temporal-attention
run_exp C3S-v08 "${COMMON[@]}" "${WIN30[@]}" --feature-set kp7 --gru-units 128,64 --bidirectional "${FOCAL_STD[@]}" --label-mode last_frame

log "All codex/03 screen experiments complete."
echo "" | tee -a "$SUMMARY"
echo "=== CODEX/03 SCREEN RESULTS ===" | tee -a "$SUMMARY"
printf "%-8s %-18s %7s %7s %8s %8s %6s %6s\n" \
    "ID" "Hypothesis" "testF1" "Rec" "FallP" "NFallP" "MinP" "Pass" | tee -a "$SUMMARY"
echo "----------------------------------------------------------------------------" | tee -a "$SUMMARY"

declare -A HYP=(
    [C3S-v01]="30f-kp7-small"
    [C3S-v02]="30f-kp7-large"
    [C3S-v03]="40f-kp7-small"
    [C3S-v04]="30f-minimal"
    [C3S-v05]="30f-low-alpha"
    [C3S-v06]="60f-kp7-small"
    [C3S-v07]="30f-attn"
    [C3S-v08]="30f-last-frame"
)

for id in C3S-v01 C3S-v02 C3S-v03 C3S-v04 C3S-v05 C3S-v06 C3S-v07 C3S-v08; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
target = float('$TARGET_MIN_PRECISION')
m = json.load(open('$mfile'))
tv = m['metrics'].get('test_video', {})
minp = float(tv.get('min_precision', 0))
print('%-8s %-18s %7.4f %7.4f %8.4f %8.4f %6.4f %6s' % (
    '$id', '${HYP[$id]}', tv.get('f1', 0), tv.get('recall', 0),
    tv.get('precision', 0), tv.get('nfall_precision', 0),
    minp, 'YES' if minp >= target else 'NO',
))
" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "codex/03 screen summary written to $SUMMARY"
