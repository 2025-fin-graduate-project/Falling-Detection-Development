#!/usr/bin/env bash
# TCN Baseline Phase 1
#
# Fixed:
#   dataset   : splits_v2 variants
#   data_scope: all
#   feature   : kp12
#   postproc  : threshold x min_consecutive sweep [1,3,5]
#   deploy    : FP32 + INT8 TFLite export/eval
#
# Variants:
#   PP raw|filtered x LB-2|LB-3
#
# Usage:
#   bash scripts/run_tcn_phase1.sh
#   SMOKE=1 bash scripts/run_tcn_phase1.sh
#   EXTRA_ARGS="--epochs 20 --no-export-tflite" bash scripts/run_tcn_phase1.sh

set -uo pipefail

cd "$(dirname "$0")/.."
source scripts/env_tensorflow_cuda.sh

OUTROOT="results/tcn_baseline_phase1"
SUMMARY="$OUTROOT/summary.log"
mkdir -p "$OUTROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

require_files() {
    local missing=0
    for path in "$@"; do
        if [[ ! -f "$path" ]]; then
            log "MISSING $path"
            missing=1
        fi
    done
    return "$missing"
}

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id already complete"
        return 0
    fi
    log "START $id"
    if uv run python scripts/train_baseline.py \
            --experiment-id "$id" \
            --output-root "$OUTROOT" \
            --quiet \
            "$@" \
            ${SMOKE:+--smoke} \
            ${EXTRA_ARGS:-} \
            2>&1 | tee "$logfile"; then
        log "OK    $id"
    else
        log "FAIL  $id"
        return 1
    fi
}

COMMON=(
    --model-type tcn
    --tcn-channels 32,32,64,96
    --tcn-dilations 1,1,1,1
    --tcn-kernel-size 3
    --feature-set kp12
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --early-stop-patience 15 --epochs 100
    --min-val-precision 0.90
)

RAW_LB2=(
    --preprocessing raw
    --train-csv dataset/splits_v2/train.csv
    --val-csv dataset/splits_v2/val.csv
    --test-csv dataset/splits_v2/test.csv
    --label-column label
)
RAW_LB3=(
    --preprocessing raw
    --train-csv dataset/lb3_v2/train.csv
    --val-csv dataset/lb3_v2/val.csv
    --test-csv dataset/lb3_v2/test.csv
    --label-column label_3class
    --positive-labels 1,2
    --num-classes 3
)
FILT_LB2=(
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv dataset/splits_v2_filtered/val.csv
    --test-csv dataset/splits_v2_filtered/test.csv
    --label-column label
)
FILT_LB3=(
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv dataset/splits_v2_filtered/val.csv
    --test-csv dataset/splits_v2_filtered/test.csv
    --label-column label_3class
    --positive-labels 1,2
    --num-classes 3
)

log "Checking TensorFlow GPU visibility"
bash scripts/check_tensorflow_gpu.sh 2>&1 | tee "$OUTROOT/gpu_check.log"

if require_files dataset/splits_v2/train.csv dataset/splits_v2/val.csv dataset/splits_v2/test.csv; then
    run_exp TCN-P1-v01 "${COMMON[@]}" "${RAW_LB2[@]}"
else
    log "SKIP raw LB-2 because splits_v2 is incomplete"
fi

if require_files dataset/lb3_v2/train.csv dataset/lb3_v2/val.csv dataset/lb3_v2/test.csv; then
    run_exp TCN-P1-v02 "${COMMON[@]}" "${RAW_LB3[@]}"
else
    log "SKIP raw LB-3 because lb3_v2 is incomplete"
fi

if require_files dataset/splits_v2_filtered/train.csv dataset/splits_v2_filtered/val.csv dataset/splits_v2_filtered/test.csv; then
    run_exp TCN-P1-v03 "${COMMON[@]}" "${FILT_LB2[@]}"
    run_exp TCN-P1-v04 "${COMMON[@]}" "${FILT_LB3[@]}"
else
    log "SKIP filtered experiments because splits_v2_filtered is incomplete"
fi

echo "" | tee -a "$SUMMARY"
echo "=== TCN PHASE 1 RESULTS ===" | tee -a "$SUMMARY"
printf "%-12s %-8s %-4s %8s %8s %8s %8s %6s\n" \
    "ID" "PP" "LB" "testF1" "recall" "prec" "valF1" "mincn" | tee -a "$SUMMARY"
echo "--------------------------------------------------------------------" | tee -a "$SUMMARY"

for id in TCN-P1-v01 TCN-P1-v02 TCN-P1-v03 TCN-P1-v04; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 - "$id" "$mfile" <<'PY'
import json
import sys

exp_id, path = sys.argv[1], sys.argv[2]
meta = {
    "TCN-P1-v01": ("raw", "2"),
    "TCN-P1-v02": ("raw", "3"),
    "TCN-P1-v03": ("filtered", "2"),
    "TCN-P1-v04": ("filtered", "3"),
}
m = json.load(open(path))
tv = m["metrics"].get("test_video", {})
vv = m["metrics"].get("val_video", {})
ts = m.get("threshold_selection", {})
pp, lb = meta[exp_id]
print(
    "%-12s %-8s %-4s %8.4f %8.4f %8.4f %8.4f %6d"
    % (
        exp_id,
        pp,
        lb,
        tv.get("f1", 0.0),
        tv.get("recall", 0.0),
        tv.get("precision", 0.0),
        vv.get("f1", 0.0),
        ts.get("min_consecutive", 1),
    )
)
PY
    else
        echo "$id NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "TCN Phase 1 summary written to $SUMMARY"
