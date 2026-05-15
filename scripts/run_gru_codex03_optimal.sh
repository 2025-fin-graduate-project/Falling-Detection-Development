#!/usr/bin/env bash
# codex/03 - GRU LB-2 precision search
#
# Goal:
#   Find a GRU-based model with
#   min(Fall precision, Non-Fall precision) >= 0.92.
#
# Core ideas from docs:
#   - filtered splits are the primary dataset for high-precision GRU runs
#   - conv-pre must stay enabled
#   - short windows can better isolate the falling interval
#   - reducing noisy keypoints may improve precision
#   - focal loss and threshold/min-consecutive sweep are useful precision tools
#
# Added hypotheses in this runner:
#   - a 40-frame middle window can be a precision/coverage compromise
#   - stricter validation precision selection can improve test MinP
#   - lower positive focal alpha can reduce false positives and improve FallP
#   - temporal attention may help recover recall after keypoint reduction
#   - last-frame labels may reduce ambiguous windows and improve precision

set -uo pipefail

TARGET_MIN_PRECISION="${TARGET_MIN_PRECISION:-0.92}"

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OUTROOT="${OUTROOT:-results/gru_codex03_optimal}"
SUMMARY="$OUTROOT/summary.log"
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
    --model-type gru
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --early-stop-patience 15 --epochs 100
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --min-val-precision 0.90
)

GRU_LARGE=(--gru-units 256,128 --bidirectional)
GRU_XL=(--gru-units 512,256 --bidirectional)
FOCAL_STD=(--focal-loss --focal-gamma 2.0 --focal-alpha 0.25)
FOCAL_FP_STRICT=(--focal-loss --focal-gamma 2.0 --focal-alpha 0.15)
WIN30=(--target-steps 30 --window-start-sec 3.0 --window-end-sec 9.0)
WIN40=(--target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0)
WIN60=(--target-steps 60 --window-start-sec 5.0 --window-end-sec 9.0)
VALP92=(--min-val-precision "$TARGET_MIN_PRECISION")

log "=== CODEX/03: GRU precision search, target MinP >= ${TARGET_MIN_PRECISION} ==="

# Group A: document-backed keypoint/window search.
run_exp C3-v01 "${COMMON[@]}" "${GRU_LARGE[@]}" "${FOCAL_STD[@]}" "${WIN30[@]}" --feature-set kp7
run_exp C3-v02 "${COMMON[@]}" "${GRU_LARGE[@]}" "${FOCAL_STD[@]}" "${WIN30[@]}" --feature-set minimal
run_exp C3-v03 "${COMMON[@]}" "${GRU_XL[@]}"    "${FOCAL_STD[@]}" "${WIN30[@]}" --feature-set kp12
run_exp C3-v04 "${COMMON[@]}" "${GRU_LARGE[@]}" "${FOCAL_STD[@]}" "${WIN60[@]}" --feature-set kp7

# Group B: precision-oriented hypotheses.
run_exp C3-v05 "${COMMON[@]}" "${GRU_LARGE[@]}" "${FOCAL_STD[@]}"       "${WIN40[@]}" --feature-set kp7
run_exp C3-v06 "${COMMON[@]}" "${GRU_LARGE[@]}" "${FOCAL_STD[@]}"       "${WIN30[@]}" --feature-set kp7 --temporal-attention
run_exp C3-v07 "${COMMON[@]}" "${GRU_LARGE[@]}" "${FOCAL_FP_STRICT[@]}" "${WIN30[@]}" --feature-set kp7
run_exp C3-v08 "${COMMON[@]}" "${GRU_LARGE[@]}" "${FOCAL_STD[@]}"       "${WIN30[@]}" --feature-set kp7 "${VALP92[@]}"
run_exp C3-v09 "${COMMON[@]}" "${GRU_LARGE[@]}" "${FOCAL_FP_STRICT[@]}" "${WIN40[@]}" --feature-set kp7 "${VALP92[@]}"
run_exp C3-v10 "${COMMON[@]}" "${GRU_LARGE[@]}" "${FOCAL_STD[@]}"       "${WIN30[@]}" --feature-set kp7 --label-mode last_frame

log "All codex/03 experiments complete."
echo "" | tee -a "$SUMMARY"
echo "=== CODEX/03 RESULTS ===" | tee -a "$SUMMARY"
printf "%-8s %-20s %5s %-8s %-9s %7s %7s %8s %8s %6s %6s\n" \
    "ID" "Hypothesis" "Win" "KP" "GRU" "testF1" "Rec" "FallP" "NFallP" "MinP" "Pass" | tee -a "$SUMMARY"
echo "------------------------------------------------------------------------------------------------" | tee -a "$SUMMARY"

declare -A HYP=(
    [C3-v01]="30f-kp7"
    [C3-v02]="30f-minimal"
    [C3-v03]="30f-xl-kp12"
    [C3-v04]="60f-kp7"
    [C3-v05]="40f-kp7"
    [C3-v06]="30f-kp7-attn"
    [C3-v07]="30f-low-alpha"
    [C3-v08]="30f-valp92"
    [C3-v09]="40f-low-alpha-valp92"
    [C3-v10]="30f-last-frame"
)
declare -A WIN=([C3-v01]=30 [C3-v02]=30 [C3-v03]=30 [C3-v04]=60 [C3-v05]=40 [C3-v06]=30 [C3-v07]=30 [C3-v08]=30 [C3-v09]=40 [C3-v10]=30)
declare -A KP=([C3-v01]=kp7 [C3-v02]=minimal [C3-v03]=kp12 [C3-v04]=kp7 [C3-v05]=kp7 [C3-v06]=kp7 [C3-v07]=kp7 [C3-v08]=kp7 [C3-v09]=kp7 [C3-v10]=kp7)
declare -A GRU=([C3-v01]=256,128 [C3-v02]=256,128 [C3-v03]=512,256 [C3-v04]=256,128 [C3-v05]=256,128 [C3-v06]=256,128 [C3-v07]=256,128 [C3-v08]=256,128 [C3-v09]=256,128 [C3-v10]=256,128)

for id in C3-v01 C3-v02 C3-v03 C3-v04 C3-v05 C3-v06 C3-v07 C3-v08 C3-v09 C3-v10; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
target = float('$TARGET_MIN_PRECISION')
m = json.load(open('$mfile'))
tv = m['metrics'].get('test_video', {})
minp = float(tv.get('min_precision', 0))
print('%-8s %-20s %5d %-8s %-9s %7.4f %7.4f %8.4f %8.4f %6.4f %6s' % (
    '$id', '${HYP[$id]}', ${WIN[$id]}, '${KP[$id]}', '${GRU[$id]}',
    tv.get('f1', 0), tv.get('recall', 0),
    tv.get('precision', 0), tv.get('nfall_precision', 0),
    minp, 'YES' if minp >= target else 'NO',
))
" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

python3 -c "
import glob, json, os
rows = []
for path in glob.glob('$OUTROOT/*/metrics.json'):
    with open(path) as fh:
        m = json.load(fh)
    tv = m.get('metrics', {}).get('test_video', {})
    rows.append((float(tv.get('min_precision', 0)), float(tv.get('f1', 0)), os.path.basename(os.path.dirname(path)), tv))
if rows:
    minp, f1, exp_id, tv = sorted(rows, reverse=True)[0]
    print('')
    print('BEST %s  F1=%.4f  MinP=%.4f  FallP=%.4f  NFallP=%.4f' % (
        exp_id, f1, minp, tv.get('precision', 0), tv.get('nfall_precision', 0)
    ))
" | tee -a "$SUMMARY"

log "codex/03 summary written to $SUMMARY"
