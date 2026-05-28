#!/usr/bin/env bash
# GRU Baseline Phase 1  (v01 ~ v16)
#
# 2×2×4 factorial:  PP (raw|filtered) × LB (2|3) × Arch (base|bidir|attn|bidir+attn)
#
# Fixed (not variables):
#   dataset   : splits_v2  (C5~C8 only, outlier-removed)
#   data_scope: all
#   feature   : kp12
#   model     : GRU(256,128) + Conv1D-pre 2×64×k5
#   postproc  : consecutive+threshold sweep [1,3,5] (default)
#
# Variables:
#   PP    raw → v01–v08   /  filtered → v09–v16
#   LB    2   → v01–v04, v09–v12   /  3 → v05–v08, v13–v16
#   Arch  1=base  2=bidir  3=attn  4=bidir+attn
#
# v01: raw  LB-2 base      v09: filt LB-2 base
# v02: raw  LB-2 bidir     v10: filt LB-2 bidir
# v03: raw  LB-2 attn      v11: filt LB-2 attn
# v04: raw  LB-2 bidir+attn  v12: filt LB-2 bidir+attn
# v05: raw  LB-3 base      v13: filt LB-3 base
# v06: raw  LB-3 bidir     v14: filt LB-3 bidir
# v07: raw  LB-3 attn      v15: filt LB-3 attn
# v08: raw  LB-3 bidir+attn  v16: filt LB-3 bidir+attn

set -uo pipefail

# ── GPU: expose nvidia pip-package CUDA libs to TensorFlow ───────────────────
_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# ─────────────────────────────────────────────────────────────────────────────

OUTROOT="results/gru_baseline_phase1"
SUMMARY="$OUTROOT/summary.log"
mkdir -p "$OUTROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id — already complete"
        return 0
    fi
    log "START $id"
    if uv run python scripts/train_baseline.py \
            --experiment-id "$id" \
            --output-root   "$OUTROOT" \
            --quiet \
            "$@" \
            2>&1 | tee "$logfile"; then
        log "OK    $id"
    else
        log "FAIL  $id"
    fi
}

# ── Shared hyperparameters ────────────────────────────────────────────────────
COMMON=(
    --model-type gru
    --gru-units 256,128
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --feature-set kp12
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --early-stop-patience 15 --epochs 100
    --min-val-precision 0.90
)

# ── Dataset arrays ─────────────────────────────────────────────────────────────
RAW_LB2=(
    --preprocessing raw
    --train-csv dataset/splits_v2/train.csv
    --val-csv   dataset/splits_v2/val.csv
    --test-csv  dataset/splits_v2/test.csv
    --label-column label
)
RAW_LB3=(
    --preprocessing raw
    --train-csv dataset/lb3_v2/train.csv
    --val-csv   dataset/lb3_v2/val.csv
    --test-csv  dataset/lb3_v2/test.csv
    --label-column label_3class
    --positive-labels 1,2
    --num-classes 3
)
FILT_LB2=(
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label
)
FILT_LB3=(
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label_3class
    --positive-labels 1,2
    --num-classes 3
)

# ════════════════════════════════════════════════════════════════════════════════
# Group A: raw + LB-2  (v01–v04)
# ════════════════════════════════════════════════════════════════════════════════
log "=== GROUP A: raw + LB-2 ==="

run_exp P1-v01 "${COMMON[@]}" "${RAW_LB2[@]}"

run_exp P1-v02 "${COMMON[@]}" "${RAW_LB2[@]}" \
    --bidirectional

run_exp P1-v03 "${COMMON[@]}" "${RAW_LB2[@]}" \
    --temporal-attention

run_exp P1-v04 "${COMMON[@]}" "${RAW_LB2[@]}" \
    --bidirectional --temporal-attention

# ════════════════════════════════════════════════════════════════════════════════
# Group B: raw + LB-3  (v05–v08)
# ════════════════════════════════════════════════════════════════════════════════
log "=== GROUP B: raw + LB-3 ==="

run_exp P1-v05 "${COMMON[@]}" "${RAW_LB3[@]}"

run_exp P1-v06 "${COMMON[@]}" "${RAW_LB3[@]}" \
    --bidirectional

run_exp P1-v07 "${COMMON[@]}" "${RAW_LB3[@]}" \
    --temporal-attention

run_exp P1-v08 "${COMMON[@]}" "${RAW_LB3[@]}" \
    --bidirectional --temporal-attention

# ════════════════════════════════════════════════════════════════════════════════
# Wait for splits_v2_filtered to be built before filtered experiments
# ════════════════════════════════════════════════════════════════════════════════
log "Waiting for dataset/splits_v2_filtered/test.csv ..."
while [[ ! -f "dataset/splits_v2_filtered/test.csv" ]]; do
    sleep 30
done
log "splits_v2_filtered ready — starting filtered experiments"

# ════════════════════════════════════════════════════════════════════════════════
# Group C: filtered + LB-2  (v09–v12)
# ════════════════════════════════════════════════════════════════════════════════
log "=== GROUP C: filtered + LB-2 ==="

run_exp P1-v09 "${COMMON[@]}" "${FILT_LB2[@]}"

run_exp P1-v10 "${COMMON[@]}" "${FILT_LB2[@]}" \
    --bidirectional

run_exp P1-v11 "${COMMON[@]}" "${FILT_LB2[@]}" \
    --temporal-attention

run_exp P1-v12 "${COMMON[@]}" "${FILT_LB2[@]}" \
    --bidirectional --temporal-attention

# ════════════════════════════════════════════════════════════════════════════════
# Group D: filtered + LB-3  (v13–v16)
# ════════════════════════════════════════════════════════════════════════════════
log "=== GROUP D: filtered + LB-3 ==="

run_exp P1-v13 "${COMMON[@]}" "${FILT_LB3[@]}"

run_exp P1-v14 "${COMMON[@]}" "${FILT_LB3[@]}" \
    --bidirectional

run_exp P1-v15 "${COMMON[@]}" "${FILT_LB3[@]}" \
    --temporal-attention

run_exp P1-v16 "${COMMON[@]}" "${FILT_LB3[@]}" \
    --bidirectional --temporal-attention

# ════════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════════
log "All Phase 1 experiments complete."
echo "" | tee -a "$SUMMARY"
echo "=== PHASE 1 RESULTS ===" | tee -a "$SUMMARY"

printf "%-10s %-4s %-4s %-12s %7s %7s %8s %8s %8s %6s %6s\n" \
    "ID" "PP" "LB" "Arch" "testF1" "Rec" "FallPrec" "NFallPrc" "MinPrec" "valF1" "mincn" | tee -a "$SUMMARY"
echo "-------------------------------------------------------------------------------" | tee -a "$SUMMARY"

declare -A PP_MAP=([P1-v01]=raw [P1-v02]=raw [P1-v03]=raw [P1-v04]=raw
                   [P1-v05]=raw [P1-v06]=raw [P1-v07]=raw [P1-v08]=raw
                   [P1-v09]=filt [P1-v10]=filt [P1-v11]=filt [P1-v12]=filt
                   [P1-v13]=filt [P1-v14]=filt [P1-v15]=filt [P1-v16]=filt)
declare -A LB_MAP=([P1-v01]=2 [P1-v02]=2 [P1-v03]=2 [P1-v04]=2
                   [P1-v05]=3 [P1-v06]=3 [P1-v07]=3 [P1-v08]=3
                   [P1-v09]=2 [P1-v10]=2 [P1-v11]=2 [P1-v12]=2
                   [P1-v13]=3 [P1-v14]=3 [P1-v15]=3 [P1-v16]=3)
declare -A ARCH_MAP=([P1-v01]=base   [P1-v02]=bidir  [P1-v03]=attn   [P1-v04]=bidir+attn
                     [P1-v05]=base   [P1-v06]=bidir  [P1-v07]=attn   [P1-v08]=bidir+attn
                     [P1-v09]=base   [P1-v10]=bidir  [P1-v11]=attn   [P1-v12]=bidir+attn
                     [P1-v13]=base   [P1-v14]=bidir  [P1-v15]=attn   [P1-v16]=bidir+attn)

for id in P1-v01 P1-v02 P1-v03 P1-v04 P1-v05 P1-v06 P1-v07 P1-v08 \
          P1-v09 P1-v10 P1-v11 P1-v12 P1-v13 P1-v14 P1-v15 P1-v16; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
m = json.load(open('$mfile'))
tv = m['metrics'].get('test_video', {})
vv = m['metrics'].get('val_video', {})
ts = m.get('threshold_selection', {})
fall_p  = tv.get('precision', 0)
nfall_p = tv.get('nfall_precision', float('nan'))
min_p   = tv.get('min_precision', fall_p)
print(f'%-10s %-4s %-4s %-12s %7.4f %7.4f %8.4f %8.4f %8.4f %6.4f %6d' % (
    '$id', '${PP_MAP[$id]}', '${LB_MAP[$id]}', '${ARCH_MAP[$id]}',
    tv.get('f1', 0), tv.get('recall', 0), fall_p, nfall_p, min_p,
    vv.get('f1', 0), ts.get('min_consecutive', 1)
))
" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "Phase 1 summary written to $SUMMARY"
