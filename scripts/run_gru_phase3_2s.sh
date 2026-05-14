#!/usr/bin/env bash
# Phase 3 — Short-Window Baseline: LB-2 AND LB-3 with 1.5s / 2.0s windows
#
# Context:
#   Current Phase 1/2 use target_steps=60 in segment [5.0,9.0)s.
#   At 20fps that is a 3-second window; class 1 (falling) coverage = 4.2%.
#
#   Falling phase duration ≈ 21 frames (1.05s at 20fps).
#   For a window of size W frames to contain ONLY class-1 frames:
#     the window must end before the fallen phase starts.
#   With segment [3.0,9.0)s:
#     W=30 (1.5s): 89% fall-video coverage, mean 15 class-1 windows
#     W=40 (2.0s): 73% fall-video coverage, mean 12 class-1 windows
#   Current W=60: only 4.2% coverage → LB-3 class-1 is nearly unlearnable.
#
# Grid (2 × 2 × LB):
#   P3-v01: LB-2, bidir,       W=30 (1.5s)     ← primary LB-2 baseline
#   P3-v02: LB-2, bidir,       W=40 (2.0s)     ← window size comparison
#   P3-v03: LB-3, bidir,       W=30 (1.5s)     ← primary LB-3 baseline
#   P3-v04: LB-3, bidir,       W=40 (2.0s)     ← window size comparison
#   P3-v05: LB-2, bidir+attn,  W=30 (1.5s)     ← attention variant
#   P3-v06: LB-3, bidir+attn,  W=30 (1.5s)
#   P3-v07: LB-2, bidir+focal, W=30 (1.5s)     ← focal loss for precision
#   P3-v08: LB-3, bidir+focal, W=30 (1.5s)
#
# All: filtered PP, GRU(256,128), conv-pre 2×64k5, kp12

set -uo pipefail

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OUTROOT="results/gru_phase3_2s"
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

# ── Fixed base config ─────────────────────────────────────────────────────────
BASE=(
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --window-start-sec 3.0
    --window-end-sec   9.0
    --feature-set kp12
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --early-stop-patience 15 --epochs 100
    --min-val-precision 0.90
)

GRU=(
    --model-type gru
    --gru-units 256,128
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --bidirectional
)

# v05~v08: unidirectional (stateful 배포 호환)
GRU_UNI=(
    --model-type gru
    --gru-units 256,128
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
)

LB2=(
    --label-column label
)

LB3=(
    --label-column label_3class
    --positive-labels 1,2
    --num-classes 3
)

FOCAL=(
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
)

# ════════════════════════════════════════════════════════════════════════════════
log "=== PHASE 3: Short-window LB-2 and LB-3 ==="

# ── Window size comparison ──────────────────────────────────────────────────
# v01 vs v02: does LB-2 prefer 1.5s or 2.0s?
run_exp P3-v01 "${BASE[@]}" "${GRU[@]}" "${LB2[@]}" --target-steps 30
run_exp P3-v02 "${BASE[@]}" "${GRU[@]}" "${LB2[@]}" --target-steps 40

# v03 vs v04: same for LB-3 (class-1 coverage: 89% vs 73%)
run_exp P3-v03 "${BASE[@]}" "${GRU[@]}" "${LB3[@]}" --target-steps 30
run_exp P3-v04 "${BASE[@]}" "${GRU[@]}" "${LB3[@]}" --target-steps 40

# ── Architecture + focal variants at W=30 (unidirectional — stateful 배포 호환) ──
run_exp P3-v05 "${BASE[@]}" "${GRU_UNI[@]}" "${LB2[@]}" --target-steps 30 \
    --temporal-attention

run_exp P3-v06 "${BASE[@]}" "${GRU_UNI[@]}" "${LB3[@]}" --target-steps 30 \
    --temporal-attention

run_exp P3-v07 "${BASE[@]}" "${GRU_UNI[@]}" "${LB2[@]}" --target-steps 30 \
    "${FOCAL[@]}"

run_exp P3-v08 "${BASE[@]}" "${GRU_UNI[@]}" "${LB3[@]}" --target-steps 30 \
    "${FOCAL[@]}"

# ════════════════════════════════════════════════════════════════════════════════
log "All Phase 3 experiments complete."
echo ""
echo "=== PHASE 3 RESULTS ===" | tee -a "$SUMMARY"
printf "%-8s %-4s %-14s %5s %7s %7s %8s %8s %6s\n" \
    "ID" "LB" "Arch" "W(f)" "testF1" "Rec" "FallP" "NFallP" "MinP" | tee -a "$SUMMARY"
echo "------------------------------------------------------------------------" | tee -a "$SUMMARY"

declare -A LB_MAP=(
    [P3-v01]=2 [P3-v02]=2 [P3-v03]=3 [P3-v04]=3
    [P3-v05]=2 [P3-v06]=3 [P3-v07]=2 [P3-v08]=3
)
declare -A ARCH_MAP=(
    [P3-v01]="bidir"      [P3-v02]="bidir"
    [P3-v03]="bidir"      [P3-v04]="bidir"
    [P3-v05]="uni+attn"   [P3-v06]="uni+attn"
    [P3-v07]="uni+focal"  [P3-v08]="uni+focal"
)
declare -A WIN_MAP=(
    [P3-v01]=30 [P3-v02]=40 [P3-v03]=30 [P3-v04]=40
    [P3-v05]=30 [P3-v06]=30 [P3-v07]=30 [P3-v08]=30
)

for id in P3-v01 P3-v02 P3-v03 P3-v04 P3-v05 P3-v06 P3-v07 P3-v08; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
m = json.load(open('$mfile'))
tv = m['metrics'].get('test_video', {})
ts = m.get('threshold_selection', {})
print('%-8s %-4s %-14s %5d %7.4f %7.4f %8.4f %8.4f %6.4f' % (
    '$id', '${LB_MAP[$id]}', '${ARCH_MAP[$id]}', ${WIN_MAP[$id]},
    tv.get('f1',0), tv.get('recall',0),
    tv.get('precision',0), tv.get('nfall_precision',float('nan')),
    tv.get('min_precision',0),
))
" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "Phase 3 summary written to $SUMMARY"
