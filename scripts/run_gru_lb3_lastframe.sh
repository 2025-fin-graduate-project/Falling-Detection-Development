#!/usr/bin/env bash
# LB-3 Last-Frame Labeling Experiments (LF-v01 ~ LF-v08)
#
# Purpose: Train LB-3 models with --label-mode last_frame so that
#   class 1 (falling) windows are learnable.
#   With segment_max, class 1 windows are ~0.1% (falling lasts 21 frames < 60-frame window).
#   With last_frame, class 1 windows = ~21 per fall video → ~133K in train set.
#
# 2 × 4 factorial: PP (raw|filtered) × Arch (base|bidir|attn|bidir+attn)
# LB fixed at 3 (label_3class, positive_labels=1,2)
# label_mode fixed at last_frame
#
# LF-v01: raw   base
# LF-v02: raw   bidir
# LF-v03: raw   attn
# LF-v04: raw   bidir+attn
# LF-v05: filt  base
# LF-v06: filt  bidir
# LF-v07: filt  attn
# LF-v08: filt  bidir+attn

set -uo pipefail

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OUTROOT="results/gru_lb3_lastframe"
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
    --label-mode last_frame
    --label-column label_3class
    --positive-labels 1,2
    --num-classes 3
)

RAW=(
    --preprocessing raw
    --train-csv dataset/lb3_v2/train.csv
    --val-csv   dataset/lb3_v2/val.csv
    --test-csv  dataset/lb3_v2/test.csv
)

FILT=(
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
)

# ════════════════════════════════════════════════════════════════════════════════
log "=== LF GROUP A: raw + last_frame ==="
run_exp LF-v01 "${COMMON[@]}" "${RAW[@]}"
run_exp LF-v02 "${COMMON[@]}" "${RAW[@]}" --bidirectional
run_exp LF-v03 "${COMMON[@]}" "${RAW[@]}" --temporal-attention
run_exp LF-v04 "${COMMON[@]}" "${RAW[@]}" --bidirectional --temporal-attention

# ════════════════════════════════════════════════════════════════════════════════
log "=== LF GROUP B: filtered + last_frame ==="
run_exp LF-v05 "${COMMON[@]}" "${FILT[@]}"
run_exp LF-v06 "${COMMON[@]}" "${FILT[@]}" --bidirectional
run_exp LF-v07 "${COMMON[@]}" "${FILT[@]}" --temporal-attention
run_exp LF-v08 "${COMMON[@]}" "${FILT[@]}" --bidirectional --temporal-attention

# ════════════════════════════════════════════════════════════════════════════════
log "All LF experiments complete."
echo ""
echo "=== LAST-FRAME LB-3 RESULTS ===" | tee -a "$SUMMARY"

printf "%-8s %-4s %-12s %7s %7s %8s %8s %8s %6s\n" \
    "ID" "PP" "Arch" "testF1" "Rec" "FallPrec" "NFallPrc" "MinPrec" "mincn" | tee -a "$SUMMARY"
echo "------------------------------------------------------------------------" | tee -a "$SUMMARY"

declare -A PP_MAP=([LF-v01]=raw  [LF-v02]=raw  [LF-v03]=raw  [LF-v04]=raw
                   [LF-v05]=filt [LF-v06]=filt [LF-v07]=filt [LF-v08]=filt)
declare -A ARCH_MAP=([LF-v01]=base [LF-v02]=bidir [LF-v03]=attn [LF-v04]=bidir+attn
                     [LF-v05]=base [LF-v06]=bidir [LF-v07]=attn [LF-v08]=bidir+attn)

for id in LF-v01 LF-v02 LF-v03 LF-v04 LF-v05 LF-v06 LF-v07 LF-v08; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
m = json.load(open('$mfile'))
tv = m['metrics'].get('test_video', {})
ts = m.get('threshold_selection', {})
print(f'%-8s %-4s %-12s %7.4f %7.4f %8.4f %8.4f %8.4f %6d' % (
    '$id', '${PP_MAP[$id]}', '${ARCH_MAP[$id]}',
    tv.get('f1', 0), tv.get('recall', 0),
    tv.get('precision', 0), tv.get('nfall_precision', float('nan')),
    tv.get('min_precision', tv.get('precision', 0)),
    ts.get('min_consecutive', 1)
))
" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "Summary written to $SUMMARY"
