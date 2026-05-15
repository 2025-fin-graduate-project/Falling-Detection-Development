#!/usr/bin/env bash
# Phase 7 — Compact Quantizable GRU
#
# Goal: INT8 TFLite model < 512 KB that achieves MinP ≥ 0.93 on STM32N6.
#
# Why P3-v02 (MinP=0.9556) is NOT deployable:
#   GRU(256,128) bidir, 40f, unroll=True for TFLite → 4,740 KB INT8
#   STM32N6 SRAM budget ≈ 1 MB → fail.
#
# INT8 model size ≈ weights + unrolled-graph-ops
#   = f(units², n_features, target_steps)
#
# Size estimates (unidirectional, unroll=True):
#   GRU(128,64) + kp7  (27 feat) + 40f → ~350 KB  [target range]
#   GRU(128,64) + kp7  (27 feat) + 30f → ~280 KB  [safer]
#   GRU(128,64) + min  (15 feat) + 30f → ~220 KB  [lean]
#   GRU(64,32)  + kp7  (27 feat) + 30f → ~130 KB  [very compact]
#   GRU(64,32)  + min  (15 feat) + 20f → ~60  KB  [ultra-compact]
#
# Performance estimate (no bidir costs ~-0.01 vs bidir; smaller units ~-0.02):
#   GRU(128,64) 40f kp7  → estimated MinP ~0.93  (P3-v02 0.9556 - bidir - units)
#   GRU(128,64) 30f kp7  → estimated MinP ~0.92
#   GRU(64,32)  40f kp7  → estimated MinP ~0.90
#
# Experiments (6개):
#   Q7-v01: GRU(128,64) no-bidir, kp7,     40f  ← primary target
#   Q7-v02: GRU(128,64) no-bidir, kp7,     30f
#   Q7-v03: GRU(128,64) no-bidir, minimal, 40f  ← lightest that may hit 0.93
#   Q7-v04: GRU(64,32)  no-bidir, kp7,     40f  ← very compact
#   Q7-v05: GRU(64,32)  no-bidir, kp7,     30f
#   Q7-v06: GRU(64,32)  no-bidir, minimal, 30f  ← ultra-compact baseline

set -uo pipefail

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OUTROOT="results/gru_compact_quant"
SUMMARY="$OUTROOT/summary.log"
mkdir -p "$OUTROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id — already complete"; return 0
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

# ── Fixed ─────────────────────────────────────────────────────────────────────
BASE=(
    --model-type gru
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --early-stop-patience 15 --epochs 100
    --min-val-precision 0.90
    # NOTE: no --bidirectional → unidirectional GRU
    # unroll=True applied at export time by reexport_tflite.py
)

WIN40=(--target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0)
WIN30=(--target-steps 30 --window-start-sec 3.0 --window-end-sec 9.0)
WIN20=(--target-steps 20 --window-start-sec 3.0 --window-end-sec 9.0)

# ════════════════════════════════════════════════════════════════════════════════
log "=== Phase 7: Compact Quantizable GRU ==="

# Q7-v01: GRU(128,64) + kp7 + 40f  — primary target
run_exp Q7-v01 "${BASE[@]}" --gru-units 128,64 "${WIN40[@]}" --feature-set kp7

# Q7-v02: GRU(128,64) + kp7 + 30f
run_exp Q7-v02 "${BASE[@]}" --gru-units 128,64 "${WIN30[@]}" --feature-set kp7

# Q7-v03: GRU(128,64) + minimal + 40f
run_exp Q7-v03 "${BASE[@]}" --gru-units 128,64 "${WIN40[@]}" --feature-set minimal

# Q7-v04: GRU(64,32) + kp7 + 40f
run_exp Q7-v04 "${BASE[@]}" --gru-units 64,32  "${WIN40[@]}" --feature-set kp7

# Q7-v05: GRU(64,32) + kp7 + 30f
run_exp Q7-v05 "${BASE[@]}" --gru-units 64,32  "${WIN30[@]}" --feature-set kp7

# Q7-v06: GRU(64,32) + minimal + 30f  — ultra-compact
run_exp Q7-v06 "${BASE[@]}" --gru-units 64,32  "${WIN30[@]}" --feature-set minimal

# ════════════════════════════════════════════════════════════════════════════════
log "All Phase 7 experiments complete."
echo ""
echo "=== PHASE 7 RESULTS ===" | tee -a "$SUMMARY"
printf "%-8s %-12s %-8s %4s %7s %7s %8s %8s %6s %10s\n" \
    "ID" "Units" "KP" "Win" "testF1" "Rec" "FallP" "NFallP" "MinP" "INT8_KB" | tee -a "$SUMMARY"
echo "------------------------------------------------------------------------------" | tee -a "$SUMMARY"
echo "P3-v02   (256,128)bidir kp12(13) 40f  0.9704 0.9850  0.9563  0.9556  0.9556  4740 [ref, too large]" | tee -a "$SUMMARY"

declare -A UNITS_MAP=(
    [Q7-v01]="(128,64)" [Q7-v02]="(128,64)" [Q7-v03]="(128,64)"
    [Q7-v04]="(64,32)"  [Q7-v05]="(64,32)"  [Q7-v06]="(64,32)"
)
declare -A KP_MAP=(
    [Q7-v01]=kp7 [Q7-v02]=kp7 [Q7-v03]=minimal
    [Q7-v04]=kp7 [Q7-v05]=kp7 [Q7-v06]=minimal
)
declare -A WIN_MAP=(
    [Q7-v01]=40 [Q7-v02]=30 [Q7-v03]=40
    [Q7-v04]=40 [Q7-v05]=30 [Q7-v06]=30
)

for id in Q7-v01 Q7-v02 Q7-v03 Q7-v04 Q7-v05 Q7-v06; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
m = json.load(open('$mfile'))
tv = m['metrics'].get('test_video', {})
ep = m.get('export_paths', {})
int8_kb = ep.get('model_int8_size_kb', 0)
mark = ' ★' if tv.get('min_precision',0) >= 0.93 else ''
size_ok = ' ✓' if 0 < int8_kb < 512 else (' ⚠' if int8_kb > 0 else ' ?')
print('%-8s %-12s %-8s %4d %7.4f %7.4f %8.4f %8.4f %6.4f %8.1f%s%s' % (
    '$id', '${UNITS_MAP[$id]}', '${KP_MAP[$id]}', ${WIN_MAP[$id]},
    tv.get('f1',0), tv.get('recall',0),
    tv.get('precision',0), tv.get('nfall_precision',0),
    tv.get('min_precision',0), int8_kb, mark, size_ok
))
" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "Phase 7 summary written to $SUMMARY"
