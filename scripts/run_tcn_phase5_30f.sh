#!/usr/bin/env bash
# Phase 5 — TCN+focal + 30f window: targeting MinP ≥ 0.93
#
# Hypothesis: combining Phase 2 best arch (TCN+focal) with Phase 3 insight
# (30f window, [3.0,9.0)s) will force the model to learn actual falling
# motion rather than post-fall static posture.
#
# Phase 2 best (60f): TCN[64,64,128,128]+focal  F1=0.9253  MinP=0.9044
# Phase 3 insight: 30f window gives 89% class-1 coverage vs 4.2% with 60f
#
# Variables:
#   v01: standard channels [64,64,128,128], kp12
#   v02: larger channels   [128,128,256,256], kp12        ← more capacity
#   v03: wider dilations   [1,2,4,8,16,32], kp12          ← longer context
#   v04: standard + kp7 (7 kp, docs recommended)
#   v05: standard + kp_minimal (5 kp, lightest)
#   v06: larger channels + kp7                            ← capacity + space
#
# All: TCN+focal, filtered, LB-2, 30f, [3.0,9.0)s

set -uo pipefail

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OUTROOT="results/tcn_phase5_30f"
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

# ── Fixed: TCN+focal, filtered, LB-2, 30f, [3.0,9.0)s ───────────────────────
BASE=(
    --model-type tcn
    --tcn-dilations 1,2,4,8 --tcn-kernel-size 3
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label
    --data-scope all
    --target-steps 30
    --window-start-sec 3.0
    --window-end-sec   9.0
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --early-stop-patience 15 --epochs 100
    --min-val-precision 0.90
)

CH_STD=(--tcn-channels 64,64,128,128)
CH_LRG=(--tcn-channels 128,128,256,256)

# ════════════════════════════════════════════════════════════════════════════════
log "=== Phase 5: TCN+focal + 30f window ==="

# v01: 기본 구성 (Phase 2와 동일 채널, 윈도우만 30f로)
run_exp P5-v01 "${BASE[@]}" "${CH_STD[@]}" --feature-set kp12

# v02: 더 큰 채널 — 더 많은 파라미터로 낙상 동작 패턴 학습
run_exp P5-v02 "${BASE[@]}" "${CH_LRG[@]}" --feature-set kp12

# v03: dilation 확장 [1,2,4,8,16,32] — 더 넓은 시간 컨텍스트 포착
run_exp P5-v03 "${BASE[@]}" "${CH_STD[@]}" --feature-set kp12 \
    --tcn-dilations 1,2,4,8,16,32

# v04: 표준 채널 + kp7 (docs 권장 7개 KP)
run_exp P5-v04 "${BASE[@]}" "${CH_STD[@]}" --feature-set kp7

# v05: 표준 채널 + minimal (5개 KP, 최경량)
run_exp P5-v05 "${BASE[@]}" "${CH_STD[@]}" --feature-set minimal

# v06: 대형 채널 + kp7 — capacity × 권장 KP
run_exp P5-v06 "${BASE[@]}" "${CH_LRG[@]}" --feature-set kp7

# ════════════════════════════════════════════════════════════════════════════════
log "All Phase 5 experiments complete."
echo ""
echo "=== PHASE 5 RESULTS ===" | tee -a "$SUMMARY"
printf "%-8s %-24s %-8s %7s %7s %8s %8s %6s\n" \
    "ID" "Channels" "KP" "testF1" "Rec" "FallP" "NFallP" "MinP" | tee -a "$SUMMARY"
echo "----------------------------------------------------------------------" | tee -a "$SUMMARY"
echo "P2-v06   [64,64,128,128]   kp12(13)  0.9253  0.9332  0.9044  0.9048  0.9044  [60f baseline]" | tee -a "$SUMMARY"

declare -A CH_MAP=(
    [P5-v01]="[64,64,128,128]"  [P5-v02]="[128,128,256,256]"
    [P5-v03]="[64,64,128,128]"  [P5-v04]="[64,64,128,128]"
    [P5-v05]="[64,64,128,128]"  [P5-v06]="[128,128,256,256]"
)
declare -A KP_MAP=(
    [P5-v01]="kp12(13)" [P5-v02]="kp12(13)"
    [P5-v03]="kp12(13),d+" [P5-v04]="kp7(7)"
    [P5-v05]="min(5)"   [P5-v06]="kp7(7)"
)

for id in P5-v01 P5-v02 P5-v03 P5-v04 P5-v05 P5-v06; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
m = json.load(open('$mfile'))
tv = m['metrics'].get('test_video', {})
ts = m.get('threshold_selection', {})
mark = ' ★' if tv.get('min_precision',0) >= 0.93 else ''
print('%-8s %-24s %-10s %7.4f %7.4f %8.4f %8.4f %6.4f%s' % (
    '$id', '${CH_MAP[$id]}', '${KP_MAP[$id]}',
    tv.get('f1',0), tv.get('recall',0),
    tv.get('precision',0), tv.get('nfall_precision',0),
    tv.get('min_precision',0), '$mark',
))
" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "Phase 5 summary written to $SUMMARY"
