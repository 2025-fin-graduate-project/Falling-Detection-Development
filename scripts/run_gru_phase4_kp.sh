#!/usr/bin/env bash
# Phase 4 — Keypoint Ablation × Window Size
#
# Reference: docs/analysis/dataset_analysis_report.md § 3. 키포인트 선정
#
# 3-Tier keypoint sets (from analysis):
#   minimal (5개): kp{0,5,6,11,12} — 코·양어깨·양골반  (최소, 가장 경량)
#   kp7     (7개): kp{0,5,6,7,8,11,12} — minimal + 팔꿈치  (권장 균형점)
#   kp12   (13개): kp{0..12}            — 현재 기준선 (baseline)
#
# 제거 이유:
#   kp8  (손목 kp9,10 포함)   → 신뢰도 최하 ~0.33, 낙상 중 노이즈
#   all  (무릎·발목 kp13-16)  → VHSSC 상관 < 0.05, 낙상 예측 기여 없음
#
# kp12 baselines already exist:
#   60f: P2-v06  F1=0.9253  MinP=0.9044
#   30f: P3-v07  (see Phase 3 results)
#
# New experiments (4개):
#   P4-v01: 60f, minimal
#   P4-v02: 60f, kp7
#   P4-v03: 30f, minimal
#   P4-v04: 30f, kp7

set -uo pipefail

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OUTROOT="results/gru_phase4_kp"
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

# ── Shared: TCN+focal, filtered, LB-2 ────────────────────────────────────────
BASE=(
    --model-type tcn
    --tcn-channels 64,64,128,128 --tcn-dilations 1,2,4,8 --tcn-kernel-size 3
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --early-stop-patience 15 --epochs 100
    --min-val-precision 0.90
)

WIN60=(--target-steps 60 --window-start-sec 5.0 --window-end-sec 9.0)
WIN30=(--target-steps 30 --window-start-sec 3.0 --window-end-sec 9.0)

# ════════════════════════════════════════════════════════════════════════════════
log "=== GROUP A: 60f window (baseline P2-v06 kp12: F1=0.9253 MinP=0.9044) ==="

run_exp P4-v01 "${BASE[@]}" "${WIN60[@]}" --feature-set minimal
run_exp P4-v02 "${BASE[@]}" "${WIN60[@]}" --feature-set kp7

# ════════════════════════════════════════════════════════════════════════════════
log "=== GROUP B: 30f window (baseline P3-v07 kp12: see Phase3 results) ==="

run_exp P4-v03 "${BASE[@]}" "${WIN30[@]}" --feature-set minimal
run_exp P4-v04 "${BASE[@]}" "${WIN30[@]}" --feature-set kp7

# ════════════════════════════════════════════════════════════════════════════════
log "All Phase 4 experiments complete."
echo ""
echo "=== PHASE 4 RESULTS ===" | tee -a "$SUMMARY"
printf "%-8s %5s %-8s %3s %7s %7s %8s %8s %6s\n" \
    "ID" "Win" "KP" "nKP" "testF1" "Rec" "FallP" "NFallP" "MinP" | tee -a "$SUMMARY"
echo "-------------------------------------------------------------------" | tee -a "$SUMMARY"

# Baselines
echo "P2-v06   60f  kp12      13  0.9253  0.9332  0.9044  0.9048  0.9044  [60f baseline]" | tee -a "$SUMMARY"
echo "P3-v07   30f  kp12      13  (see Phase 3 results)                    [30f baseline]" | tee -a "$SUMMARY"

declare -A WIN_MAP=([P4-v01]=60 [P4-v02]=60 [P4-v03]=30 [P4-v04]=30)
declare -A KP_MAP=([P4-v01]=minimal [P4-v02]=kp7 [P4-v03]=minimal [P4-v04]=kp7)
declare -A NKP_MAP=([P4-v01]=5 [P4-v02]=7 [P4-v03]=5 [P4-v04]=7)

for id in P4-v01 P4-v02 P4-v03 P4-v04; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
m = json.load(open('$mfile'))
tv = m['metrics'].get('test_video', {})
ts = m.get('threshold_selection', {})
print('%-8s %5d %-8s %3d %7.4f %7.4f %8.4f %8.4f %6.4f' % (
    '$id', ${WIN_MAP[$id]}, '${KP_MAP[$id]}', ${NKP_MAP[$id]},
    tv.get('f1',0), tv.get('recall',0),
    tv.get('precision',0), tv.get('nfall_precision',0),
    tv.get('min_precision',0),
))
" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "Phase 4 summary written to $SUMMARY"
