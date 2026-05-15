#!/usr/bin/env bash
# Phase 4 (재설계) — Unidirectional GRU × KP × Window ablation
#
# 배포 제약 (STM32N6 Cortex-M55 + STedgeAI):
#   - Unidirectional GRU 필수 (stateful 스트리밍 호환)
#   - STedgeAI (.keras 직접 임포트) → INT8 양자화
#   - 목표: Float MinP ≥ 0.93 / INT8 MinP ≥ 0.90
#
# Phase 3 결과 요약 (bidir, LB-2):
#   P3-v01: GRU(256,128) bidir, 30f  MinP=0.9315 ★
#   P3-v02: GRU(256,128) bidir, 40f  MinP=0.9556 ★  ← 40f가 우세
#
# Phase 4 목표:
#   - bidir → unidirectional 전환 시 성능 손실 측정
#   - KP 축소 (kp7, minimal) 효과 확인
#   - focal loss 조합 최적화
#
# Experiments (6개):
#   P4-v01: GRU(256,128) uni, LB-2, kp12, 40f          ← bidir 대비 uni 기준선
#   P4-v02: GRU(256,128) uni, LB-2, kp12, 40f + focal  ← focal 추가
#   P4-v03: GRU(256,128) uni, LB-2, kp7,  40f          ← 권장 KP
#   P4-v04: GRU(256,128) uni, LB-2, kp7,  40f + focal  ← 권장 KP + focal
#   P4-v05: GRU(256,128) uni, LB-2, minimal, 40f       ← 최소 KP
#   P4-v06: GRU(256,128) uni, LB-2, minimal, 40f+focal ← 최소 KP + focal

set -uo pipefail

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OUTROOT="results/gru_phase4_uni_kp"
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
    --gru-units 256,128
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    # NOTE: --bidirectional 없음 → unidirectional (stateful 배포 호환)
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --early-stop-patience 15 --epochs 100
    --min-val-precision 0.90
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
)

FOCAL=(--focal-loss --focal-gamma 2.0 --focal-alpha 0.25)

# ════════════════════════════════════════════════════════════════════════════════
log "=== Phase 4: Unidirectional GRU × KP ablation (STM32 배포 기준) ==="

# 기준선: kp12, no focal vs focal
run_exp P4-v01 "${BASE[@]}" --feature-set kp12
run_exp P4-v02 "${BASE[@]}" --feature-set kp12 "${FOCAL[@]}"

# 권장 KP (kp7)
run_exp P4-v03 "${BASE[@]}" --feature-set kp7
run_exp P4-v04 "${BASE[@]}" --feature-set kp7  "${FOCAL[@]}"

# 최소 KP (minimal)
run_exp P4-v05 "${BASE[@]}" --feature-set minimal
run_exp P4-v06 "${BASE[@]}" --feature-set minimal "${FOCAL[@]}"

# ════════════════════════════════════════════════════════════════════════════════
log "All Phase 4 experiments complete."
echo ""
echo "=== PHASE 4 RESULTS (Unidirectional GRU) ===" | tee -a "$SUMMARY"
printf "%-8s %-8s %-6s %7s %7s %8s %8s %6s\n" \
    "ID" "KP" "Focal" "testF1" "Rec" "FallP" "NFallP" "MinP" | tee -a "$SUMMARY"
echo "-------------------------------------------------------------------" | tee -a "$SUMMARY"
echo "P3-v02  kp12(13)  bidir  0.9704 0.9850  0.9563  0.9556  0.9556  [bidir ref]" | tee -a "$SUMMARY"

declare -A KP_MAP=([P4-v01]=kp12 [P4-v02]=kp12 [P4-v03]=kp7 [P4-v04]=kp7 [P4-v05]=minimal [P4-v06]=minimal)
declare -A FC_MAP=([P4-v01]=no   [P4-v02]=yes  [P4-v03]=no  [P4-v04]=yes [P4-v05]=no       [P4-v06]=yes)

for id in P4-v01 P4-v02 P4-v03 P4-v04 P4-v05 P4-v06; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
m = json.load(open('$mfile'))
tv = m['metrics'].get('test_video', {})
mark = ' ★' if tv.get('min_precision',0) >= 0.93 else ''
print('%-8s %-8s %-6s %7.4f %7.4f %8.4f %8.4f %6.4f%s' % (
    '$id', '${KP_MAP[$id]}', '${FC_MAP[$id]}',
    tv.get('f1',0), tv.get('recall',0),
    tv.get('precision',0), tv.get('nfall_precision',0),
    tv.get('min_precision',0), mark,
))" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "Phase 4 summary written to $SUMMARY"
