#!/usr/bin/env bash
# Phase 5 (재설계) — Compact Quantizable Unidirectional GRU
#
# 배포 제약 (STM32N6 Cortex-M55 + STedgeAI):
#   - Unidirectional GRU (stateful 스트리밍 호환)
#   - STedgeAI INT8 양자화: 모델 가중치 크기 기준
#   - 목표: Float MinP ≥ 0.93 / INT8 MinP ≥ 0.90
#   - 가중치 크기 목표: < 512 KB INT8
#
# Phase 4 결과 기반으로 최적 KP 선택 후 실행
# (Phase 4 완료 전 이 스크립트는 실행하지 말 것)
#
# 모델 크기 예상 (INT8, 가중치 기준):
#   GRU(256,128) uni + kp7  → ~370 KB  (Phase 4 기준)
#   GRU(128,64)  uni + kp7  → ~110 KB  ← 목표 범위
#   GRU(128,64)  uni + min  → ~80  KB
#   GRU(64,32)   uni + kp7  → ~30  KB
#   GRU(64,32)   uni + min  → ~22  KB
#
# Experiments (6개):
#   P5-v01: GRU(128,64) uni, LB-2, kp7,  40f          ← primary target
#   P5-v02: GRU(128,64) uni, LB-2, kp7,  40f + focal
#   P5-v03: GRU(128,64) uni, LB-2, minimal, 40f
#   P5-v04: GRU(128,64) uni, LB-2, minimal, 40f+focal
#   P5-v05: GRU(64,32)  uni, LB-2, kp7,  40f + focal  ← ultra-compact
#   P5-v06: GRU(64,32)  uni, LB-2, minimal, 40f+focal ← smallest

set -uo pipefail

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OUTROOT="results/gru_phase5_compact"
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
log "=== Phase 5: Compact Quantizable GRU (STM32 배포 최적화) ==="

# GRU(128,64) — 경량
run_exp P5-v01 "${BASE[@]}" --gru-units 128,64 --feature-set kp7
run_exp P5-v02 "${BASE[@]}" --gru-units 128,64 --feature-set kp7     "${FOCAL[@]}"
run_exp P5-v03 "${BASE[@]}" --gru-units 128,64 --feature-set minimal
run_exp P5-v04 "${BASE[@]}" --gru-units 128,64 --feature-set minimal "${FOCAL[@]}"

# GRU(64,32) — 초경량
run_exp P5-v05 "${BASE[@]}" --gru-units 64,32  --feature-set kp7     "${FOCAL[@]}"
run_exp P5-v06 "${BASE[@]}" --gru-units 64,32  --feature-set minimal "${FOCAL[@]}"

# ════════════════════════════════════════════════════════════════════════════════
log "All Phase 5 experiments complete."
echo ""
echo "=== PHASE 5 RESULTS (Compact Unidirectional GRU) ===" | tee -a "$SUMMARY"
printf "%-8s %-10s %-8s %-6s %7s %8s %6s %10s\n" \
    "ID" "Units" "KP" "Focal" "testF1" "NFallP" "MinP" "INT8_KB" | tee -a "$SUMMARY"
echo "----------------------------------------------------------------------" | tee -a "$SUMMARY"

declare -A UNITS_MAP=([P5-v01]="(128,64)" [P5-v02]="(128,64)" [P5-v03]="(128,64)"
                      [P5-v04]="(128,64)" [P5-v05]="(64,32)"  [P5-v06]="(64,32)")
declare -A KP_MAP=([P5-v01]=kp7 [P5-v02]=kp7 [P5-v03]=minimal
                   [P5-v04]=minimal [P5-v05]=kp7 [P5-v06]=minimal)
declare -A FC_MAP=([P5-v01]=no [P5-v02]=yes [P5-v03]=no [P5-v04]=yes [P5-v05]=yes [P5-v06]=yes)

for id in P5-v01 P5-v02 P5-v03 P5-v04 P5-v05 P5-v06; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
m = json.load(open('$mfile'))
tv = m['metrics'].get('test_video', {})
ep = m.get('export_paths', {})
int8_kb = ep.get('model_int8_size_kb', 0)
mark = ' ★' if tv.get('min_precision',0) >= 0.93 else ''
size_tag = ' ✓' if 0 < int8_kb < 512 else (' ⚠' if int8_kb >= 512 else ' ?')
print('%-8s %-10s %-8s %-6s %7.4f %8.4f %6.4f %8.1f%s%s' % (
    '$id', '${UNITS_MAP[$id]}', '${KP_MAP[$id]}', '${FC_MAP[$id]}',
    tv.get('f1',0), tv.get('nfall_precision',0),
    tv.get('min_precision',0), int8_kb, mark, size_tag,
))" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "Phase 5 summary written to $SUMMARY"
