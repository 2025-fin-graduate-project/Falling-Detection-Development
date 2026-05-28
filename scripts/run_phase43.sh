#!/usr/bin/env bash
# Phase 43: 보고서 준비 — STedgeAI analyze + INT8 host eval
#
# 대상 모델 (P41+P42 ev_minpr 상위 + h128 대표):
#   1. P42-kp17-w60      ev=0.9211  h64  ← 최고
#   2. P41-kp13-w60      ev=0.9167  h64
#   3. P41-kp17-w40      ev=0.9123  h64
#   4. P41-vel-kp13-w40  ev=0.9123  h64  (velocity 효과 대표)
#   5. P41-raw-kp7-w40   ev=0.9167  h64  (raw 최고)
#   6. P42-kp13-w40-h128 ev=0.8991  h128 (h128 w40)
#   7. P42-kp13-w60-h128 ev=0.9035  h128 (h128 w60, val=0.9604)
#
# STedgeAI analyze: STedgeAI 내부 Python으로 실행
# INT8 host eval: uv run python으로 실행

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':'):/usr/local/cuda/lib64"

STEDGEAI_PY="/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python"
OUT="$REPO/results/phase43_report"
mkdir -p "$OUT/logs"

TARGETS=(
    "results/phase42_cross/P42-kp17-w60"
    "results/phase41_ablation/P41-kp13-w60"
    "results/phase41_ablation/P41-kp17-w40"
    "results/phase41_ablation/P41-vel-kp13-w40"
    "results/phase41_ablation/P41-raw-kp7-w40"
    "results/phase42_cross/P42-kp13-w40-h128"
    "results/phase42_cross/P42-kp13-w60-h128"
)

echo "=== Phase 43: 보고서 준비 ==="
echo "  Started: $(date)"

# ════════════════════════════════════════════════════════════════════════════
# Step 1: STedgeAI analyze (Flash / MACC / activation)
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Step 1: STedgeAI analyze ==="

for EXP in "${TARGETS[@]}"; do
    ID=$(basename "$EXP")
    LOG="$OUT/logs/${ID}_analyze.log"

    # 이미 analyze 완료 여부 확인
    if python3 -c "
import json, pathlib
m = json.loads(pathlib.Path('$EXP/metrics.json').read_text())
ok = m.get('stedgeai', {}).get('analyze', {}).get('analyze_ok', False)
exit(0 if ok else 1)
" 2>/dev/null; then
        echo "  SKIP analyze $ID (already done)"
        continue
    fi

    echo "  --- analyze $ID ---"
    "$STEDGEAI_PY" scripts/util/export_stedgeai.py \
        --exp-dir "$EXP" --target stm32n6 \
        2>&1 | tee "$LOG"
    echo "  DONE analyze $ID  $(date +%H:%M)"
done

# ════════════════════════════════════════════════════════════════════════════
# Step 2: INT8 host eval (STedgeAI --mode host)
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Step 2: INT8 host eval ==="

for EXP in "${TARGETS[@]}"; do
    ID=$(basename "$EXP")
    LOG="$OUT/logs/${ID}_int8eval.log"

    # 이미 host eval 완료 여부 확인
    if python3 -c "
import json, pathlib
m = json.loads(pathlib.Path('$EXP/metrics.json').read_text())
ok = 'stedgeai_host_eval' in m
exit(0 if ok else 1)
" 2>/dev/null; then
        echo "  SKIP host_eval $ID (already done)"
        continue
    fi

    echo "  --- host_eval $ID ---"
    uv run python scripts/util/eval_stedgeai_host.py \
        --exp-dir "$EXP" --eval-stride 5 \
        2>&1 | tee "$LOG"
    echo "  DONE host_eval $ID  $(date +%H:%M)"
done

# ════════════════════════════════════════════════════════════════════════════
# Step 3: 시각화 + 통합 테이블
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Step 3: 시각화 및 통합 테이블 생성 ==="
uv run python scripts/plot_phase43_results.py --out-dir "$OUT" \
    2>&1 | tee "$OUT/logs/plot.log"

echo ""
echo "=== Phase 43 ALL DONE  $(date) ==="
echo "  결과: $OUT/"
