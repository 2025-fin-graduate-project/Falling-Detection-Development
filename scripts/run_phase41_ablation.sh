#!/usr/bin/env bash
# Phase 41: Comprehensive ablation — keypoint / window / preprocessing / architecture
# Baseline (already done): P37-pure-h64-kp13-w40-gru → ev_test=0.9452
#
# Usage:
#   tmux new -s p41 "bash scripts/run_phase41_ablation.sh 2>&1 | tee results/phase41_ablation/run.log"
#   bash scripts/run_phase41_ablation.sh --wave 2   # run only wave 2

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/results/phase41_ablation"
FILT="$REPO/dataset/splits_v2_class_balanced_filtered"
RAW="$REPO/dataset/splits_v2_class_balanced"
mkdir -p "$OUT/logs"

WAVE="${1:-all}"
[[ "$WAVE" == "--wave" ]] && WAVE="$2"

# ── 공통 하이퍼파라미터 ──────────────────────────────────────────────────────
COMMON_GRU="--model-type gru --hidden-sizes 64 32
  --epochs 80 --batch-size 512 --lr 1e-3 --dropout 0.3
  --focal-alpha 0.65 --focal-gamma 2.0
  --fall-stride 1 --nfall-stride 5 --seed 42
  --pure-window --pure-margin 5
  --out-root $OUT"

COMMON_TCN="--model-type tcn --hidden-sizes 64 32
  --epochs 80 --batch-size 512 --lr 1e-3 --dropout 0.3
  --focal-alpha 0.65 --focal-gamma 2.0
  --fall-stride 1 --nfall-stride 5 --seed 42
  --pure-window --pure-margin 5
  --out-root $OUT"

# ── 실행 헬퍼 ────────────────────────────────────────────────────────────────
run_exp() {
    local id="$1"; shift
    if [ -f "$OUT/$id/metrics.json" ]; then
        echo "  SKIP $id"
        return
    fi
    echo "  START $id  $(date +%H:%M)"
    uv run python -u scripts/train_window_phase37.py \
        --exp-id "$id" "$@" \
        > "$OUT/logs/${id}.log" 2>&1 &
}

wait_jobs() {
    echo "  [waiting for ${1:-wave} jobs...]"
    wait
    echo "  [wave done $(date +%H:%M)]"
}

# ════════════════════════════════════════════════════════════════════════════
# Wave 1: 키포인트 ablation (w40, filtered, GRU 64,32)
# ════════════════════════════════════════════════════════════════════════════
if [[ "$WAVE" == "all" || "$WAVE" == "1" ]]; then
    echo "=== Wave 1: Keypoint ablation (kp5/kp7/kp9/kp11/kp17 vs kp13 baseline) ==="
    run_exp P41-kp5-w40      $COMMON_GRU --feature-set kp5  --window-size 40 --data-dir "$FILT"
    run_exp P41-kp7-w40      $COMMON_GRU --feature-set kp7  --window-size 40 --data-dir "$FILT"
    run_exp P41-kp9-w40      $COMMON_GRU --feature-set kp9  --window-size 40 --data-dir "$FILT"
    run_exp P41-kp11-w40     $COMMON_GRU --feature-set kp11 --window-size 40 --data-dir "$FILT"
    run_exp P41-kp17-w40     $COMMON_GRU --feature-set kp17 --window-size 40 --data-dir "$FILT"
    wait_jobs "wave 1"
fi

# ════════════════════════════════════════════════════════════════════════════
# Wave 2: 윈도우 ablation + 전처리 ablation (kp13, GRU 64,32)
# ════════════════════════════════════════════════════════════════════════════
if [[ "$WAVE" == "all" || "$WAVE" == "2" ]]; then
    echo "=== Wave 2: Window ablation (20f/30f/60f) + Filter ablation ==="
    run_exp P41-kp13-w20     $COMMON_GRU --feature-set kp13 --window-size 20 --data-dir "$FILT"
    run_exp P41-kp13-w30     $COMMON_GRU --feature-set kp13 --window-size 30 --data-dir "$FILT"
    run_exp P41-kp13-w60     $COMMON_GRU --feature-set kp13 --window-size 60 --data-dir "$FILT"
    run_exp P41-raw-kp13-w40 $COMMON_GRU --feature-set kp13 --window-size 40 --data-dir "$RAW"
    wait_jobs "wave 2"
fi

# ════════════════════════════════════════════════════════════════════════════
# Wave 3: Velocity + GRU h128 기준선 + TCN 비교
# ════════════════════════════════════════════════════════════════════════════
if [[ "$WAVE" == "all" || "$WAVE" == "3" ]]; then
    echo "=== Wave 3: Velocity / Architecture (GRU h128, TCN) ==="
    run_exp P41-vel-kp13-w40    $COMMON_GRU --feature-set kp13 --window-size 40 --use-velocity --data-dir "$FILT"
    run_exp P41-vel-kp7-w40     $COMMON_GRU --feature-set kp7  --window-size 40 --use-velocity --data-dir "$FILT"
    # TCN: kp13 w40 baseline
    run_exp P41-tcn-kp13-w40    $COMMON_TCN --feature-set kp13 --window-size 40 --data-dir "$FILT"
    # TCN: kp7 w40
    run_exp P41-tcn-kp7-w40     $COMMON_TCN --feature-set kp7  --window-size 40 --data-dir "$FILT"
    wait_jobs "wave 3"
fi

# ════════════════════════════════════════════════════════════════════════════
# Wave 4: 교차 조합 (best keypoint × best window, raw 확장)
# ════════════════════════════════════════════════════════════════════════════
if [[ "$WAVE" == "all" || "$WAVE" == "4" ]]; then
    echo "=== Wave 4: Cross combinations + raw variants ==="
    run_exp P41-kp7-w60         $COMMON_GRU --feature-set kp7  --window-size 60 --data-dir "$FILT"
    run_exp P41-kp7-w30         $COMMON_GRU --feature-set kp7  --window-size 30 --data-dir "$FILT"
    run_exp P41-raw-kp7-w40     $COMMON_GRU --feature-set kp7  --window-size 40 --data-dir "$RAW"
    run_exp P41-raw-kp13-w60    $COMMON_GRU --feature-set kp13 --window-size 60 --data-dir "$RAW"
    wait_jobs "wave 4"
fi

echo ""
echo "=== Phase 41 완료 ==="
echo "결과: $OUT"
echo ""
# 간단 요약
python3 -c "
import json, pathlib
rows = []
for m in sorted(pathlib.Path('$OUT').rglob('metrics.json')):
    try:
        d = json.loads(m.read_text())
        ue = d.get('unified_eval', {})
        ev = ue.get('event', {}).get('min_pr', None)
        cfg = d.get('config', {})
        rows.append((m.parent.name, cfg.get('feature_set','?'), cfg.get('window_size','?'), ev))
    except: pass
rows.sort(key=lambda x: -(x[3] or 0))
print(f'  {\"ID\":<35} {\"feat\":^6} {\"ws\":>4}  {\"ev_test\":>8}')
print('  ' + '-'*58)
for r in rows:
    print(f'  {r[0]:<35} {str(r[1]):^6} {str(r[2]):>4}  {str(r[3]):>8}')
" 2>/dev/null || true
