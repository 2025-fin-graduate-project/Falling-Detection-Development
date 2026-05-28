#!/usr/bin/env bash
# Phase 42: Targeted cross-experiments based on Phase 41 analysis
#
# Phase 41 key findings:
#   - kp13-w60-filt = 0.9167 (best filtered), raw-kp7-w40 = 0.9167 (raw surprisingly strong)
#   - kp13-w40-filt baseline was MISSING in P41 (critical gap — filled here)
#   - Raw data competitive/better at kp7; also strong at kp13-w60
#   - Velocity: marginally helps kp13 (+0.004), hurts kp7
#   - TCN clearly underperforms GRU; hidden-size effect unexplored
#
# Phase 42 targeted cross (1-2 variable changes per experiment):
#   A. Gap fill: kp13-w40-filt (baseline)
#   B. Raw × other vars: raw-kp7-w60, raw-kp17-w40, raw-kp13-w40-vel
#   C. Best filtered × other vars: vel-kp13-w60, kp17-w60
#   D. Model size: kp13-w40-h128, kp13-w60-h128

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':'):/usr/local/cuda/lib64"

OUT="$REPO/results/phase42_cross"
FILT="$REPO/dataset/splits_v2_class_balanced_filtered"
RAW="$REPO/dataset/splits_v2_class_balanced"
mkdir -p "$OUT/logs"

COMMON_GRU64="--model-type gru --hidden-sizes 64 32
  --epochs 80 --batch-size 512 --lr 1e-3 --dropout 0.3
  --focal-alpha 0.65 --focal-gamma 2.0
  --fall-stride 1 --nfall-stride 5 --seed 42
  --pure-window --pure-margin 5
  --out-root $OUT"

COMMON_GRU128="--model-type gru --hidden-sizes 128 64
  --epochs 80 --batch-size 512 --lr 1e-3 --dropout 0.3
  --focal-alpha 0.65 --focal-gamma 2.0
  --fall-stride 1 --nfall-stride 5 --seed 42
  --pure-window --pure-margin 5
  --out-root $OUT"

run_exp() {
    local id="$1"; shift
    if [ -f "$OUT/$id/metrics.json" ]; then
        echo "  SKIP $id"
        return
    fi
    echo "=== $id ==="
    uv run python -u scripts/train_window_phase37.py \
        --exp-id "$id" "$@" \
        2>&1 | tee "$OUT/logs/${id}.log"
    echo "=== DONE $id  $(date +%H:%M) ==="
}

echo "=== Phase 42: Targeted Cross-Experiments ==="
echo "  Started: $(date)"

# ════════════════════════════════════════════════════════════════════════════
# Group A: Critical gap fill
# ════════════════════════════════════════════════════════════════════════════
echo "=== Group A: Gap fill — kp13-w40-filt (missing P41 baseline) ==="
run_exp P42-kp13-w40      $COMMON_GRU64  --feature-set kp13 --window-size 40 --data-dir "$FILT"

# ════════════════════════════════════════════════════════════════════════════
# Group B: Raw × other variables
# ════════════════════════════════════════════════════════════════════════════
echo "=== Group B: Raw × other vars ==="
run_exp P42-raw-kp7-w60   $COMMON_GRU64  --feature-set kp7  --window-size 60 --data-dir "$RAW"
run_exp P42-raw-kp17-w40  $COMMON_GRU64  --feature-set kp17 --window-size 40 --data-dir "$RAW"
run_exp P42-raw-kp13-vel  $COMMON_GRU64  --feature-set kp13 --window-size 40 --use-velocity --data-dir "$RAW"

# ════════════════════════════════════════════════════════════════════════════
# Group C: Best filtered config × other variables
# ════════════════════════════════════════════════════════════════════════════
echo "=== Group C: Best filtered × other vars ==="
run_exp P42-vel-kp13-w60  $COMMON_GRU64  --feature-set kp13 --window-size 60 --use-velocity --data-dir "$FILT"
run_exp P42-kp17-w60      $COMMON_GRU64  --feature-set kp17 --window-size 60 --data-dir "$FILT"

# ════════════════════════════════════════════════════════════════════════════
# Group D: Model size effect (hidden 128,64 vs baseline 64,32)
# ════════════════════════════════════════════════════════════════════════════
echo "=== Group D: Hidden size 128,64 ==="
run_exp P42-kp13-w40-h128 $COMMON_GRU128 --feature-set kp13 --window-size 40 --data-dir "$FILT"
run_exp P42-kp13-w60-h128 $COMMON_GRU128 --feature-set kp13 --window-size 60 --data-dir "$FILT"

echo "=== Phase 42 ALL DONE  $(date) ==="
