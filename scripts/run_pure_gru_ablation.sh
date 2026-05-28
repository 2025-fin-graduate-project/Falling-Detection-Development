#!/usr/bin/env bash
# Pure GRU ablation: filter × kp × window
# Filter A (raw), EMA (splits_v2_ema), D (filtered)
# KP: kp7, kp12, all
# Window: 20, 30, 40, 50
# Model: GRU(128,64), conv-pre-layers=0

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

OUT="$REPO/results/pure_gru_ablation"
LOG="$OUT/run.log"
mkdir -p "$OUT"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# GPU 환경
_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 공통 하이퍼파라미터
COMMON="
  --model-type gru --gru-units 128,64
  --conv-pre-layers 0
  --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
  --data-scope all
  --early-stop-patience 15 --epochs 100
  --dropout-rate 0.3 --noise-std 0.02
  --train-negative-stride 2
  --min-val-precision 0.90
  --output-root $OUT
"

DIR_A="$REPO/dataset/splits_v2"
DIR_EMA="$REPO/dataset/splits_v2_ema"
DIR_D="$REPO/dataset/splits_v2_filtered"

run_exp() {
    local id="$1"; shift
    if [[ -f "$OUT/$id/metrics.json" ]]; then
        log "SKIP $id"
        return 0
    fi
    log "TRAIN → $id"
    uv run python -u "$REPO/scripts/train_baseline.py" \
        --experiment-id "$id" $COMMON "$@" \
        2>&1 | tee "$OUT/${id}.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { log "FAIL $id"; return 1; }
    log "DONE $id"
}

log "=== Pure GRU Ablation 시작 ==="

# ── Filter A (raw) ─────────────────────────────────────────────
log "-- Filter A --"
run_exp "A-kp7-w30"  --feature-set kp7  --target-steps 30 --preprocessing raw \
    --train-csv "$DIR_A/train.csv" --val-csv "$DIR_A/val.csv" --test-csv "$DIR_A/test.csv"

run_exp "A-kp7-w40"  --feature-set kp7  --target-steps 40 --preprocessing raw \
    --train-csv "$DIR_A/train.csv" --val-csv "$DIR_A/val.csv" --test-csv "$DIR_A/test.csv"

run_exp "A-kp12-w30" --feature-set kp12 --target-steps 30 --preprocessing raw \
    --train-csv "$DIR_A/train.csv" --val-csv "$DIR_A/val.csv" --test-csv "$DIR_A/test.csv"

run_exp "A-kp12-w40" --feature-set kp12 --target-steps 40 --preprocessing raw \
    --train-csv "$DIR_A/train.csv" --val-csv "$DIR_A/val.csv" --test-csv "$DIR_A/test.csv"

run_exp "A-all-w40"  --feature-set all  --target-steps 40 --preprocessing raw \
    --train-csv "$DIR_A/train.csv" --val-csv "$DIR_A/val.csv" --test-csv "$DIR_A/test.csv"

# ── Filter EMA (One-Euro + EMA, no VHSSC EMA) ─────────────────
log "-- Filter EMA --"
run_exp "EMA-kp7-w30"  --feature-set kp7  --target-steps 30 --preprocessing filtered \
    --train-csv "$DIR_EMA/train.csv" --val-csv "$DIR_EMA/val.csv" --test-csv "$DIR_EMA/test.csv"

run_exp "EMA-kp7-w40"  --feature-set kp7  --target-steps 40 --preprocessing filtered \
    --train-csv "$DIR_EMA/train.csv" --val-csv "$DIR_EMA/val.csv" --test-csv "$DIR_EMA/test.csv"

run_exp "EMA-kp7-w50"  --feature-set kp7  --target-steps 50 --preprocessing filtered \
    --train-csv "$DIR_EMA/train.csv" --val-csv "$DIR_EMA/val.csv" --test-csv "$DIR_EMA/test.csv"

run_exp "EMA-kp12-w20" --feature-set kp12 --target-steps 20 --preprocessing filtered \
    --train-csv "$DIR_EMA/train.csv" --val-csv "$DIR_EMA/val.csv" --test-csv "$DIR_EMA/test.csv"

run_exp "EMA-kp12-w30" --feature-set kp12 --target-steps 30 --preprocessing filtered \
    --train-csv "$DIR_EMA/train.csv" --val-csv "$DIR_EMA/val.csv" --test-csv "$DIR_EMA/test.csv"

run_exp "EMA-kp12-w40" --feature-set kp12 --target-steps 40 --preprocessing filtered \
    --train-csv "$DIR_EMA/train.csv" --val-csv "$DIR_EMA/val.csv" --test-csv "$DIR_EMA/test.csv"

run_exp "EMA-kp12-w50" --feature-set kp12 --target-steps 50 --preprocessing filtered \
    --train-csv "$DIR_EMA/train.csv" --val-csv "$DIR_EMA/val.csv" --test-csv "$DIR_EMA/test.csv"

run_exp "EMA-all-w30"  --feature-set all  --target-steps 30 --preprocessing filtered \
    --train-csv "$DIR_EMA/train.csv" --val-csv "$DIR_EMA/val.csv" --test-csv "$DIR_EMA/test.csv"

run_exp "EMA-all-w40"  --feature-set all  --target-steps 40 --preprocessing filtered \
    --train-csv "$DIR_EMA/train.csv" --val-csv "$DIR_EMA/val.csv" --test-csv "$DIR_EMA/test.csv"

run_exp "EMA-all-w50"  --feature-set all  --target-steps 50 --preprocessing filtered \
    --train-csv "$DIR_EMA/train.csv" --val-csv "$DIR_EMA/val.csv" --test-csv "$DIR_EMA/test.csv"

# ── Filter D (full filtered) ───────────────────────────────────
log "-- Filter D --"
run_exp "D-kp7-w40"   --feature-set kp7  --target-steps 40 --preprocessing filtered \
    --train-csv "$DIR_D/train.csv" --val-csv "$DIR_D/val.csv" --test-csv "$DIR_D/test.csv"

run_exp "D-kp12-w30"  --feature-set kp12 --target-steps 30 --preprocessing filtered \
    --train-csv "$DIR_D/train.csv" --val-csv "$DIR_D/val.csv" --test-csv "$DIR_D/test.csv"

run_exp "D-kp12-w40"  --feature-set kp12 --target-steps 40 --preprocessing filtered \
    --train-csv "$DIR_D/train.csv" --val-csv "$DIR_D/val.csv" --test-csv "$DIR_D/test.csv"

run_exp "D-kp12-w50"  --feature-set kp12 --target-steps 50 --preprocessing filtered \
    --train-csv "$DIR_D/train.csv" --val-csv "$DIR_D/val.csv" --test-csv "$DIR_D/test.csv"

run_exp "D-all-w40"   --feature-set all  --target-steps 40 --preprocessing filtered \
    --train-csv "$DIR_D/train.csv" --val-csv "$DIR_D/val.csv" --test-csv "$DIR_D/test.csv"

run_exp "D-all-w50"   --feature-set all  --target-steps 50 --preprocessing filtered \
    --train-csv "$DIR_D/train.csv" --val-csv "$DIR_D/val.csv" --test-csv "$DIR_D/test.csv"

log "=== 전체 완료 ==="

# 결과 요약
log ""
log "=== 결과 요약 ==="
python3 - "$OUT" << 'PYEOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for m in sorted(out.glob("*/metrics.json")):
    d = json.loads(m.read_text())
    tv = d.get("metrics", {}).get("test_video", {})
    rows.append((m.parent.name,
                 tv.get("min_precision", 0),
                 tv.get("f1", 0),
                 tv.get("recall", 0)))

rows.sort(key=lambda x: -x[1])
print(f"{'ID':<22} {'MinP':>6} {'F1':>6} {'Recall':>7}")
print("-" * 45)
for r in rows:
    print(f"{r[0]:<22} {r[1]:>6.4f} {r[2]:>6.4f} {r[3]:>7.4f}")
PYEOF
