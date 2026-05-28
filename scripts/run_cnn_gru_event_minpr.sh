#!/usr/bin/env bash
# Conv+GRU event-level MinPR > 0.91 실험
#
# 근거: P27-vm0 (event MinPR=0.9280) 재현 및 변형
#   - Dataset: splits_v2_class_balanced_filtered (class-balanced)
#   - checkpoint_monitor=val_video_min_pr + threshold_eval_level=event
#   - Conv2+GRU(128,64), kp7, w=40, filtered — 이미 0.9280 달성
#
# 변형 목표: kp12/w30/w50으로 더 높은 event MinPR 탐색

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

OUT="$REPO/results/cnn_gru_event_minpr"
LOG="$OUT/run.log"
mkdir -p "$OUT"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# GPU 환경
_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 데이터셋
CB_FILT="$REPO/dataset/splits_v2_class_balanced_filtered"
FILT="$REPO/dataset/splits_v2_filtered"

# 공통 하이퍼파라미터 (P27-vm0 기반 — 기본값 변경분 명시적으로 고정)
COMMON="
  --model-type gru
  --gru-units 128,64
  --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
  --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
  --data-scope all
  --epochs 30 --early-stop-patience 5
  --dropout-rate 0.3 --noise-std 0.02
  --train-positive-stride 1 --train-negative-stride 2
  --checkpoint-monitor val_video_min_pr
  --threshold-eval-level event
  --preprocessing filtered
  --window-start-sec 3.0 --window-end-sec 9.0
  --batch-size 512
  --eval-stride 2
  --no-class-weight
  --output-root $OUT
  --quiet
"

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

log "=== Conv+GRU event-level MinPR 실험 시작 ==="
log "목표: event-level MinPR > 0.91  (기준: P27-vm0=0.9280)"

# ── Group A: P27-vm0 재현 + seed 변형 (class_balanced_filtered, kp7, w40) ──
log "-- A: P27-vm0 재현 + seed --"
for seed in 42 0 1 2; do
    run_exp "A-kp7-w40-s${seed}" \
        --feature-set kp7 --target-steps 40 \
        --train-csv "$CB_FILT/train.csv" --val-csv "$CB_FILT/val.csv" --test-csv "$CB_FILT/test.csv" \
        --seed "$seed"
done

# ── Group B: kp12, class_balanced_filtered ──
log "-- B: kp12 (더 많은 feature) --"
run_exp "B-kp12-w40-s42" \
    --feature-set kp12 --target-steps 40 \
    --train-csv "$CB_FILT/train.csv" --val-csv "$CB_FILT/val.csv" --test-csv "$CB_FILT/test.csv" \
    --seed 42

run_exp "B-kp12-w30-s42" \
    --feature-set kp12 --target-steps 30 \
    --train-csv "$CB_FILT/train.csv" --val-csv "$CB_FILT/val.csv" --test-csv "$CB_FILT/test.csv" \
    --seed 42

run_exp "B-kp12-w50-s42" \
    --feature-set kp12 --target-steps 50 \
    --train-csv "$CB_FILT/train.csv" --val-csv "$CB_FILT/val.csv" --test-csv "$CB_FILT/test.csv" \
    --seed 42

run_exp "B-kp12-w40-s0" \
    --feature-set kp12 --target-steps 40 \
    --train-csv "$CB_FILT/train.csv" --val-csv "$CB_FILT/val.csv" --test-csv "$CB_FILT/test.csv" \
    --seed 0

# ── Group C: kp7, window 변형 ──
log "-- C: kp7 window 변형 --"
run_exp "C-kp7-w30-s42" \
    --feature-set kp7 --target-steps 30 \
    --train-csv "$CB_FILT/train.csv" --val-csv "$CB_FILT/val.csv" --test-csv "$CB_FILT/test.csv" \
    --seed 42

run_exp "C-kp7-w50-s42" \
    --feature-set kp7 --target-steps 50 \
    --train-csv "$CB_FILT/train.csv" --val-csv "$CB_FILT/val.csv" --test-csv "$CB_FILT/test.csv" \
    --seed 42

# ── Group D: standard splits_v2_filtered (비교용) ──
log "-- D: standard splits_v2_filtered 비교 --"
run_exp "D-kp7-w40-s42" \
    --feature-set kp7 --target-steps 40 \
    --train-csv "$FILT/train.csv" --val-csv "$FILT/val.csv" --test-csv "$FILT/test.csv" \
    --seed 42

run_exp "D-kp12-w40-s42" \
    --feature-set kp12 --target-steps 40 \
    --train-csv "$FILT/train.csv" --val-csv "$FILT/val.csv" --test-csv "$FILT/test.csv" \
    --seed 42

log "=== 전체 완료 ==="

# 결과 요약
log ""
log "=== 결과 요약 (event-level MinPR) ==="
python3 - "$OUT" << 'PYEOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for m in sorted(out.glob("*/metrics.json")):
    d = json.loads(m.read_text())
    tv = d.get("metrics", {}).get("test_video", {})
    ts = d.get("threshold_selection", {})
    rows.append((
        m.parent.name,
        tv.get("min_pr", 0),
        tv.get("f1", 0),
        tv.get("precision", 0),
        tv.get("nfall_precision", 0),
        tv.get("recall", 0),
        tv.get("nfall_recall", 0),
        ts.get("threshold", 0),
    ))

rows.sort(key=lambda x: -x[1])
print(f"{'ID':<22} {'MinPR':>7} {'F1':>6} {'FallP':>7} {'NFallP':>7} {'FallR':>7} {'NFallR':>7} {'Thr':>6}")
print("-" * 78)
for r in rows:
    marker = " ★" if r[1] >= 0.91 else ""
    print(f"{r[0]:<22} {r[1]:>7.4f} {r[2]:>6.4f} {r[3]:>7.4f} {r[4]:>7.4f} {r[5]:>7.4f} {r[6]:>7.4f} {r[7]:>6.3f}{marker}")
PYEOF
