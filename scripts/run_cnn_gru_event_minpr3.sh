#!/usr/bin/env bash
# Conv+GRU event MinPR v3: focal_alpha sweep
# 근거: v2 best(E-kp7-w40-s0=0.9079)의 병목은 NFallP (FN이 많음)
#   alpha=0.25로 fall 클래스 다운웨이트 → FN 증가 → NFallP 저하
#   alpha 올리면 fall 감지 강화 → FN 감소 → NFallP 개선 기대
# Base config: kp7 w40 s0 epochs=100 patience=15 standard filtered

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

OUT="$REPO/results/cnn_gru_event_minpr3"
LOG="$OUT/run.log"
mkdir -p "$OUT"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

FILT="$REPO/dataset/splits_v2_filtered"

run_exp() {
    local id="$1"; shift
    if [[ -f "$OUT/$id/metrics.json" ]]; then log "SKIP $id"; return 0; fi
    log "TRAIN → $id"
    uv run python -u "$REPO/scripts/train_baseline.py" \
        --experiment-id "$id" \
        --model-type gru --gru-units 128,64 \
        --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
        --focal-loss --focal-gamma 2.0 \
        --data-scope all --epochs 100 --early-stop-patience 15 \
        --dropout-rate 0.3 --noise-std 0.02 \
        --train-positive-stride 1 --train-negative-stride 2 \
        --checkpoint-monitor val_video_min_pr \
        --threshold-eval-level event \
        --preprocessing filtered \
        --window-start-sec 3.0 --window-end-sec 9.0 \
        --batch-size 512 --eval-stride 2 --no-class-weight \
        --feature-set kp7 --target-steps 40 --seed 0 \
        --train-csv "$FILT/train.csv" --val-csv "$FILT/val.csv" --test-csv "$FILT/test.csv" \
        --output-root "$OUT" --quiet \
        "$@" 2>&1 | tee "$OUT/${id}.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { log "FAIL $id"; return 1; }
    log "DONE $id"
}

log "=== event MinPR v3: focal_alpha sweep (base: kp7 w40 s0) ==="

# alpha sweep: 0.35 / 0.50 / 0.65
run_exp "F-a35-kp7-w40-s0"  --focal-alpha 0.35
run_exp "F-a50-kp7-w40-s0"  --focal-alpha 0.50
run_exp "F-a65-kp7-w40-s0"  --focal-alpha 0.65

# kp12도 best alpha로 추가 (sweep 결과 보고 결정)
run_exp "F-a50-kp12-w40-s0" --focal-alpha 0.50 --feature-set kp12

log "=== 전체 완료 ==="
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
    ts = d.get("threshold_selection", {})
    rc = d.get("run_config", {})
    rows.append((
        m.parent.name,
        tv.get("min_pr", 0),
        tv.get("f1", 0),
        tv.get("precision", 0),
        tv.get("nfall_precision", 0),
        tv.get("recall", 0),
        tv.get("nfall_recall", 0),
        ts.get("threshold", 0),
        ts.get("min_consecutive", 0),
        rc.get("focal_alpha", "?"),
    ))
rows.sort(key=lambda x: -x[1])
print(f"{'ID':<24} {'MinPR':>7} {'F1':>6} {'FallP':>7} {'NFallP':>7} {'FallR':>7} {'NFallR':>7} {'Thr':>5} {'MC':>2} {'alpha':>5}")
print("-" * 92)
for r in rows:
    marker = " ★" if r[1] >= 0.91 else (" △" if r[1] >= 0.90 else "")
    print(f"{r[0]:<24} {r[1]:>7.4f} {r[2]:>6.4f} {r[3]:>7.4f} {r[4]:>7.4f} {r[5]:>7.4f} {r[6]:>7.4f} {r[7]:>5.3f} {r[8]:>2} {r[9]:>5}{marker}")
PYEOF
