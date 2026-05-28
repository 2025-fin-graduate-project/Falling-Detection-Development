#!/usr/bin/env bash
# Conv+GRU event-level MinPR v2: standard splits_v2_filtered + epochs=100
# v1(epochs=30)이 0.88에 그쳐서 CLAUDE.md 기준 epochs=100/patience=15로 재시도

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

OUT="$REPO/results/cnn_gru_event_minpr2"
LOG="$OUT/run.log"
mkdir -p "$OUT"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

FILT="$REPO/dataset/splits_v2_filtered"
OUT_ARG="$OUT"

run_exp() {
    local id="$1"; shift
    if [[ -f "$OUT_ARG/$id/metrics.json" ]]; then log "SKIP $id"; return 0; fi
    log "TRAIN → $id"
    uv run python -u "$REPO/scripts/train_baseline.py" \
        --experiment-id "$id" \
        --model-type gru --gru-units 128,64 \
        --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
        --focal-loss --focal-gamma 2.0 --focal-alpha 0.25 \
        --data-scope all --epochs 100 --early-stop-patience 15 \
        --dropout-rate 0.3 --noise-std 0.02 \
        --train-positive-stride 1 --train-negative-stride 2 \
        --checkpoint-monitor val_video_min_pr \
        --threshold-eval-level event \
        --preprocessing filtered \
        --window-start-sec 3.0 --window-end-sec 9.0 \
        --batch-size 512 --eval-stride 2 --no-class-weight \
        --output-root "$OUT_ARG" --quiet \
        "$@" 2>&1 | tee "$OUT_ARG/${id}.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { log "FAIL $id"; return 1; }
    log "DONE $id"
}

log "=== event MinPR v2: standard filtered, epochs=100 ==="

run_exp "E-kp7-w40-s42"  --feature-set kp7  --target-steps 40 \
    --train-csv "$FILT/train.csv" --val-csv "$FILT/val.csv" --test-csv "$FILT/test.csv" --seed 42

run_exp "E-kp7-w40-s0"   --feature-set kp7  --target-steps 40 \
    --train-csv "$FILT/train.csv" --val-csv "$FILT/val.csv" --test-csv "$FILT/test.csv" --seed 0

run_exp "E-kp12-w40-s42" --feature-set kp12 --target-steps 40 \
    --train-csv "$FILT/train.csv" --val-csv "$FILT/val.csv" --test-csv "$FILT/test.csv" --seed 42

run_exp "E-kp7-w30-s42"  --feature-set kp7  --target-steps 30 \
    --train-csv "$FILT/train.csv" --val-csv "$FILT/val.csv" --test-csv "$FILT/test.csv" --seed 42

run_exp "E-kp12-w30-s42" --feature-set kp12 --target-steps 30 \
    --train-csv "$FILT/train.csv" --val-csv "$FILT/val.csv" --test-csv "$FILT/test.csv" --seed 42

log "=== 전체 완료 ==="
log ""
log "=== 결과 요약 (event MinPR) ==="
python3 - "$OUT_ARG" << 'PYEOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = [(m.parent.name,
         json.loads(m.read_text()).get("metrics",{}).get("test_video",{}).get("min_pr",0),
         json.loads(m.read_text()).get("metrics",{}).get("test_video",{}).get("f1",0))
        for m in out.glob("*/metrics.json")]
rows.sort(key=lambda x: -x[1])
print(f"{'ID':<22} {'MinPR':>7} {'F1':>6}")
print("-" * 38)
for r in rows:
    marker = " ★" if r[1] >= 0.91 else (" △" if r[1] >= 0.88 else "")
    print(f"{r[0]:<22} {r[1]:>7.4f} {r[2]:>6.4f}{marker}")
PYEOF
