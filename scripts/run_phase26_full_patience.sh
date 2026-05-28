#!/usr/bin/env bash
# Phase 26: Full patience (epochs=100, patience=15) — 핵심 실험
# 가설: patience=5/epoch=30은 GRU(256,128) 수렴에 부족. 충분한 훈련 시간이 0.93+ 달성의 열쇠.
#
# 실험 축:
#   v01: GRU(128,64)  kp7  — P21-v01 복제, patience 증가
#   v02: GRU(256,128) kp7  — 메인 후보 (용량 + 시간)
#   v03: GRU(256,128) kp12 — 하체 키포인트 추가

set -uo pipefail
cd "$(dirname "$0")/.."

OUTROOT="results/phase26_full_patience"
SUMMARY="$OUTROOT/summary.log"
GPU_WAIT_MAX_USED_MB=2200
GPU_WAIT_INTERVAL_SEC=60

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTROOT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

wait_for_gpu() {
    while true; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
        if [[ -n "$used" && "$used" -le "$GPU_WAIT_MAX_USED_MB" ]]; then
            log "GPU ready: ${used}MiB"
            return 0
        fi
        log "GPU busy: ${used:-?}MiB — waiting ${GPU_WAIT_INTERVAL_SEC}s"
        sleep "$GPU_WAIT_INTERVAL_SEC"
    done
}

COMMON=(
    --model-type gru
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
    --preprocessing filtered
    --label-column label --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2 --eval-stride 2
    --epochs 100 --early-stop-patience 15 --batch-size 512
    --min-consecutive-values 1,2,3,4,5,7,9 --threshold-count 37
    --checkpoint-monitor val_event_min_pr
    --threshold-eval-level event --event-tolerance-windows 2
    --no-export-tflite --quiet --seed 42
)

CB_TRAIN="dataset/splits_v2_class_balanced_filtered/train.csv"
CB_VAL="dataset/splits_v2_class_balanced_filtered/val.csv"
CB_TEST="dataset/splits_v2_class_balanced_filtered/test.csv"

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP $id — already complete"
        return 0
    fi
    wait_for_gpu
    log "START $id"
    uv run python scripts/train_baseline.py \
        --experiment-id "$id" --output-root "$OUTROOT" \
        "${COMMON[@]}" "$@" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    if [[ "$rc" -eq 0 ]]; then
        python3 - "$OUTROOT" "$id" <<'PYEOF' | tee -a "$SUMMARY"
import json, sys
from pathlib import Path
outroot = Path(sys.argv[1]); exp_id = sys.argv[2]
m = json.loads((outroot / exp_id / "metrics.json").read_text())
thr = m["threshold_selection"]
te = m["metrics"]["test_event_video"]
tv = m["metrics"]["test_video"]
print(
    f"[OK] {exp_id} "
    f"event={te['min_pr']:.4f} video={tv['min_pr']:.4f} "
    f"thr={thr['threshold']:.3f} mc={thr['min_consecutive']}"
)
cm = te.get("confusion_matrix")
if cm:
    print(f"  event_CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
PYEOF
    else
        log "FAIL $id (exit $rc)"
    fi
    return "$rc"
}

log "=== Phase 26: Full patience (epochs=100, patience=15) ==="

# v01: GRU(128,64) kp7 — P21-v01 완전 재현, patience만 늘림
run_exp "P26-v01" \
    --gru-units 128,64 --feature-set kp7 \
    --train-csv "$CB_TRAIN" --val-csv "$CB_VAL" --test-csv "$CB_TEST"

# v02: GRU(256,128) kp7 — 메인 후보
run_exp "P26-v02" \
    --gru-units 256,128 --feature-set kp7 \
    --train-csv "$CB_TRAIN" --val-csv "$CB_VAL" --test-csv "$CB_TEST"

# v03: GRU(256,128) kp12 — 하체 포함 45 features
run_exp "P26-v03" \
    --gru-units 256,128 --feature-set kp12 \
    --train-csv "$CB_TRAIN" --val-csv "$CB_VAL" --test-csv "$CB_TEST"

log "=== Phase 26 done ==="

python3 << 'PYEOF'
import json
from pathlib import Path

base = Path("results/phase26_full_patience")
rows = []
for d in sorted(base.iterdir()):
    mj = d / "metrics.json"
    if not mj.exists(): continue
    m = json.loads(mj.read_text())
    te = m.get("metrics", {}).get("test_event_video", {})
    tv = m.get("metrics", {}).get("test_video", {})
    thr = m.get("threshold_selection", {})
    ev = te.get("min_pr", 0); vv = tv.get("min_pr", 0)
    flag = " *** TARGET" if ev >= 0.93 else ""
    rows.append((ev, d.name, vv, thr.get("threshold","?"), thr.get("min_consecutive","?")))

rows.sort(reverse=True)
print("\n=== Phase 26 summary ===")
for ev, name, vv, thr, mc in rows:
    flag = " *** TARGET" if ev >= 0.93 else ""
    print(f"  {name}: event={ev:.4f} video={vv:.4f} thr={thr} mc={mc}{flag}")
PYEOF
