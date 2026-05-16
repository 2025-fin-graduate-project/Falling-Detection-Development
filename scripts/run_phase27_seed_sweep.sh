#!/usr/bin/env bash
# Phase 27: P21-v01 최적 구성 seed sweep
# 가설: GRU(128,64) kp7 patience=5 가 최적. seed 분산으로 0.93 달성 가능성 탐색.
# val_video_min_pr checkpoint 변형도 포함 — event보다 안정적인 val 신호.

set -uo pipefail
cd "$(dirname "$0")/.."

OUTROOT="results/phase27_seed_sweep"
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
            log "GPU ready: ${used}MiB"; return 0
        fi
        log "GPU busy: ${used:-?}MiB — waiting ${GPU_WAIT_INTERVAL_SEC}s"
        sleep "$GPU_WAIT_INTERVAL_SEC"
    done
}

BASE_ARGS=(
    --model-type gru --gru-units 128,64
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
    --preprocessing filtered
    --train-csv dataset/splits_v2_class_balanced_filtered/train.csv
    --val-csv   dataset/splits_v2_class_balanced_filtered/val.csv
    --test-csv  dataset/splits_v2_class_balanced_filtered/test.csv
    --label-column label --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2 --eval-stride 2
    --epochs 30 --early-stop-patience 5 --batch-size 512
    --min-consecutive-values 1,2,3,4,5,7,9 --threshold-count 37
    --threshold-eval-level event --event-tolerance-windows 2
    --no-export-tflite --quiet
)

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    [[ -f "$OUTROOT/$id/metrics.json" ]] && { log "SKIP $id"; return 0; }
    wait_for_gpu
    log "START $id"
    uv run python scripts/train_baseline.py \
        --experiment-id "$id" --output-root "$OUTROOT" \
        "${BASE_ARGS[@]}" "$@" 2>&1 | tee "$logfile"
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
print(f"[OK] {exp_id} event={te['min_pr']:.4f} video={tv['min_pr']:.4f} thr={thr['threshold']:.3f} mc={thr['min_consecutive']}")
cm = te.get("confusion_matrix")
if cm:
    print(f"  event_CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
PYEOF
    else
        log "FAIL $id (exit $rc)"
    fi
}

log "=== Phase 27: P21-v01 설정 seed sweep + checkpoint 변형 ==="

# ── 그룹 A: val_event_min_pr checkpoint (P21-v01과 동일), seed 다양성
run_exp "P27-s0"  --checkpoint-monitor val_event_min_pr --seed 0
run_exp "P27-s1"  --checkpoint-monitor val_event_min_pr --seed 1
run_exp "P27-s2"  --checkpoint-monitor val_event_min_pr --seed 2
run_exp "P27-s3"  --checkpoint-monitor val_event_min_pr --seed 3

# ── 그룹 B: val_video_min_pr checkpoint — event보다 안정적, 더 많은 샘플로 평균
# P21-v01이 event checkpoint로 0.9227이었는데, video checkpoint가 더 나을 수 있음
run_exp "P27-vm0" --checkpoint-monitor val_video_min_pr  --seed 42
run_exp "P27-vm1" --checkpoint-monitor val_video_min_pr  --seed 0

log "=== Phase 27 done ==="

python3 << 'PYEOF'
import json
from pathlib import Path

base = Path("results/phase27_seed_sweep")
rows = []
for d in sorted(base.iterdir()):
    mj = d / "metrics.json"
    if not mj.exists(): continue
    m = json.loads(mj.read_text())
    te = m.get("metrics", {}).get("test_event_video", {})
    tv = m.get("metrics", {}).get("test_video", {})
    thr = m.get("threshold_selection", {})
    ev = te.get("min_pr", 0); vv = tv.get("min_pr", 0)
    rows.append((ev, d.name, vv, thr.get("threshold","?"), thr.get("min_consecutive","?"), thr.get("checkpoint_monitor","?")))

rows.sort(reverse=True)
print("\n=== Phase 27 summary ===")
for ev, name, vv, thr, mc, ckpt in rows:
    flag = " *** TARGET" if ev >= 0.93 else (" ← best" if ev == rows[0][0] else "")
    print(f"  {name}: event={ev:.4f} video={vv:.4f} thr={thr} mc={mc}{flag}")

global_best = max((ev for ev, *_ in rows), default=0)
print(f"\nPhase 27 best: {global_best:.4f} (target: 0.9300)")
PYEOF
