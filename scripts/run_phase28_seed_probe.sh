#!/usr/bin/env bash
# Phase 28: seed probe — val_video_min_pr 확장 탐색
# 가설: val_video_min_pr + seed=42가 0.9241(신기록). 다른 seed가 0.93+ 달성 가능한지 탐색.
# 병목: NFallPrecision 0.924 (FN=18 → 16으로 2건 감소 필요)
#
# 그룹 A: val_video_min_pr (high-upside checkpoint)
#   seeds: 5, 7, 13, 77, 100, 123
# 그룹 B: val_event_min_pr (안정적 baseline 추가 탐색)
#   seeds: 5, 7, 100

set -uo pipefail
cd "$(dirname "$0")/.."

OUTROOT="results/phase28_seed_probe"
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
ev = te['min_pr']; flag = " *** TARGET!" if ev >= 0.93 else (" ← best?" if ev >= 0.92 else "")
print(f"[OK] {exp_id} event={ev:.4f} video={tv['min_pr']:.4f} thr={thr['threshold']:.3f} mc={thr['min_consecutive']}{flag}")
cm = te.get("confusion_matrix")
if cm:
    print(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
PYEOF
    else
        log "FAIL $id (exit $rc)"
    fi
}

log "=== Phase 28: seed probe (val_video_min_pr 확장 + val_event_min_pr 추가) ==="
log "목표: 0.9241(P27-vm0) 돌파 → ≥0.93 달성. 병목=NFallPrecision(FN 18→16 필요)"

# ── 그룹 A: val_video_min_pr — high-upside, seed 의존성 강함
run_exp "P28-vm5"   --checkpoint-monitor val_video_min_pr  --seed 5
run_exp "P28-vm7"   --checkpoint-monitor val_video_min_pr  --seed 7
run_exp "P28-vm13"  --checkpoint-monitor val_video_min_pr  --seed 13
run_exp "P28-vm77"  --checkpoint-monitor val_video_min_pr  --seed 77
run_exp "P28-vm100" --checkpoint-monitor val_video_min_pr  --seed 100
run_exp "P28-vm123" --checkpoint-monitor val_video_min_pr  --seed 123

# ── 그룹 B: val_event_min_pr — 안정적, 추가 seed
run_exp "P28-ev5"   --checkpoint-monitor val_event_min_pr  --seed 5
run_exp "P28-ev7"   --checkpoint-monitor val_event_min_pr  --seed 7
run_exp "P28-ev100" --checkpoint-monitor val_event_min_pr  --seed 100

log "=== Phase 28 done ==="

python3 << 'PYEOF'
import json
from pathlib import Path

base = Path("results/phase28_seed_probe")
rows = []
for d in sorted(base.iterdir()):
    mj = d / "metrics.json"
    if not mj.exists(): continue
    m = json.loads(mj.read_text())
    te = m.get("metrics", {}).get("test_event_video", {})
    tv = m.get("metrics", {}).get("test_video", {})
    thr = m.get("threshold_selection", {})
    ev = te.get("min_pr", 0); vv = tv.get("min_pr", 0)
    cm = te.get("confusion_matrix")
    fn = cm[1][0] if cm else "?"
    rows.append((ev, d.name, vv, thr.get("threshold","?"), thr.get("min_consecutive","?"), fn))

rows.sort(reverse=True)
print("\n=== Phase 28 summary ===")
print(f"  {'ID':<14} {'EventMinP':>10} {'VideoMinP':>10} {'thr':>6} {'mc':>4} {'FN':>5}  flag")
for ev, name, vv, thr, mc, fn in rows:
    flag = " *** TARGET!" if ev >= 0.93 else (" ← best" if ev == rows[0][0] else "")
    print(f"  {name:<14} {ev:>10.4f} {vv:>10.4f} {thr:>6} {mc:>4} {fn:>5} {flag}")

best = rows[0][0] if rows else 0
prev_best = 0.9241
print(f"\nPhase 28 best: {best:.4f}  (prev best P27-vm0: {prev_best:.4f}, target: 0.9300)")
hits = [r for r in rows if r[0] >= 0.93]
print(f"≥0.93 달성: {len(hits)} 건" + (" — 목표 달성!" if hits else " — 추가 전략 필요"))
PYEOF
