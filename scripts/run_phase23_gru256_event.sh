#!/usr/bin/env bash
# Phase 23: GRU(256,128) + val_event_min_pr checkpoint + class_balanced_filtered
# Hypothesis: P19-v04 (GRU256, val_loss) reached event MinPR 0.9153;
#             P21-v01 (GRU128, val_event_min_pr) reached 0.9227 (+0.0106 vs P20).
#             Combining GRU(256,128) capacity with event-level checkpoint on
#             class-balanced data should push event MinPR past 0.93.

set -uo pipefail

OUTROOT="${OUTROOT:-results/phase23_gru256_event}"
SUMMARY="$OUTROOT/summary.log"

BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_STRIDE="${EVAL_STRIDE:-2}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
TARGET_MIN_PR="${TARGET_MIN_PR:-0.93}"
GPU_WAIT_MAX_USED_MB="${GPU_WAIT_MAX_USED_MB:-2200}"
GPU_WAIT_INTERVAL_SEC="${GPU_WAIT_INTERVAL_SEC:-60}"

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTROOT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

wait_for_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "nvidia-smi not found; continuing"
        return 0
    fi
    while true; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
        if [[ -n "$used" && "$used" -le "$GPU_WAIT_MAX_USED_MB" ]]; then
            log "GPU ready: used=${used}MiB <= ${GPU_WAIT_MAX_USED_MB}MiB"
            return 0
        fi
        log "GPU busy: used=${used:-unknown}MiB; waiting ${GPU_WAIT_INTERVAL_SEC}s"
        sleep "$GPU_WAIT_INTERVAL_SEC"
    done
}

COMMON=(
    --model-type gru
    --gru-units 256,128
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
    --preprocessing filtered
    --train-csv dataset/splits_v2_class_balanced_filtered/train.csv
    --val-csv   dataset/splits_v2_class_balanced_filtered/val.csv
    --test-csv  dataset/splits_v2_class_balanced_filtered/test.csv
    --label-column label
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --eval-stride "$EVAL_STRIDE"
    --epochs "$EPOCHS" --early-stop-patience "$PATIENCE" --batch-size "$BATCH_SIZE"
    --min-consecutive-values 1,2,3,4,5,7,9
    --threshold-count 37
    --checkpoint-monitor val_event_min_pr
    --threshold-eval-level event
    --event-tolerance-windows 2
    --no-export-tflite
    --quiet
)

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id - already complete"
        return 0
    fi
    wait_for_gpu
    log "START $id"
    uv run python scripts/train_baseline.py \
        --experiment-id "$id" --output-root "$OUTROOT" \
        "${COMMON[@]}" "$@" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    if [[ "$rc" -eq 0 ]]; then
        log_metrics "$id" "OK"
    else
        log "FAIL  $id  (exit $rc)"
    fi
    return "$rc"
}

log_metrics() {
    local id="$1"
    local status="$2"
    python3 - "$OUTROOT" "$id" "$status" <<'PYEOF' | tee -a "$SUMMARY"
import json, sys
from pathlib import Path

outroot = Path(sys.argv[1])
exp_id = sys.argv[2]
status = sys.argv[3]
m = json.loads((outroot / exp_id / "metrics.json").read_text())
thr = m["threshold_selection"]
tv = m["metrics"]["test_video"]
te = m["metrics"]["test_event_video"]
print(
    f"[{status}] {exp_id} "
    f"test_event_min_pr={te['min_pr']:.4f} "
    f"test_video_min_pr={tv['min_pr']:.4f} "
    f"thr={thr['threshold']:.3f} mc={thr['min_consecutive']} "
    f"thr_eval={thr.get('eval_level','video')}"
)
cm = te.get("confusion_matrix")
if cm:
    print(f"  event_CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
PYEOF
}

print_summary() {
    log "=== Phase 23 summary ==="
    python3 - "$OUTROOT" "$TARGET_MIN_PR" <<'PYEOF'
import json, sys
from pathlib import Path

outroot = Path(sys.argv[1])
target = float(sys.argv[2])
rows = []
for d in sorted(outroot.iterdir()):
    mj = d / "metrics.json"
    if not mj.exists(): continue
    m = json.loads(mj.read_text())
    te = m.get("metrics", {}).get("test_event_video", {})
    tv = m.get("metrics", {}).get("test_video", {})
    thr = m.get("threshold_selection", {})
    flag = "PASS" if te.get("min_pr", 0.0) >= target else ""
    rows.append((te.get("min_pr", 0), d.name, tv.get("min_pr", 0), thr.get("threshold","?"), thr.get("min_consecutive","?"), flag))

rows.sort(reverse=True)
print("id,event_min_pr,video_min_pr,threshold,min_consecutive,flag")
for ev, name, vminp, thr, mc, flag in rows:
    print(f"{name},{ev:.4f},{vminp:.4f},{thr},{mc},{flag}")
PYEOF
}

log "=== Phase 23: GRU(256,128) + val_event_min_pr checkpoint + class_balanced ==="

# v01: direct scaling of P21-v01 — GRU(256,128) + event checkpoint, neg_stride=2
run_exp "P23-v01" --seed 42

# v02: kp12 (45 features) — more keypoints with GRU(256,128) + event checkpoint
# (neg_stride=1 dropped: P21-v02 showed it hurts with event checkpoint, patience=5)
run_exp "P23-v02" --feature-set kp12 \
    --train-csv dataset/splits_v2_class_balanced_filtered/train.csv \
    --val-csv   dataset/splits_v2_class_balanced_filtered/val.csv \
    --test-csv  dataset/splits_v2_class_balanced_filtered/test.csv \
    --seed 42

# v03: seed variation for robustness check
run_exp "P23-v03" --seed 0

print_summary | tee -a "$SUMMARY"
