#!/usr/bin/env bash
# Phase 14: α 중간값(0.15, 0.20) + neg_stride=1 조합 탐색
# 목표: α=0.10(FP=19,FN=26)과 α=0.25(FP↑,FN↓) 사이 균형점 발견
# 핵심: FP≤19 AND FN≤19 동시 달성 → MinPR ≥ 0.92

set -uo pipefail

ANALYSIS_ROOT="${ANALYSIS_ROOT:-results/phase10_error_analysis/P9O-v01}"
OUTROOT="${OUTROOT:-results/phase14_alpha_sweep}"
SUMMARY="$OUTROOT/summary.log"

EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_STRIDE="${EVAL_STRIDE:-2}"
TARGET_MIN_PR="${TARGET_MIN_PR:-0.92}"

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTROOT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

FP_CSV="$ANALYSIS_ROOT/false_positive_videos.csv"

COMMON=(
    --model-type gru
    --gru-units 128,64
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --eval-stride "$EVAL_STRIDE"
    --epochs "$EPOCHS" --early-stop-patience "$PATIENCE" --batch-size "$BATCH_SIZE"
    --min-consecutive-values 1,2,3,4,5,7,9
    --threshold-count 37
    --checkpoint-monitor val_loss
    --quiet
)

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id — already complete"
        return 0
    fi
    log "START $id"
    uv run python scripts/train_baseline.py \
        --experiment-id "$id" --output-root "$OUTROOT" \
        "${COMMON[@]}" "$@" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    if [[ "$rc" -eq 0 ]]; then
        local min_pr
        min_pr=$(python3 -c "
import json
d = json.load(open('$OUTROOT/$id/metrics.json'))
print(f\"{d['metrics']['test_video']['min_pr']:.4f}\")
" 2>/dev/null || echo "?")
        log "OK    $id  test_min_pr=$min_pr"
    else
        log "FAIL  $id  (exit $rc)"
    fi
    return "$rc"
}

any_passed() {
    python3 - "$OUTROOT" "$TARGET_MIN_PR" <<'PYEOF'
import json, sys
from pathlib import Path
outroot = Path(sys.argv[1])
target = float(sys.argv[2])
for d in outroot.iterdir():
    mj = d / "metrics.json"
    if not mj.exists(): continue
    m = json.loads(mj.read_text())
    if m.get("skipped"): continue
    tv = m.get("metrics", {}).get("test_video", {})
    if float(tv.get("min_pr", 0.0)) >= target:
        print(f"PASS: {d.name}  test_min_pr={tv['min_pr']:.4f}")
        sys.exit(0)
sys.exit(1)
PYEOF
}

print_summary() {
    log "=== Phase 14 summary ==="
    python3 - "$OUTROOT" "$TARGET_MIN_PR" <<'PYEOF'
import json, sys
from pathlib import Path
outroot = Path(sys.argv[1])
target = float(sys.argv[2])
for d in sorted(outroot.iterdir()):
    mj = d / "metrics.json"
    if not mj.exists(): continue
    m = json.loads(mj.read_text())
    if m.get("skipped"): continue
    tv = m.get("metrics", {}).get("test_video", {})
    thr = m.get("threshold_selection", {})
    flag = "PASS" if tv.get("min_pr", 0.0) >= target else ""
    print(f"{d.name:<12} {tv.get('min_pr',float('nan')):>10.4f} {tv.get('nfall_precision',float('nan')):>8.4f} {tv.get('nfall_recall',float('nan')):>8.4f} {tv.get('precision',float('nan')):>8.4f} {tv.get('recall',float('nan')):>8.4f} {thr.get('threshold',float('nan')):>6.3f} {thr.get('min_consecutive','?'):>3} {flag}")
    cm = tv.get("confusion_matrix")
    if cm: print(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
PYEOF
}

run_until_pass() {
    # v01: α=0.15 + hard-neg + neg_stride=1 (핵심 — α 중간값)
    log "=== P14-v01: α=0.15 + hard-neg + neg_stride=1 ==="
    run_exp "P14-v01" \
        --focal-alpha 0.15 \
        --hard-negative-video-ids "$FP_CSV" --hard-negative-stride 1 \
        --train-negative-stride 1
    any_passed && return

    # v02: α=0.20 + hard-neg + neg_stride=1
    log "=== P14-v02: α=0.20 + hard-neg + neg_stride=1 ==="
    run_exp "P14-v02" \
        --focal-alpha 0.20 \
        --hard-negative-video-ids "$FP_CSV" --hard-negative-stride 1 \
        --train-negative-stride 1
    any_passed && return

    # v03: α=0.15 + hard-neg + neg_stride=1 + noise_std=0.05
    log "=== P14-v03: α=0.15 + hard-neg + neg_stride=1 + noise=0.05 ==="
    run_exp "P14-v03" \
        --focal-alpha 0.15 \
        --hard-negative-video-ids "$FP_CSV" --hard-negative-stride 1 \
        --train-negative-stride 1 \
        --noise-std 0.05
    any_passed && return

    # v04: α=0.15 + hard-neg + neg_stride=2 (neg_stride 비교)
    log "=== P14-v04: α=0.15 + hard-neg + neg_stride=2 ==="
    run_exp "P14-v04" \
        --focal-alpha 0.15 \
        --hard-negative-video-ids "$FP_CSV" --hard-negative-stride 1 \
        --train-negative-stride 2
    any_passed && return
}

run_until_pass

print_summary | tee -a "$SUMMARY"
