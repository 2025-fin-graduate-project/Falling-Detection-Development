#!/usr/bin/env bash
# Phase 17: pose-relative kp7 features.
# Hypothesis: persistent non-fall false positives come from over-reliance on
# absolute coordinates / HSSC motion, not from insufficient keypoint count.

set -uo pipefail

OUTROOT="${OUTROOT:-results/phase17_pose_rel}"
SUMMARY="$OUTROOT/summary.log"

BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_STRIDE="${EVAL_STRIDE:-2}"
TARGET_MIN_PR="${TARGET_MIN_PR:-0.92}"

REL_TRAIN="dataset/splits_v2_filtered_rel/train.csv"
REL_VAL="dataset/splits_v2_filtered_rel/val.csv"
REL_TEST="dataset/splits_v2_filtered_rel/test.csv"

if [[ ! -f "$REL_TRAIN" || ! -f "$REL_VAL" || ! -f "$REL_TEST" ]]; then
    echo "Pose-relative splits not found; building dataset/splits_v2_filtered_rel ..."
    uv run python scripts/util/build_filtered_v2_splits_rel.py || exit $?
fi

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTROOT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

COMMON=(
    --model-type gru
    --gru-units 128,64
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0
    --preprocessing filtered
    --train-csv "$REL_TRAIN"
    --val-csv   "$REL_VAL"
    --test-csv  "$REL_TEST"
    --label-column label
    --label-mode segment_max
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --eval-stride "$EVAL_STRIDE"
    --batch-size "$BATCH_SIZE"
    --min-consecutive-values 1,2,3,4,5,7,9
    --threshold-count 37
    --checkpoint-monitor val_loss
    --quiet
)

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id - already complete"
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
    if not mj.exists():
        continue
    m = json.loads(mj.read_text())
    if m.get("skipped"):
        continue
    tv = m.get("metrics", {}).get("test_video", {})
    if float(tv.get("min_pr", 0.0)) >= target:
        print(f"PASS: {d.name}  test_min_pr={tv['min_pr']:.4f}")
        sys.exit(0)
sys.exit(1)
PYEOF
}

print_summary() {
    log "=== Phase 17 pose-relative summary ==="
    python3 - "$OUTROOT" "$TARGET_MIN_PR" <<'PYEOF'
import json, sys
from pathlib import Path
outroot = Path(sys.argv[1])
target = float(sys.argv[2])
print(f"{'ID':<14} {'MinPR':>8} {'NFallP':>8} {'NFallR':>8} {'FallP':>8} {'FallR':>8} {'thr':>6} {'mc':>3} PASS")
for d in sorted(outroot.iterdir()):
    mj = d / "metrics.json"
    if not mj.exists():
        continue
    m = json.loads(mj.read_text())
    if m.get("skipped"):
        continue
    tv = m.get("metrics", {}).get("test_video", {})
    thr = m.get("threshold_selection", {})
    flag = "PASS" if tv.get("min_pr", 0.0) >= target else ""
    print(f"{d.name:<14} {tv.get('min_pr',float('nan')):>8.4f} {tv.get('nfall_precision',float('nan')):>8.4f} {tv.get('nfall_recall',float('nan')):>8.4f} {tv.get('precision',float('nan')):>8.4f} {tv.get('recall',float('nan')):>8.4f} {thr.get('threshold',float('nan')):>6.3f} {thr.get('min_consecutive','?'):>3} {flag}")
    cm = tv.get("confusion_matrix")
    if cm:
        print(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
PYEOF
}

log "=== Phase 17 pose-relative features: target=${TARGET_MIN_PR}, batch=${BATCH_SIZE}, eval_stride=${EVAL_STRIDE} ==="

# v01: relative coordinates and pose-collapse features only.
run_exp "P17-v01" \
    --feature-set kp7rel --focal-alpha 0.25 --epochs 30 --early-stop-patience 5 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v02: same relative representation, but keep existing HSSC/RWHC/VHSSC/AHSSC features.
run_exp "P17-v02" \
    --feature-set kp7releng --focal-alpha 0.25 --epochs 30 --early-stop-patience 5 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v03: lower fall-side focal alpha to defend non-fall after coordinate normalization.
run_exp "P17-v03" \
    --feature-set kp7rel --focal-alpha 0.15 --epochs 30 --early-stop-patience 5 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v04: seed repeat for stability.
run_exp "P17-v04" \
    --feature-set kp7rel --focal-alpha 0.25 --epochs 30 --early-stop-patience 5 --seed 0
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

print_summary | tee -a "$SUMMARY"
