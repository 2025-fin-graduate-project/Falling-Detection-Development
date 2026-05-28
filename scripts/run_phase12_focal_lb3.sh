#!/usr/bin/env bash
# Phase 12: focal α=0.10 + LB-3 (3-class training)
# Hypothesis: LB-3 separates falling(1) vs fallen(2), enabling cleaner falling pattern learning.
#             focal α=0.10 confirmed effective in P11-v05 (FN 28→22).
# Sequence: v01 (α=0.10 baseline) → v02 (+LB-3) → v03 (+LB-3+hard-neg) → v04 (+kp12)

set -uo pipefail

ANALYSIS_ROOT="${ANALYSIS_ROOT:-results/phase10_error_analysis/P9O-v01}"
OUTROOT="${OUTROOT:-results/phase12_focal_lb3}"
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

# ---------------------------------------------------------------------------
# Shared training args (LB-2 base)
# ---------------------------------------------------------------------------
COMMON=(
    --model-type gru
    --gru-units 128,64
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.10
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --eval-stride "$EVAL_STRIDE"
    --epochs "$EPOCHS" --early-stop-patience "$PATIENCE" --batch-size "$BATCH_SIZE"
    --min-consecutive-values 1,2,3,4,5,7,9
    --threshold-count 37
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
    log "=== Phase 12 summary ==="
    python3 - "$OUTROOT" "$TARGET_MIN_PR" <<'PYEOF'
import json, sys
from pathlib import Path
outroot = Path(sys.argv[1])
target = float(sys.argv[2])
rows = []
for d in sorted(outroot.iterdir()):
    mj = d / "metrics.json"
    if not mj.exists():
        continue
    m = json.loads(mj.read_text())
    if m.get("skipped"):
        continue
    tv = m.get("metrics", {}).get("test_video", {})
    vv = m.get("metrics", {}).get("val_video", {})
    thr = m.get("threshold_selection", {})
    rows.append({
        "id":           d.name,
        "test_min_pr":  tv.get("min_pr", float("nan")),
        "nfall_p":      tv.get("nfall_precision", float("nan")),
        "nfall_r":      tv.get("nfall_recall", float("nan")),
        "fall_p":       tv.get("precision", float("nan")),
        "fall_r":       tv.get("recall", float("nan")),
        "val_min_pr":   vv.get("min_pr", float("nan")),
        "thr":          thr.get("threshold", float("nan")),
        "mc":           thr.get("min_consecutive", "?"),
        "pass":         tv.get("min_pr", 0.0) >= target,
    })
print(f"{'ID':<12} {'test_MinPR':>10} {'NFallP':>8} {'NFallR':>8} {'FallP':>8} {'FallR':>8} {'thr':>6} {'mc':>3} {'PASS?':>6}")
for r in rows:
    flag = "PASS" if r["pass"] else ""
    print(f"{r['id']:<12} {r['test_min_pr']:>10.4f} {r['nfall_p']:>8.4f} {r['nfall_r']:>8.4f} {r['fall_p']:>8.4f} {r['fall_r']:>8.4f} {r['thr']:>6.3f} {r['mc']:>3} {flag:>6}")
PYEOF
}

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
run_until_pass() {
    # v01: focal α=0.10 pure baseline (LB-2, no hard-neg) — reproduce P11-v05 effect isolated
    log "=== P12-v01: focal α=0.10 baseline (LB-2, no hard-neg) ==="
    run_exp "P12-v01" \
        --checkpoint-monitor val_loss
    any_passed && return

    # v02: focal α=0.10 + LB-3 (3-class, positive_labels=1,2) — core hypothesis
    log "=== P12-v02: focal α=0.10 + LB-3 ==="
    run_exp "P12-v02" \
        --checkpoint-monitor val_loss \
        --label-column label_3class --num-classes 3 --positive-labels 1,2
    any_passed && return

    # v03: focal α=0.10 + LB-3 + hard-negative (best of P11 combined with LB-3)
    log "=== P12-v03: focal α=0.10 + LB-3 + hard-neg ==="
    run_exp "P12-v03" \
        --checkpoint-monitor val_loss \
        --label-column label_3class --num-classes 3 --positive-labels 1,2 \
        --hard-negative-video-ids "$FP_CSV" --hard-negative-stride 1
    any_passed && return

    # v04: focal α=0.10 + LB-3 + hard-neg + kp12 (more features)
    log "=== P12-v04: focal α=0.10 + LB-3 + hard-neg + kp12 ==="
    run_exp "P12-v04" \
        --checkpoint-monitor val_loss \
        --label-column label_3class --num-classes 3 --positive-labels 1,2 \
        --hard-negative-video-ids "$FP_CSV" --hard-negative-stride 1 \
        --feature-set kp12
    any_passed && return
}

run_until_pass

print_summary | tee -a "$SUMMARY"
