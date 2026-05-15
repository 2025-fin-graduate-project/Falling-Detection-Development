#!/usr/bin/env bash
# Phase 13: focal α 탐색 + train_negative_stride=1 조합
# 핵심: α=0.10 + neg_stride=1 (미시험), α=0.05 탐색
# 목표: FP≤19 AND FN≤19 동시 달성 → MinPR ≥ 0.92

set -uo pipefail

ANALYSIS_ROOT="${ANALYSIS_ROOT:-results/phase10_error_analysis/P9O-v01}"
OUTROOT="${OUTROOT:-results/phase13_alpha_negstride}"
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
# Shared base args
# ---------------------------------------------------------------------------
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
    log "=== Phase 13 summary ==="
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
    thr = m.get("threshold_selection", {})
    rows.append({
        "id":          d.name,
        "test_min_pr": tv.get("min_pr", float("nan")),
        "nfall_p":     tv.get("nfall_precision", float("nan")),
        "nfall_r":     tv.get("nfall_recall", float("nan")),
        "fall_p":      tv.get("precision", float("nan")),
        "fall_r":      tv.get("recall", float("nan")),
        "thr":         thr.get("threshold", float("nan")),
        "mc":          thr.get("min_consecutive", "?"),
        "pass":        tv.get("min_pr", 0.0) >= target,
        "cm":          tv.get("confusion_matrix"),
    })
print(f"{'ID':<12} {'test_MinPR':>10} {'NFallP':>8} {'NFallR':>8} {'FallP':>8} {'FallR':>8} {'thr':>6} {'mc':>3} {'PASS?':>6}")
for r in rows:
    flag = "PASS" if r["pass"] else ""
    print(f"{r['id']:<12} {r['test_min_pr']:>10.4f} {r['nfall_p']:>8.4f} {r['nfall_r']:>8.4f} {r['fall_p']:>8.4f} {r['fall_r']:>8.4f} {r['thr']:>6.3f} {r['mc']:>3} {flag:>6}")
    if r["cm"]:
        cm = r["cm"]
        print(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
PYEOF
}

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
run_until_pass() {
    # v01: α=0.10 + hard-neg + neg_stride=1 (핵심 미시험 조합)
    log "=== P13-v01: α=0.10 + hard-neg + neg_stride=1 ==="
    run_exp "P13-v01" \
        --focal-alpha 0.10 \
        --hard-negative-video-ids "$FP_CSV" --hard-negative-stride 1 \
        --train-negative-stride 1
    any_passed && return

    # v02: α=0.05 + hard-neg (더 강한 non-fall 손실)
    log "=== P13-v02: α=0.05 + hard-neg ==="
    run_exp "P13-v02" \
        --focal-alpha 0.05 \
        --hard-negative-video-ids "$FP_CSV" --hard-negative-stride 1 \
        --train-negative-stride 2
    any_passed && return

    # v03: α=0.05 + hard-neg + neg_stride=1
    log "=== P13-v03: α=0.05 + hard-neg + neg_stride=1 ==="
    run_exp "P13-v03" \
        --focal-alpha 0.05 \
        --hard-negative-video-ids "$FP_CSV" --hard-negative-stride 1 \
        --train-negative-stride 1
    any_passed && return

    # v04: α=0.10 + hard-neg + noise_std=0.05 (강한 augmentation)
    log "=== P13-v04: α=0.10 + hard-neg + noise_std=0.05 ==="
    run_exp "P13-v04" \
        --focal-alpha 0.10 \
        --hard-negative-video-ids "$FP_CSV" --hard-negative-stride 1 \
        --train-negative-stride 2 \
        --noise-std 0.05
    any_passed && return
}

run_until_pass

print_summary | tee -a "$SUMMARY"
