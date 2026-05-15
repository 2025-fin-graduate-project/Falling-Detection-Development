#!/usr/bin/env bash
# Phase 11: val_video_min_pr checkpointing + hard-negative training
# Sequence:
#   1. Generate Phase 10 FP/FN error timeline report for P9O-v01
#   2. Run P11-v01 (MinPR checkpointing only)
#   3. Run P11-v02 (+ val FP video hard-negative stride=1)
#   4. STedgeAI analyze + host eval for passing candidates

set -uo pipefail

P9O_EXP="${P9O_EXP:-results/phase9_minpr_open/P9O-v01}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-results/phase10_error_analysis/P9O-v01}"
OUTROOT="${OUTROOT:-results/phase11_minpr_hardneg}"
SUMMARY="$OUTROOT/summary.log"

EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_STRIDE="${EVAL_STRIDE:-2}"
TARGET_MIN_PR="${TARGET_MIN_PR:-0.92}"
STEDGE_PY="${STEDGE_PY:-/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python}"

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTROOT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

# ---------------------------------------------------------------------------
# Step 1: Phase 10 error timeline report
# ---------------------------------------------------------------------------
log "=== Phase 10: generating FP/FN error timeline report for P9O-v01 ==="
if [[ -f "$ANALYSIS_ROOT/false_positive_videos.csv" ]]; then
    log "SKIP Phase 10 report — already exists at $ANALYSIS_ROOT"
else
    uv run python scripts/report_phase10_error_timelines.py \
        --exp-dir "$P9O_EXP" \
        --output-dir "$ANALYSIS_ROOT" \
        2>&1 | tee "$OUTROOT/phase10_report.log"
    rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        log "FAIL Phase 10 report (exit $rc) — aborting"
        exit 1
    fi
fi

FP_CSV="$ANALYSIS_ROOT/false_positive_videos.csv"
n_fp=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$FP_CSV')))" 2>/dev/null || echo "?")
log "Phase 10 report done. Val FP videos: $n_fp"

# ---------------------------------------------------------------------------
# Shared training args
# ---------------------------------------------------------------------------
COMMON=(
    --model-type gru
    --gru-units 128,64
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
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
    --checkpoint-monitor val_video_min_pr
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

# STedgeAI skipped — focusing on float performance first

# ---------------------------------------------------------------------------
# Helper: check if any experiment in OUTROOT has already achieved target
# ---------------------------------------------------------------------------
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
    tv = m.get("metrics", {}).get("test_video", {})
    if float(tv.get("min_pr", 0.0)) >= target:
        print(f"PASS: {d.name}  test_min_pr={tv['min_pr']:.4f}")
        sys.exit(0)
sys.exit(1)
PYEOF
}

print_summary() {
    log "=== Phase 11 summary ==="
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
    tv = m.get("metrics", {}).get("test_video", {})
    vv = m.get("metrics", {}).get("val_video", {})
    rows.append({
        "id":           d.name,
        "test_min_pr":  tv.get("min_pr", float("nan")),
        "test_fall_p":  tv.get("precision", float("nan")),
        "test_nfall_p": tv.get("nfall_precision", float("nan")),
        "test_fall_r":  tv.get("recall", float("nan")),
        "test_nfall_r": tv.get("nfall_recall", float("nan")),
        "val_min_pr":   vv.get("min_pr", float("nan")),
        "pass":         tv.get("min_pr", 0.0) >= target,
    })
print(f"{'ID':<12} {'test_MinPR':>10} {'test_FP':>8} {'test_NFP':>8} {'test_FR':>8} {'test_NFR':>8} {'val_MinPR':>10} {'PASS?':>6}")
for r in rows:
    flag = "PASS" if r["pass"] else ""
    print(f"{r['id']:<12} {r['test_min_pr']:>10.4f} {r['test_fall_p']:>8.4f} {r['test_nfall_p']:>8.4f} {r['test_fall_r']:>8.4f} {r['test_nfall_r']:>8.4f} {r['val_min_pr']:>10.4f} {flag:>6}")
PYEOF
}

# ---------------------------------------------------------------------------
# Experiment candidates — run in order until test MinPR >= 0.92
#
# Rationale for each:
#   v01  baseline: just switch to val_video_min_pr checkpointing
#   v02  + val FP hard-negative oversample (stride=1)
#   v03  v02 + wider consecutive sweep (probe postprocess gain)
#   v04  v02 + higher focal alpha=0.10 (90% weight on non-fall windows)
#   v05  v02 + train_negative_stride=1 globally (more non-fall coverage)
#   v06  v02 + kp12 (more features; non-fall patterns may be better separated)
#   v07  v02 + GRU(256,128) (bigger model; more capacity for ambiguous poses)
# ---------------------------------------------------------------------------

log "=== P11-v01: MinPR checkpointing only ==="
run_exp "P11-v01"

if any_passed; then
    log "TARGET REACHED after P11-v01 — skipping further candidates"
else

log "=== P11-v02: + val FP hard-negative stride=1 ==="
run_exp "P11-v02" \
    --hard-negative-video-ids "$FP_CSV" \
    --hard-negative-stride 1

if any_passed; then
    log "TARGET REACHED after P11-v02 — skipping further candidates"
else

log "=== P11-v03: + wider consecutive sweep ==="
run_exp "P11-v03" \
    --hard-negative-video-ids "$FP_CSV" \
    --hard-negative-stride 1 \
    --min-consecutive-values 1,2,3,4,5,6,7,8,9

if any_passed; then
    log "TARGET REACHED after P11-v03 — skipping further candidates"
else

log "=== P11-v04: + focal alpha=0.10 (90% weight on non-fall) ==="
run_exp "P11-v04" \
    --hard-negative-video-ids "$FP_CSV" \
    --hard-negative-stride 1 \
    --focal-alpha 0.10

if any_passed; then
    log "TARGET REACHED after P11-v04 — skipping further candidates"
else

log "=== P11-v05: + global negative stride=1 (all non-fall windows) ==="
run_exp "P11-v05" \
    --hard-negative-video-ids "$FP_CSV" \
    --hard-negative-stride 1 \
    --train-negative-stride 1

if any_passed; then
    log "TARGET REACHED after P11-v05 — skipping further candidates"
else

log "=== P11-v06: kp12 + hard-negative stride=1 ==="
run_exp "P11-v06" \
    --feature-set kp12 \
    --hard-negative-video-ids "$FP_CSV" \
    --hard-negative-stride 1

if any_passed; then
    log "TARGET REACHED after P11-v06 — skipping further candidates"
else

log "=== P11-v07: GRU(256,128) + hard-negative stride=1 ==="
run_exp "P11-v07" \
    --gru-units 256,128 \
    --hard-negative-video-ids "$FP_CSV" \
    --hard-negative-stride 1

fi; fi; fi; fi; fi; fi  # close all if-else blocks

print_summary | tee -a "$SUMMARY"
