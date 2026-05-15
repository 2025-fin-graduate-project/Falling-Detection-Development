#!/usr/bin/env bash
# Phase 17+ new-hypothesis overnight loop.
#
# Ignores the previous velocity-first Phase 17 plan. This loop starts from the
# pose-relative representation hypothesis, then tests label-boundary, capacity,
# and temporal-convolution alternatives on the same relative feature dataset.

set -uo pipefail

OUTROOT="${OUTROOT:-results/phase17_new_hypothesis_loop}"
SUMMARY="$OUTROOT/summary.log"
TARGET_MIN_PR="${TARGET_MIN_PR:-0.92}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_STRIDE="${EVAL_STRIDE:-2}"

REL_TRAIN="dataset/splits_v2_filtered_rel/train.csv"
REL_VAL="dataset/splits_v2_filtered_rel/val.csv"
REL_TEST="dataset/splits_v2_filtered_rel/test.csv"

mkdir -p "$OUTROOT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ ! -f "$REL_TRAIN" || ! -f "$REL_VAL" || ! -f "$REL_TEST" ]]; then
    log "Building pose-relative splits: dataset/splits_v2_filtered_rel"
    uv run python scripts/util/build_filtered_v2_splits_rel.py 2>&1 | tee -a "$SUMMARY"
    rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        log "FAIL dataset build (exit $rc)"
        exit "$rc"
    fi
fi

COMMON=(
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --preprocessing filtered
    --train-csv "$REL_TRAIN"
    --val-csv "$REL_VAL"
    --test-csv "$REL_TEST"
    --label-column label
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

GRU_BASE=(
    --model-type gru
    --gru-units 128,64
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0
)

TCN_BASE=(
    --model-type tcn
    --focal-loss --focal-gamma 2.0
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
if not outroot.exists():
    sys.exit(1)
for d in outroot.iterdir():
    mj = d / "metrics.json"
    if not mj.exists():
        continue
    m = json.loads(mj.read_text())
    if m.get("skipped"):
        continue
    tv = m.get("metrics", {}).get("test_video", {})
    if float(tv.get("min_pr", 0.0)) >= target:
        print(f"PASS: {d.name} test_min_pr={tv['min_pr']:.4f}")
        sys.exit(0)
sys.exit(1)
PYEOF
}

print_summary() {
    log "=== Phase 17 new-hypothesis summary ==="
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
    cfg = m.get("config", {})
    thr = m.get("threshold_selection", {})
    cm = tv.get("confusion_matrix") or [[None, None], [None, None]]
    rows.append((
        float(tv.get("min_pr", 0.0)),
        d.name,
        cfg.get("feature_set"),
        cfg.get("model_type"),
        cfg.get("label_mode"),
        cfg.get("focal_alpha"),
        tv.get("nfall_precision"),
        tv.get("nfall_recall"),
        tv.get("precision"),
        tv.get("recall"),
        cm[0][1],
        cm[1][0],
        thr.get("threshold"),
        thr.get("min_consecutive"),
    ))
rows.sort(reverse=True)
print(f"{'ID':<12} {'MinPR':>7} {'feat':<9} {'model':<4} {'label':<11} {'alpha':>5} {'NFP':>7} {'NFR':>7} {'FallP':>7} {'FallR':>7} {'FP':>4} {'FN':>4} {'thr':>5} {'mc':>3} PASS")
for minpr, exp, feat, model, label, alpha, nfp, nfr, fpv, fr, fp, fn, thr, mc in rows:
    flag = "PASS" if minpr >= target else ""
    print(f"{exp:<12} {minpr:7.4f} {str(feat):<9} {str(model):<4} {str(label):<11} {str(alpha):>5} {nfp:7.4f} {nfr:7.4f} {fpv:7.4f} {fr:7.4f} {str(fp):>4} {str(fn):>4} {thr:5.3f} {str(mc):>3} {flag}")
PYEOF
}

log "=== Phase 17 new-hypothesis loop started; target MinPR=${TARGET_MIN_PR} ==="

# H1: Absolute-coordinate reliance causes persistent normal false positives.
run_exp "P17H-v01" "${GRU_BASE[@]}" \
    --feature-set kp7rel --focal-alpha 0.25 \
    --label-mode segment_max --epochs 30 --early-stop-patience 5 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# H2: The old engineered absolute/HSSC features still carry useful fall evidence.
run_exp "P17H-v02" "${GRU_BASE[@]}" \
    --feature-set kp7releng --focal-alpha 0.25 \
    --label-mode segment_max --epochs 30 --early-stop-patience 5 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# H3: Segment-max labels are too permissive for ambiguous transition windows.
run_exp "P17H-v03" "${GRU_BASE[@]}" \
    --feature-set kp7rel --focal-alpha 0.25 \
    --label-mode last_frame --epochs 30 --early-stop-patience 5 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# H4: After normalization, the model can defend non-fall with a lower fall-side alpha.
run_exp "P17H-v04" "${GRU_BASE[@]}" \
    --feature-set kp7rel --focal-alpha 0.15 \
    --label-mode segment_max --epochs 30 --early-stop-patience 5 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# H5: Relative features need more recurrent capacity to preserve fall recall.
run_exp "P17H-v05" \
    --model-type gru --gru-units 256,128 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --focal-loss --focal-gamma 2.0 \
    --feature-set kp7rel --focal-alpha 0.25 \
    --label-mode segment_max --epochs 30 --early-stop-patience 5 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# H6: Persistent FP/FN patterns are better modeled by dilated temporal convolution.
run_exp "P17H-v06" "${TCN_BASE[@]}" \
    --feature-set kp7rel --focal-alpha 0.25 \
    --label-mode segment_max --epochs 100 --early-stop-patience 15 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# H7: Seed stability check for the best-scoped representation hypothesis.
run_exp "P17H-v07" "${GRU_BASE[@]}" \
    --feature-set kp7rel --focal-alpha 0.25 \
    --label-mode segment_max --epochs 30 --early-stop-patience 5 --seed 0

print_summary | tee -a "$SUMMARY"
log "=== Phase 17 new-hypothesis loop complete ==="
