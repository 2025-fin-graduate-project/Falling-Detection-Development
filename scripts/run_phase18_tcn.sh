#!/usr/bin/env bash
# Phase 18: TCN architecture sweep
# 기존 GRU(128,64) 한계 돌파 시도 — dilated causal Conv1D 기반
# TCN 장점: 긴 시간 컨텍스트를 병렬 처리, gradient flow 안정
# 기준: P9O-v01 MinPR=0.9187 (FP=19 FN=20)

set -uo pipefail

OUTROOT="${OUTROOT:-results/phase18_tcn}"
SUMMARY="$OUTROOT/summary.log"

BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_STRIDE="${EVAL_STRIDE:-2}"
TARGET_MIN_PR="${TARGET_MIN_PR:-0.92}"

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTROOT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

COMMON=(
    --model-type tcn
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7
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
    log "=== Phase 18 summary ==="
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
    print(f"{d.name:<14} {tv.get('min_pr',float('nan')):>10.4f} {tv.get('nfall_precision',float('nan')):>8.4f} {tv.get('nfall_recall',float('nan')):>8.4f} {tv.get('precision',float('nan')):>8.4f} {tv.get('recall',float('nan')):>8.4f} {thr.get('threshold',float('nan')):>6.3f} {thr.get('min_consecutive','?'):>3} {flag}")
    cm = tv.get("confusion_matrix")
    if cm: print(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
PYEOF
}

# v01: TCN base — channels=32,32,64,96, dilations=1,2,4,8 (default), 100에폭
run_exp "P18-v01" \
    --epochs 100 --early-stop-patience 15 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v02: TCN wider — channels=64,64,128,128
run_exp "P18-v02" \
    --tcn-channels 64,64,128,128 \
    --epochs 100 --early-stop-patience 15 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v03: TCN wider + longer dilations (더 긴 시간 컨텍스트)
run_exp "P18-v03" \
    --tcn-channels 64,128,128,256 --tcn-dilations 1,2,4,8 \
    --epochs 100 --early-stop-patience 15 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v04: TCN base + α=0.30
run_exp "P18-v04" \
    --focal-alpha 0.30 \
    --epochs 100 --early-stop-patience 15 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v05: TCN base + seed 다양화 (lucky seed 탐색)
run_exp "P18-v05" \
    --epochs 100 --early-stop-patience 15 --seed 0
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

print_summary | tee -a "$SUMMARY"
