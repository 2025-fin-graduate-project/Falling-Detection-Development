#!/usr/bin/env bash
# Phase 17: Per-keypoint velocity features (kp7kv) sweep
# 새 피처: kp0/5/6/11/12 각각의 vy, vx — 개별 관절 운동 정보 추가
# 기대 효과: 낙상 시 몸통 기울기 (어깨↓ 엉덩이↑ 또는 반대)가 신호로 작용 → FN↓
# 데이터셋: dataset/splits_v2_filtered_kv/ (build_filtered_v2_splits_kv.py로 생성)
# 기준: P9O-v01 MinPR=0.9187 (FP=19 FN=20)

set -uo pipefail

OUTROOT="${OUTROOT:-results/phase17_velocity_features}"
SUMMARY="$OUTROOT/summary.log"

BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_STRIDE="${EVAL_STRIDE:-2}"
TARGET_MIN_PR="${TARGET_MIN_PR:-0.92}"

KV_TRAIN="dataset/splits_v2_filtered_kv/train.csv"
KV_VAL="dataset/splits_v2_filtered_kv/val.csv"
KV_TEST="dataset/splits_v2_filtered_kv/test.csv"

# 데이터셋 존재 확인
if [[ ! -f "$KV_TRAIN" ]]; then
    echo "ERROR: $KV_TRAIN not found. Run: uv run python scripts/util/build_filtered_v2_splits_kv.py" >&2
    exit 1
fi

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTROOT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

# P9O-v01 조건(30에폭+patience=5)이 best이므로 그걸 기본으로
COMMON=(
    --model-type gru
    --gru-units 128,64
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7kv
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0
    --preprocessing filtered
    --train-csv "$KV_TRAIN"
    --val-csv   "$KV_VAL"
    --test-csv  "$KV_TEST"
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
    log "=== Phase 17 summary ==="
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

# v01: α=0.25, 30에폭, seed=42 (P9O-v01 조건 + velocity 피처)
run_exp "P17-v01" \
    --focal-alpha 0.25 --epochs 30 --early-stop-patience 5 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v02: α=0.25, 100에폭 (velocity가 긴 훈련에서도 도움이 되는지 확인)
run_exp "P17-v02" \
    --focal-alpha 0.25 --epochs 100 --early-stop-patience 15 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v03: α=0.30, 30에폭 (FN 추가 개선 시도)
run_exp "P17-v03" \
    --focal-alpha 0.30 --epochs 30 --early-stop-patience 5 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v04: α=0.25, 30에폭, seed=0 (seed 다양화)
run_exp "P17-v04" \
    --focal-alpha 0.25 --epochs 30 --early-stop-patience 5 --seed 0
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

print_summary | tee -a "$SUMMARY"
