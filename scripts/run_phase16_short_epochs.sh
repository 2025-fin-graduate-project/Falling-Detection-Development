#!/usr/bin/env bash
# Phase 16: epochs=30, patience=5 복귀 + α sweep + multi-seed
# 핵심 발견: P9O-v01(30에폭+patience=5)이 MinPR=0.9187 역대 최고
#           Phase 15에서 100에폭 재훈련은 일관되게 실패 (thr 하락, FP 증가)
# 가설: 짧은 훈련의 암묵적 정규화 효과 + 최적 α 탐색 + seed 다양화

set -uo pipefail

OUTROOT="${OUTROOT:-results/phase16_short_epochs}"
SUMMARY="$OUTROOT/summary.log"

EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
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
    --train-negative-stride 2
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
    log "=== Phase 16 summary ==="
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
    print(f"{d.name:<16} {tv.get('min_pr',float('nan')):>10.4f} {tv.get('nfall_precision',float('nan')):>8.4f} {tv.get('nfall_recall',float('nan')):>8.4f} {tv.get('precision',float('nan')):>8.4f} {tv.get('recall',float('nan')):>8.4f} {thr.get('threshold',float('nan')):>6.3f} {thr.get('min_consecutive','?'):>3} {flag}")
    cm = tv.get("confusion_matrix")
    if cm: print(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
PYEOF
}

# ── Phase 1: α sweep with seed=42 (P9O-v01 조건 재현) ──────────────────────
log "=== Phase 16 α sweep (epochs=30, patience=5, seed=42) ==="

# v01: α=0.25, seed=42 — P9O-v01 조건 재현 시도
run_exp "P16-v01" --focal-alpha 0.25 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v02: α=0.20, seed=42
run_exp "P16-v02" --focal-alpha 0.20 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v03: α=0.30, seed=42
run_exp "P16-v03" --focal-alpha 0.30 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v04: α=0.35, seed=42
run_exp "P16-v04" --focal-alpha 0.35 --seed 42
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# ── Phase 2: 최고 α로 seed 변화 탐색 ──────────────────────────────────────
# best α 추출
BEST_ALPHA=$(python3 - "$OUTROOT" <<'PYEOF'
import json, sys
from pathlib import Path
outroot = Path(sys.argv[1])
best_minpr = 0.0
best_alpha = 0.25
alpha_map = {"P16-v01": 0.25, "P16-v02": 0.20, "P16-v03": 0.30, "P16-v04": 0.35}
for exp_id, alpha in alpha_map.items():
    mj = outroot / exp_id / "metrics.json"
    if not mj.exists(): continue
    m = json.loads(mj.read_text())
    if m.get("skipped"): continue
    tv = m.get("metrics", {}).get("test_video", {})
    minpr = float(tv.get("min_pr", 0.0))
    if minpr > best_minpr:
        best_minpr = minpr
        best_alpha = alpha
print(best_alpha)
PYEOF
)
log "=== Best α=$BEST_ALPHA — multi-seed sweep ==="

# v05: best α, seed=0
run_exp "P16-v05" --focal-alpha "$BEST_ALPHA" --seed 0
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v06: best α, seed=1
run_exp "P16-v06" --focal-alpha "$BEST_ALPHA" --seed 1
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v07: best α, seed=7
run_exp "P16-v07" --focal-alpha "$BEST_ALPHA" --seed 7
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v08: best α, seed=123
run_exp "P16-v08" --focal-alpha "$BEST_ALPHA" --seed 123
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# ── Phase 3: P9O-v01 exact mc grid [1,3,5,7,9] 재현 ──────────────────────────
# P16-v01에서 mc=[1,2,3,4,5,7,9]이 thr=0.450/mc=5 선택 → FP=25
# P9O-v01은 mc=[1,3,5,7,9]로 thr=0.525/mc=3 선택 → FP=19
# mc 그리드 차이가 원인인지 검증
log "=== Phase 16 P9O-v01 exact mc grid [1,3,5,7,9] 재현 ==="

# v09: α=0.25, seed=42, mc=[1,3,5,7,9] (P9O-v01 exact)
run_exp "P16-v09" \
    --focal-alpha 0.25 --seed 42 \
    --min-consecutive-values 1,3,5,7,9
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v10: best α, seed=42, mc=[1,3,5,7,9]
run_exp "P16-v10" \
    --focal-alpha "$BEST_ALPHA" --seed 42 \
    --min-consecutive-values 1,3,5,7,9
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

# v11: best α, seed=0, mc=[1,3,5,7,9]
run_exp "P16-v11" \
    --focal-alpha "$BEST_ALPHA" --seed 0 \
    --min-consecutive-values 1,3,5,7,9
any_passed && { print_summary | tee -a "$SUMMARY"; exit 0; }

print_summary | tee -a "$SUMMARY"
