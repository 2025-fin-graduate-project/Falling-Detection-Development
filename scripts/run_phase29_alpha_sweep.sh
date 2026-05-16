#!/usr/bin/env bash
# Phase 29: focal alpha sweep — FN 감소를 위한 구조적 접근
# 가설: focal_alpha=0.25는 non-fall에 2x 학습 신호 → FN=18(놓친 낙상) 과다.
#       alpha 증가로 fall 클래스 강조 → FN≤16 → NFallPrecision≥0.93 달성.
#
# 기준: P27-vm0 (seed=42, val_video_min_pr) — FallPrecision=0.979(여유 4.9%p)
# 실험: alpha ∈ {0.35, 0.50, 0.75}
# 추가: noise_std ∈ {0.05} × seed=42 (hard 샘플 강건성)

set -uo pipefail
cd "$(dirname "$0")/.."

OUTROOT="results/phase29_alpha_sweep"
SUMMARY="$OUTROOT/summary.log"
GPU_WAIT_MAX_USED_MB=2200
GPU_WAIT_INTERVAL_SEC=60

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTROOT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

wait_for_gpu() {
    while true; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
        if [[ -n "$used" && "$used" -le "$GPU_WAIT_MAX_USED_MB" ]]; then
            log "GPU ready: ${used}MiB"; return 0
        fi
        log "GPU busy: ${used:-?}MiB — waiting ${GPU_WAIT_INTERVAL_SEC}s"
        sleep "$GPU_WAIT_INTERVAL_SEC"
    done
}

BASE_ARGS=(
    --model-type gru --gru-units 128,64
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0
    --preprocessing filtered
    --train-csv dataset/splits_v2_class_balanced_filtered/train.csv
    --val-csv   dataset/splits_v2_class_balanced_filtered/val.csv
    --test-csv  dataset/splits_v2_class_balanced_filtered/test.csv
    --label-column label --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2 --eval-stride 2
    --epochs 30 --early-stop-patience 5 --batch-size 512
    --checkpoint-monitor val_video_min_pr --seed 42
    --min-consecutive-values 1,2,3,4,5,7,9 --threshold-count 37
    --threshold-eval-level event --event-tolerance-windows 2
    --no-export-tflite --quiet
)

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    [[ -f "$OUTROOT/$id/metrics.json" ]] && { log "SKIP $id"; return 0; }
    wait_for_gpu
    log "START $id"
    uv run python scripts/train_baseline.py \
        --experiment-id "$id" --output-root "$OUTROOT" \
        "${BASE_ARGS[@]}" "$@" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    if [[ "$rc" -eq 0 ]]; then
        python3 - "$OUTROOT" "$id" <<'PYEOF' | tee -a "$SUMMARY"
import json, sys
from pathlib import Path
outroot = Path(sys.argv[1]); exp_id = sys.argv[2]
m = json.loads((outroot / exp_id / "metrics.json").read_text())
thr = m["threshold_selection"]
te = m["metrics"]["test_event_video"]
tv = m["metrics"]["test_video"]
ev = te['min_pr']
flag = " *** TARGET!" if ev >= 0.93 else (" ← best?" if ev >= 0.92 else "")
print(f"[OK] {exp_id} event={ev:.4f} video={tv['min_pr']:.4f} thr={thr['threshold']:.3f} mc={thr['min_consecutive']}{flag}")
cm = te.get("confusion_matrix")
if cm:
    fp_pr = te.get('precision', 0); nfp_pr = te.get('nfall_precision', 0)
    print(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}  fall_pr={fp_pr:.4f} nfall_pr={nfp_pr:.4f}")
PYEOF
    else
        log "FAIL $id (exit $rc)"
    fi
}

log "=== Phase 29: focal alpha sweep (seed=42, val_video_min_pr) ==="
log "기준: P27-vm0 alpha=0.25 → event=0.9241 FN=18 FallPr=0.979 NFallPr=0.924"
log "목표: NFallPr≥0.93 (FN≤16). FallPr 여유 4.9%p 활용."

# ── 그룹 A: alpha sweep (gamma=2.0 고정)
run_exp "P29-a35"  --focal-alpha 0.35
run_exp "P29-a50"  --focal-alpha 0.50
run_exp "P29-a75"  --focal-alpha 0.75

# ── 그룹 B: noise-std 강화 (alpha=0.25 기준)
# hard 낙상 샘플에 robust한 표현 학습 유도
run_exp "P29-n5"   --focal-alpha 0.25 --noise-std 0.05
run_exp "P29-n10"  --focal-alpha 0.25 --noise-std 0.10

# ── 그룹 C: alpha + noise 조합 (최유망 조합)
run_exp "P29-a50n5" --focal-alpha 0.50 --noise-std 0.05

log "=== Phase 29 done ==="

python3 << 'PYEOF'
import json
from pathlib import Path

base = Path("results/phase29_alpha_sweep")
baseline = 0.9241  # P27-vm0
rows = []
for d in sorted(base.iterdir()):
    mj = d / "metrics.json"
    if not mj.exists(): continue
    m = json.loads(mj.read_text())
    te = m.get("metrics", {}).get("test_event_video", {})
    tv = m.get("metrics", {}).get("test_video", {})
    thr = m.get("threshold_selection", {})
    ev = te.get("min_pr", 0); vv = tv.get("min_pr", 0)
    cm = te.get("confusion_matrix")
    fn = cm[1][0] if cm else "?"; fp = cm[0][1] if cm else "?"
    fp_pr = te.get("precision", 0); nfp_pr = te.get("nfall_precision", 0)
    rows.append((ev, d.name, vv, fn, fp, fp_pr, nfp_pr))

rows.sort(reverse=True)
print("\n=== Phase 29 summary ===")
print(f"  {'ID':<14} {'EventMinP':>10} {'VideoMinP':>10} {'FN':>4} {'FP':>4}  {'FallPr':>8}  {'NFallPr':>8}  flag")
for ev, name, vv, fn, fp, fp_pr, nfp_pr in rows:
    flag = " *** TARGET!" if ev >= 0.93 else (" ↑new best" if ev > baseline else "")
    print(f"  {name:<14} {ev:>10.4f} {vv:>10.4f} {fn:>4} {fp:>4}  {fp_pr:>8.4f}  {nfp_pr:>8.4f} {flag}")

best = rows[0][0] if rows else 0
print(f"\nPhase 29 best: {best:.4f}  baseline(P27-vm0): {baseline:.4f}  target: 0.9300")
hits = [r for r in rows if r[0] >= 0.93]
print(f"≥0.93 달성: {len(hits)}건" + (" — 목표 달성!" if hits else " — Phase 30 전략 필요"))
PYEOF
