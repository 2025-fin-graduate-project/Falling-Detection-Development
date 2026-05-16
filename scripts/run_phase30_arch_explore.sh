#!/usr/bin/env bash
# Phase 30: Architecture exploration — LSTM + TCN
# 가설: GRU(128,64) 파라미터 튜닝 한계(0.9241). 다른 아키텍처로 0.93+ 가능성 탐색.
#
# LSTM(128,64): GRU와 동일 크기, forget gate 추가로 장기 의존성 학습 가능성
# TCN default [32,32,64,96]: 기존 지원, causal dilated conv, 병렬 학습 효율적
# TCN large [64,64,128,128]: 더 큰 수용 필드로 낙상 패턴 포착
#
# 배포 호환성:
#   LSTM: STedgeAI stateful streaming 지원 (GRU와 동일 경로)
#   TCN: window-based (40f 전체 입력), STedgeAI Conv1D 지원 — streaming 아님

set -uo pipefail
cd "$(dirname "$0")/.."

OUTROOT="results/phase30_arch_explore"
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

COMMON=(
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
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
        "${COMMON[@]}" "$@" 2>&1 | tee "$logfile"
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
flag = " *** TARGET!" if ev >= 0.93 else (" ↑new best" if ev >= 0.9242 else (" ≥0.92" if ev >= 0.92 else ""))
print(f"[OK] {exp_id} event={ev:.4f} video={tv['min_pr']:.4f} thr={thr['threshold']:.3f} mc={thr['min_consecutive']}{flag}")
cm = te.get("confusion_matrix")
if cm:
    print(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}  fall_pr={te.get('precision',0):.4f} nfall_pr={te.get('nfall_precision',0):.4f}")
PYEOF
    else
        log "FAIL $id (exit $rc)"
    fi
}

log "=== Phase 30: Architecture exploration (LSTM + TCN) ==="
log "기준: P27-vm0 GRU(128,64) event=0.9241 / 목표: 0.93+"

# ── LSTM: GRU와 동일 크기, STedgeAI stateful streaming 호환
run_exp "P30-lstm"     --model-type lstm --gru-units 128,64

# ── LSTM seed=42 + 추가 seed (분산 확인)
run_exp "P30-lstm-s1"  --model-type lstm --gru-units 128,64 --seed 1
run_exp "P30-lstm-s2"  --model-type lstm --gru-units 128,64 --seed 2

# ── TCN default [32,32,64,96] dil=[1,2,4,8] kernel=3
run_exp "P30-tcn-d"    --model-type tcn

# ── TCN larger [64,64,128,128] — 더 큰 수용 필드
run_exp "P30-tcn-lg"   --model-type tcn \
    --tcn-channels 64,64,128,128 --tcn-dilations 1,2,4,8

# ── TCN wider kernel [64,64,128,128] kernel=5
run_exp "P30-tcn-k5"   --model-type tcn \
    --tcn-channels 64,64,128,128 --tcn-dilations 1,2,4,8 --tcn-kernel-size 5

log "=== Phase 30 done ==="

python3 << 'PYEOF'
import json
from pathlib import Path

base = Path("results/phase30_arch_explore")
baseline = 0.9241
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
    rows.append((ev, d.name, vv, fn, fp))

rows.sort(reverse=True)
print("\n=== Phase 30 summary ===")
print(f"  {'ID':<16} {'EventMinP':>10} {'VideoMinP':>10} {'FN':>4} {'FP':>4}  flag")
for ev, name, vv, fn, fp in rows:
    flag = " *** TARGET!" if ev >= 0.93 else (" ↑best" if ev > baseline else "")
    print(f"  {name:<16} {ev:>10.4f} {vv:>10.4f} {fn:>4} {fp:>4} {flag}")

best = rows[0][0] if rows else 0
print(f"\nPhase 30 best: {best:.4f}  baseline(GRU P27-vm0): {baseline:.4f}  target: 0.9300")
hits = [r for r in rows if r[0] >= 0.93]
print(f"≥0.93 달성: {len(hits)}건" + (" — 목표 달성!" if hits else ""))
PYEOF
