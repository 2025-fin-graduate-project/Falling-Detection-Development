#!/usr/bin/env bash
# Phase 30: val_loss 체인 시작 — 아키텍처 × checkpoint 재설계
#
# 방향 전환 배경:
#   val_event/video_min_pr checkpoint → 후처리(threshold×mc) 격자에 과적합
#   → 모델이 낙상에 보수적(FN↑, NFallPrecision↓, FallRecall↓) 방향으로 학습
#   → val_loss로 전환: threshold-agnostic, 실질 분리 능력(class separability) 최적화
#
# 평가 기준 변경:
#   학습: val_loss 최소화 (checkpoint + early stop)
#   보고: test_video.min_pr, test_video.f1, test_video.precision, test_video.recall
#         event MinPR은 후처리 후 참고 지표
#
# 실험:
#   P30-gru:    GRU(128,64)  — 기존 최적 구성, val_loss로만 차이
#   P30-lstm:   LSTM(128,64) — forget gate, 장기 의존성
#   P30-tcn:    TCN [32,32,64,96] — causal dilated conv, window-based
#   P30-tcn-lg: TCN [64,64,128,128] kernel=5 — 더 큰 수용 필드

set -uo pipefail
cd "$(dirname "$0")/.."

OUTROOT="results/phase30_val_loss_chain"
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
    --epochs 100 --early-stop-patience 15 --batch-size 512
    --checkpoint-monitor val_loss
    --seed 42
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
vv = m["metrics"]["val_video"]

ev_mp = te['min_pr']
vid_mp = tv['min_pr']
vid_f1 = tv['f1']
vid_pr = tv.get('precision', 0)
vid_rc = tv.get('recall', 0)

flag = " *** TARGET!" if ev_mp >= 0.93 else (" ≥0.92" if ev_mp >= 0.92 else "")
print(f"[OK] {exp_id}{flag}")
print(f"  video: f1={vid_f1:.4f} pr={vid_pr:.4f} rc={vid_rc:.4f} MinP={vid_mp:.4f}")
print(f"  event: MinP={ev_mp:.4f} thr={thr['threshold']:.3f} mc={thr['min_consecutive']}")
cm = te.get("confusion_matrix")
if cm:
    print(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}  fall_pr={te.get('precision',0):.4f} nfall_pr={te.get('nfall_precision',0):.4f} fall_rc={te.get('recall',0):.4f}")
PYEOF
    else
        log "FAIL $id (exit $rc)"
    fi
}

log "=== Phase 30: val_loss chain — architecture × checkpoint 재설계 ==="
log "전략: val_loss checkpoint로 분리 능력 최적화 / event MinPR은 후처리 참고 지표"

# ── GRU(128,64): 기존 최적 구성에서 checkpoint만 변경
run_exp "P30-gru"    --model-type gru  --gru-units 128,64

# ── LSTM(128,64): forget gate 추가, STedgeAI stateful 호환
run_exp "P30-lstm"   --model-type lstm --gru-units 128,64

# ── TCN 기본: causal dilated conv [32,32,64,96]
run_exp "P30-tcn"    --model-type tcn

# ── TCN 확장: [64,64,128,128] kernel=5
run_exp "P30-tcn-lg" --model-type tcn \
    --tcn-channels 64,64,128,128 --tcn-dilations 1,2,4,8 --tcn-kernel-size 5

log "=== Phase 30 done ==="

python3 << 'PYEOF'
import json
from pathlib import Path

base = Path("results/phase30_val_loss_chain")
rows = []
for d in sorted(base.iterdir()):
    mj = d / "metrics.json"
    if not mj.exists(): continue
    m = json.loads(mj.read_text())
    te = m.get("metrics", {}).get("test_event_video", {})
    tv = m.get("metrics", {}).get("test_video", {})
    thr = m.get("threshold_selection", {})
    ev = te.get("min_pr", 0)
    vid_f1 = tv.get("f1", 0)
    vid_pr = tv.get("precision", 0)
    vid_rc = tv.get("recall", 0)
    vid_mp = tv.get("min_pr", 0)
    cm = te.get("confusion_matrix")
    fn = cm[1][0] if cm else "?"; fp = cm[0][1] if cm else "?"
    rows.append((ev, d.name, vid_f1, vid_pr, vid_rc, vid_mp, fn, fp))

rows.sort(reverse=True)
print("\n=== Phase 30 summary (val_loss chain) ===")
print(f"  {'ID':<14} {'EventMinP':>10} {'VidMinP':>8} {'VidF1':>7} {'VidPr':>7} {'VidRc':>7} {'FN':>4} {'FP':>4}")
print("-" * 80)
for ev, name, vf1, vpr, vrc, vmp, fn, fp in rows:
    flag = " *** TARGET!" if ev >= 0.93 else (" ≥0.92" if ev >= 0.92 else "")
    print(f"  {name:<14} {ev:>10.4f} {vmp:>8.4f} {vf1:>7.4f} {vpr:>7.4f} {vrc:>7.4f} {fn:>4} {fp:>4}{flag}")

gru_ref = 0.9241
best = rows[0]
print(f"\nPhase 30 best: {best[1]} event={best[0]:.4f}")
print(f"GRU val_event_min_pr ref (P27-vm0): {gru_ref:.4f}")
print(f"개선: {best[0] - gru_ref:+.4f}")
PYEOF
