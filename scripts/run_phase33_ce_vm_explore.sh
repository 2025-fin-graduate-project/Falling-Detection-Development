#!/usr/bin/env bash
# Phase 33: CE × val_video_min_pr + LSTM×CE 탐색
#
# 가설:
#   (1) GRU+CE+val_video_min_pr: P27-vm0(focal+vm=0.9241)의 focal → CE 교체
#       CE가 val_loss에서 FN 22→17 감소 효과를 val_vm checkpoint에서도 재현하면 0.93+ 가능
#   (2) LSTM+CE+val_loss: P30-lstm(focal+vl=0.9181)의 focal → CE 교체
#       LSTM 균형 학습 특성 + CE 결합
#
# 체인: P27(GRU vm 0.9241) → P30(val_loss 체인) → P31(LSTM×ckpt) →
#       P32(GRU vl 탐색) → P33(CE×vm 핵심 미탐색)

set -uo pipefail
cd "$(dirname "$0")/.."

OUTROOT="results/phase33_ce_vm_explore"
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

# GRU + CE + val_video_min_pr (핵심 미탐색 조합)
GRU_CE_VM=(
    --model-type gru --gru-units 128,64
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --preprocessing filtered
    --train-csv dataset/splits_v2_class_balanced_filtered/train.csv
    --val-csv   dataset/splits_v2_class_balanced_filtered/val.csv
    --test-csv  dataset/splits_v2_class_balanced_filtered/test.csv
    --label-column label --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2 --eval-stride 2
    --epochs 100 --early-stop-patience 15 --batch-size 512
    --checkpoint-monitor val_video_min_pr
    --min-consecutive-values 1,2,3,4,5,7,9 --threshold-count 37
    --threshold-eval-level event --event-tolerance-windows 2
    --no-export-tflite --quiet
)

# LSTM + CE + val_loss
LSTM_CE_VL=(
    --model-type lstm --gru-units 128,64
    --target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0
    --feature-set kp7
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --preprocessing filtered
    --train-csv dataset/splits_v2_class_balanced_filtered/train.csv
    --val-csv   dataset/splits_v2_class_balanced_filtered/val.csv
    --test-csv  dataset/splits_v2_class_balanced_filtered/test.csv
    --label-column label --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2 --eval-stride 2
    --epochs 100 --early-stop-patience 15 --batch-size 512
    --checkpoint-monitor val_loss
    --min-consecutive-values 1,2,3,4,5,7,9 --threshold-count 37
    --threshold-eval-level event --event-tolerance-windows 2
    --no-export-tflite --quiet
)

run_exp() {
    local id="$1"; shift
    local args=("$@")
    local logfile="$OUTROOT/${id}.log"
    [[ -f "$OUTROOT/$id/metrics.json" ]] && { log "SKIP $id"; return 0; }
    wait_for_gpu
    log "START $id"
    uv run python scripts/train_baseline.py \
        --experiment-id "$id" --output-root "$OUTROOT" \
        "${args[@]}" 2>&1 | tee "$logfile"
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
flag = " *** TARGET!" if ev >= 0.93 else (" ↑best!" if ev > 0.9241 else (" ≥0.92" if ev >= 0.92 else ""))
print(f"[OK] {exp_id} event={ev:.4f} video={tv['min_pr']:.4f}{flag}")
cm = te.get("confusion_matrix")
if cm:
    print(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}  fall_pr={te.get('precision',0):.4f} nfall_pr={te.get('nfall_precision',0):.4f} fall_rc={te.get('recall',0):.4f}")
PYEOF
    else
        log "FAIL $id (exit $rc)"
    fi
}

log "=== Phase 33: CE × val_video_min_pr + LSTM×CE 탐색 ==="
log "기준: P27-vm0(GRU focal+vm 0.9241) / P30-lstm(LSTM focal+vl 0.9181)"

# ── (1) GRU + CE + val_video_min_pr: 핵심 미탐색 조합
run_exp "P33-gce-vm42" "${GRU_CE_VM[@]}" --seed 42
run_exp "P33-gce-vm1"  "${GRU_CE_VM[@]}" --seed 1
run_exp "P33-gce-vm0"  "${GRU_CE_VM[@]}" --seed 0

# ── (2) LSTM + CE + val_loss: P30-lstm의 focal → CE 교체
run_exp "P33-lce-vl42" "${LSTM_CE_VL[@]}" --seed 42
run_exp "P33-lce-vl1"  "${LSTM_CE_VL[@]}" --seed 1

log "=== Phase 33 done ==="

python3 << 'PYEOF'
import json
from pathlib import Path

base = Path("results/phase33_ce_vm_explore")
rows = []
for d in sorted(base.iterdir()):
    mj = d / "metrics.json"
    if not mj.exists(): continue
    m = json.loads(mj.read_text())
    te = m.get("metrics", {}).get("test_event_video", {})
    tv = m.get("metrics", {}).get("test_video", {})
    ev = te.get("min_pr", 0)
    cm = te.get("confusion_matrix")
    fn = cm[1][0] if cm else "?"; fp = cm[0][1] if cm else "?"
    rows.append((ev, d.name, tv.get("min_pr",0), fn, fp,
                 te.get("precision",0), te.get("nfall_precision",0), te.get("recall",0)))

rows.sort(reverse=True)
print("\n=== Phase 33 summary ===")
print(f"  {'ID':<16} {'EventMinP':>10} {'VidMinP':>8} {'FN':>4} {'FP':>4}  {'fall_pr':>8}  {'nfall_pr':>8}  {'fall_rc':>8}")
for ev, name, vmp, fn, fp, fp_pr, nfp_pr, rc in rows:
    flag = " *** TARGET!" if ev >= 0.93 else (" ↑best!" if ev > 0.9241 else "")
    print(f"  {name:<16} {ev:>10.4f} {vmp:>8.4f} {fn:>4} {fp:>4}  {fp_pr:>8.4f}  {nfp_pr:>8.4f}  {rc:>8.4f}{flag}")

best = rows[0][0] if rows else 0
print(f"\n최고: {rows[0][1] if rows else '?'} = {best:.4f}  (목표: 0.93, 전체최고 P27-vm0: 0.9241)")
PYEOF
