#!/usr/bin/env bash
# Phase 34: Augmentation × Focal Alpha 탐색
#
# 분석 기반:
#   - 모든 31개 실험에서 FN이 min_pr 병목: nfall_pred_pr = TN/(TN+FN) < fall_pred_pr
#   - 목표: FN 18→≤16 (현재 최선 P27-vm0: FN=18, FP=13)
#   - FP 여유: fall_pred_pr=0.97~0.98, FP 최대 ~45까지 허용
#
# 두 방향 동시 탐색:
#   (1) Focal alpha ↑ (0.25→0.35/0.40/0.45): fall 클래스 gradient 비중 증가 → FN 감소 기대
#       - val_video_min_pr 체크포인트 유지 (P27-vm0 최적 조합)
#       - Phase 29 실패(val_event_min_pr)와 다름: val_vm은 덜 편향적
#   (2) 데이터 증강: feature masking (키포인트 랜덤 마스킹), time masking (연속 프레임 마스킹)
#       - FN 감소보다는 일반화 향상으로 전반적 오류 감소 기대
#
# 기준: P27-vm0 (GRU focal α=0.25 + val_vm + seed=42) = 0.9241

set -uo pipefail
cd "$(dirname "$0")/.."

OUTROOT="results/phase34_aug_alpha"
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

# P27-vm0 기준 설정 (focal α=0.25, val_vm, seed=42)
BASE=(
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
    --epochs 100 --early-stop-patience 15 --batch-size 512
    --checkpoint-monitor val_video_min_pr
    --seed 42
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

log "=== Phase 34: Augmentation × Focal Alpha 탐색 ==="
log "기준: P27-vm0(GRU focal α=0.25 + val_vm + seed=42) = 0.9241"
log "목표: FN 18→≤16 (FP 여유 충분: 현재 FP=13)"

# ── (1) Focal alpha 상향: fall gradient 비중 증가 → FN 감소 목표
run_exp "P34-a35" "${BASE[@]}" --focal-alpha 0.35
run_exp "P34-a40" "${BASE[@]}" --focal-alpha 0.40
run_exp "P34-a45" "${BASE[@]}" --focal-alpha 0.45

# ── (2) Feature masking 증강: 키포인트 랜덤 마스킹 (occlusion 시뮬레이션)
run_exp "P34-fm10" "${BASE[@]}" --focal-alpha 0.25 --feat-mask-prob 0.10
run_exp "P34-fm20" "${BASE[@]}" --focal-alpha 0.25 --feat-mask-prob 0.20

# ── (3) Time masking 증강: 연속 프레임 마스킹 (tracking loss 시뮬레이션)
run_exp "P34-tm3" "${BASE[@]}" --focal-alpha 0.25 --time-mask-max 3
run_exp "P34-tm5" "${BASE[@]}" --focal-alpha 0.25 --time-mask-max 5

# ── (4) 결합: alpha 상향 + feature masking (가장 유망한 조합)
run_exp "P34-a35-fm10" "${BASE[@]}" --focal-alpha 0.35 --feat-mask-prob 0.10

log "=== Phase 34 done ==="

python3 << 'PYEOF'
import json
from pathlib import Path

base = Path("results/phase34_aug_alpha")
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
print("\n=== Phase 34 summary ===")
print(f"  {'ID':<16} {'EventMinP':>10} {'VidMinP':>8} {'FN':>4} {'FP':>4}  {'fall_pr':>8}  {'nfall_pr':>8}  {'fall_rc':>8}")
for ev, name, vmp, fn, fp, fp_pr, nfp_pr, rc in rows:
    flag = " *** TARGET!" if ev >= 0.93 else (" ↑best!" if ev > 0.9241 else "")
    print(f"  {name:<16} {ev:>10.4f} {vmp:>8.4f} {fn:>4} {fp:>4}  {fp_pr:>8.4f}  {nfp_pr:>8.4f}  {rc:>8.4f}{flag}")

best = rows[0][0] if rows else 0
print(f"\n최고: {rows[0][1] if rows else '?'} = {best:.4f}  (목표: 0.93, 전체최고 P27-vm0: 0.9241)")
print("기준 대비 FN 변화:")
for ev, name, vmp, fn, fp, *_ in rows:
    delta = fn - 18  # P27-vm0 FN=18 대비
    print(f"  {name:<16}: FN={fn} ({delta:+d} vs P27-vm0)")
PYEOF
