#!/usr/bin/env bash
# Phase 35: --no-class-weight × focal-alpha 탐색 + hflip 증강
#
# 분석 기반:
#   - 훈련 데이터: 60.5% fall / 39.5% non-fall (splits_v2_class_balanced_filtered)
#   - class_weight: fall_w=0.826, nfall_w=1.266 (불균형 보정 → 비율 동일화)
#   - 현재(P27-vm0): class_weight × focal(α=0.25) → 총 gradient 비율 = 3.0x non-fall 우세
#     계산: 0.605 × 0.826 × 0.25 fall / 0.395 × 1.266 × 0.75 nfall = 0.125/0.375 = 1:3
#   - 목표: 총 gradient 비율 1:1 달성 → --no-class-weight + focal α=0.40
#     계산: 0.605 × 0.40 fall / 0.395 × 0.60 nfall = 0.242/0.237 ≈ 1:1
#
# 실험 설계:
#   P35-ncw:        no-class-weight + α=0.25 → 1:1.96 (현재 1:3 → 개선)
#   P35-ncw-a35:    no-class-weight + α=0.35 → 1:1.37
#   P35-ncw-a40:    no-class-weight + α=0.40 → 1:0.98 ≈ 균형 (핵심 실험)
#   P35-ncw-a45:    no-class-weight + α=0.45 → 1:0.76 (fall 우세, FP↑ 가능)
#   P35-hflip:      baseline + hflip_prob=0.30 (새 증강)
#   P35-ncw-a40-hf: no-class-weight + α=0.40 + hflip=0.30 (균형 + 증강)
#
# 기준: P27-vm0 (GRU focal α=0.25 + val_vm + seed=42 + class_weight) = 0.9241
#       FN=18, FP=13 → FN이 병목 (nfall_pred_pr 제한)

set -uo pipefail
cd "$(dirname "$0")/.."

OUTROOT="results/phase35_ncw"
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

# P27-vm0 기준 설정 (class_weight 포함 버전)
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

log "=== Phase 35: --no-class-weight × focal-alpha + hflip 탐색 ==="
log "기준: P27-vm0(GRU focal α=0.25 + class_weight + val_vm + seed=42) = 0.9241, FN=18"
log "핵심: α=0.40 + no-class-weight → 총 gradient 비율 ≈ 1:1 (균형)"

# ── (1) no-class-weight 단독: focal alpha 변화
# α=0.25: 1:1.96 non-fall 우세 (현재 1:3에서 개선)
run_exp "P35-ncw" "${BASE[@]}" --focal-alpha 0.25 --no-class-weight

# α=0.35: 1:1.37 non-fall 우세
run_exp "P35-ncw-a35" "${BASE[@]}" --focal-alpha 0.35 --no-class-weight

# α=0.40: 1:0.98 ≈ 균형 (핵심 실험)
run_exp "P35-ncw-a40" "${BASE[@]}" --focal-alpha 0.40 --no-class-weight

# α=0.45: 1:0.76 fall 우세 (FN↓ but FP↑ 가능)
run_exp "P35-ncw-a45" "${BASE[@]}" --focal-alpha 0.45 --no-class-weight

# ── (2) 수평 뒤집기 증강 (class_weight 유지)
# hflip_prob=0.30: 30% 확률로 좌우 반전 (어깨/팔꿈치/골반 keypoint 쌍 교환 + x좌표 반전)
run_exp "P35-hflip" "${BASE[@]}" --focal-alpha 0.25 --hflip-prob 0.30

# ── (3) 균형 훈련 + 증강 결합
run_exp "P35-ncw-a40-hf" "${BASE[@]}" --focal-alpha 0.40 --no-class-weight --hflip-prob 0.30

log "=== Phase 35 done ==="

python3 << 'PYEOF'
import json
from pathlib import Path

base = Path("results/phase35_ncw")
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
print("\n=== Phase 35 summary ===")
print(f"  {'ID':<18} {'EventMinP':>10} {'VidMinP':>8} {'FN':>4} {'FP':>4}  {'fall_pr':>8}  {'nfall_pr':>8}  {'fall_rc':>8}")
for ev, name, vmp, fn, fp, fp_pr, nfp_pr, rc in rows:
    flag = " *** TARGET!" if ev >= 0.93 else (" ↑best!" if ev > 0.9241 else "")
    bias = "balanced" if name == "P35-ncw-a40" else ""
    print(f"  {name:<18} {ev:>10.4f} {vmp:>8.4f} {fn:>4} {fp:>4}  {fp_pr:>8.4f}  {nfp_pr:>8.4f}  {rc:>8.4f}{flag}  {bias}")

if rows:
    best = rows[0]
    print(f"\n최고: {best[1]} = {best[0]:.4f}  (기준 P27-vm0: 0.9241, 목표: 0.9300)")
print("\n기준 대비 FN 변화 (P27-vm0 FN=18):")
for ev, name, vmp, fn, fp, *_ in rows:
    if isinstance(fn, int):
        delta = fn - 18
        print(f"  {name:<18}: FN={fn} ({delta:+d} vs P27-vm0)")
PYEOF
