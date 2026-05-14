#!/usr/bin/env bash
# Phase 2 — Architecture Sweep for LB-2 Binary Detection
#
# Goal: push LB-2 F1 above 0.91 AND min_prec above 0.90
# Best so far: filtered bidir F1=0.9220, min_prec=0.8903 (Phase 1 v10, still running)
#
# Fixed: filtered preprocessing, LB-2, kp12, all standard hyperparams
# Variable: model arch + focal loss
#
# Quantization compatibility:
#   GRU      — CPU fallback (INT8 ok), Neural-ART NPU 미지원
#   TCN      — CONV_2D 기반 → Neural-ART NPU 풀 가속, 최우선
#
# Experiments:
#   P2-v01: GRU(256,128) bidir + focal_loss        [current best + focal]
#   P2-v02: GRU(128,64)  bidir                     [compact GRU, 배포 최적화]
#   P2-v03: GRU(128,64)  bidir + focal_loss        [compact + focal]
#   P2-v04: TCN [64,64,128,128] dil[1,2,4,8]  k3  [TCN standard — NPU 가속]
#   P2-v05: TCN [64,128,128,256] dil[1,2,4,8] k3  [TCN large — NPU 가속]
#   P2-v06: TCN [64,64,128,128] + focal_loss       [TCN + focal]
#   P2-v07: GRU(256,128) bidir, conv-pre 0 layers  [conv-pre 기여도 확인]
#
# Culling rule: 각 그룹 완료 후 F1 < 0.91 AND min_prec < 0.89 이면 SKIP 표시

set -uo pipefail

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OUTROOT="results/gru_phase2_arch"
SUMMARY="$OUTROOT/summary.log"
mkdir -p "$OUTROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id — already complete"
        return 0
    fi
    log "START $id"
    if uv run python scripts/train_baseline.py \
            --experiment-id "$id" \
            --output-root   "$OUTROOT" \
            --quiet \
            "$@" \
            2>&1 | tee "$logfile"; then
        log "OK    $id"
    else
        log "FAIL  $id"
    fi
}

# ── Fixed base config ─────────────────────────────────────────────────────────
BASE=(
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label
    --feature-set kp12
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --early-stop-patience 15 --epochs 100
    --min-val-precision 0.90
)

GRU_COMMON=(
    --model-type gru
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
)

# ════════════════════════════════════════════════════════════════════════════════
# Group A: GRU variants
# ════════════════════════════════════════════════════════════════════════════════
log "=== GROUP A: GRU variants ==="

# v01: 현재 최고 (filtered bidir) + focal loss → precision 개선 기대
run_exp P2-v01 "${BASE[@]}" "${GRU_COMMON[@]}" \
    --gru-units 256,128 --bidirectional \
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25

# v02: 소형화 (128,64 bidir) — 배포 크기 감소 + 일반화 개선 기대
run_exp P2-v02 "${BASE[@]}" "${GRU_COMMON[@]}" \
    --gru-units 128,64 --bidirectional

# v03: 소형화 + focal
run_exp P2-v03 "${BASE[@]}" "${GRU_COMMON[@]}" \
    --gru-units 128,64 --bidirectional \
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25

# Cull check: if all GRU variants < 0.91 F1, log warning
python3 -c "
import json, os
ids = ['P2-v01','P2-v02','P2-v03']
results = []
for i in ids:
    p = f'$OUTROOT/{i}/metrics.json'
    if os.path.exists(p):
        m = json.load(open(p))
        tv = m['metrics'].get('test_video',{})
        results.append((i, tv.get('f1',0), tv.get('min_precision',0)))
if results:
    best_f1 = max(r[1] for r in results)
    print(f'GRU group best F1: {best_f1:.4f}')
    for r in results:
        status = 'OK' if r[1]>=0.91 else 'BELOW'
        print(f'  {r[0]}: F1={r[1]:.4f} min_prec={r[2]:.4f} [{status}]')
" 2>/dev/null | tee -a "$SUMMARY"

# ════════════════════════════════════════════════════════════════════════════════
# Group B: TCN variants (NPU-accelerated on Neural-ART)
# ════════════════════════════════════════════════════════════════════════════════
log "=== GROUP B: TCN variants (NPU-friendly) ==="

# v04: TCN standard — [64,64,128,128] dil[1,2,4,8] k3
run_exp P2-v04 "${BASE[@]}" \
    --model-type tcn \
    --tcn-channels 64,64,128,128 --tcn-dilations 1,2,4,8 --tcn-kernel-size 3

# v05: TCN large — [64,128,128,256] dil[1,2,4,8] k3
run_exp P2-v05 "${BASE[@]}" \
    --model-type tcn \
    --tcn-channels 64,128,128,256 --tcn-dilations 1,2,4,8 --tcn-kernel-size 3

# v06: TCN standard + focal
run_exp P2-v06 "${BASE[@]}" \
    --model-type tcn \
    --tcn-channels 64,64,128,128 --tcn-dilations 1,2,4,8 --tcn-kernel-size 3 \
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25

# ════════════════════════════════════════════════════════════════════════════════
# Group C: ablation
# ════════════════════════════════════════════════════════════════════════════════
log "=== GROUP C: ablation ==="

# v07: GRU(256,128) bidir, conv-pre 제거 — conv-pre 기여도 확인
run_exp P2-v07 "${BASE[@]}" \
    --model-type gru \
    --gru-units 256,128 --bidirectional \
    --conv-pre-layers 0

# ════════════════════════════════════════════════════════════════════════════════
# Final summary
# ════════════════════════════════════════════════════════════════════════════════
log "All Phase 2 experiments complete."
echo ""
echo "=== PHASE 2 RESULTS ===" | tee -a "$SUMMARY"
printf "%-8s %-18s %7s %7s %8s %8s %6s\n" \
    "ID" "Arch" "testF1" "Rec" "FallP" "NFallP" "MinP" | tee -a "$SUMMARY"
echo "----------------------------------------------------------------------" | tee -a "$SUMMARY"

declare -A DESC=(
    [P2-v01]="GRU(256,128)bidir+focal"
    [P2-v02]="GRU(128,64)bidir"
    [P2-v03]="GRU(128,64)bidir+focal"
    [P2-v04]="TCN[64,64,128,128]"
    [P2-v05]="TCN[64,128,128,256]"
    [P2-v06]="TCN[64,64,128,128]+focal"
    [P2-v07]="GRU(256,128)bidir,noconv"
)

for id in P2-v01 P2-v02 P2-v03 P2-v04 P2-v05 P2-v06 P2-v07; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
m = json.load(open('$mfile'))
tv = m['metrics'].get('test_video', {})
print(f'%-8s %-18s %7.4f %7.4f %8.4f %8.4f %6.4f' % (
    '$id', '${DESC[$id]}',
    tv.get('f1',0), tv.get('recall',0),
    tv.get('precision',0), tv.get('nfall_precision',float('nan')),
    tv.get('min_precision',0),
))
" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "Phase 2 summary written to $SUMMARY"
