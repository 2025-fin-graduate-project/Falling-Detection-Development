#!/usr/bin/env bash
# Phase 8 — Conv 경량화 실험 + STedgeAI 파이프라인 검증
#
# 목표: MinP ≥ 0.93 유지하면서 Conv 스택 줄여 MACC/크기 감소
#   (Flash는 64 MB NOR Flash에 여유 충분 — 제약 없음)
#
# 핵심 발견 (Phase 7):
#   - GRU(128,64) + 2×Conv64 + kp7:  Flash=537 KiB, MACC=4.0M, MinP=0.934
#   - GRU(128,64) + 2×Conv64 + kp12: Flash=559 KiB, MACC=4.2M, MinP=0.939
#   - Conv 축소 전략: 레이어 수 감소 or 필터 수 감소
#
# 실험 설계:
#   P8-v01: 1×Conv64 + kp7     (Conv 1층 제거, MACC ↓)
#   P8-v02: 2×Conv32 + kp7     (필터 64→32, MACC ↓)
#   P8-v03: 1×Conv64 + minimal (최소 피처, 추론 최경량)
#
# Keras 호환:
#   main venv (TF 2.21/Keras 3.13) 사용 → export_stedgeai.py가 quantization_config 자동 strip
#
# STedgeAI 파이프라인:
#   analyze: Flash/RAM/MACC 크기 확인
#   validate: eval_stedgeai_host.py --mode host (stm32h7)로 INT8 MinP 측정

set -uo pipefail

OUTROOT="results/gru_phase8_flash"
SUMMARY="$OUTROOT/summary.log"
STEDGE_PY="/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python"
STEDGEAI="/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/stedgeai"
mkdir -p "$OUTROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

# main venv GPU LD_LIBRARY_PATH — all nvidia subdirs included
_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    _NVIDIA_LIBS=$(find "${_SITE}/nvidia" -maxdepth 2 -name "lib" -type d 2>/dev/null | tr '\n' ':')
    export LD_LIBRARY_PATH="${_NVIDIA_LIBS}/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ── 학습 함수 ──────────────────────────────────────────────────────────────
run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id — already complete"; return 0
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

# ── STedgeAI analyze+validate 함수 ────────────────────────────────────────
run_stedgeai() {
    local exp_dir="$1"
    local id; id=$(basename "$exp_dir")

    if [[ ! -f "$exp_dir/model.keras" ]]; then
        log "STEDGEAI SKIP $id — no model.keras"; return
    fi
    if python3 -c "
import json,sys
m=json.load(open('$exp_dir/metrics.json'))
sys.exit(0 if m.get('stedgeai',{}).get('analyze',{}).get('analyze_ok') else 1)
" 2>/dev/null; then
        log "STEDGEAI SKIP $id — already analyzed"; return
    fi

    log "STEDGEAI $id"
    "$STEDGE_PY" scripts/util/export_stedgeai.py \
        --exp-dir "$exp_dir" \
        --target stm32n6 \
        2>&1 | grep -E "^\[stedgeai\]|weights|activations|macc|error|analyze_ok" \
             | tee -a "$SUMMARY" || log "STEDGEAI WARN $id"
}

# ─────────────────────────────────────────────────────────────────────────────

log "=== Phase 8: Conv-optimized GRU (uv/TF 2.21) ==="

# 공통 베이스 (focal, filtered, GRU(128,64), 30f)
BASE=(
    --model-type gru
    --gru-units 128,64
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --early-stop-patience 15 --epochs 100
    --min-val-precision 0.90
    --target-steps 30 --window-start-sec 3.0 --window-end-sec 9.0
)

log "=== Part A: 학습 ==="

# P8-v01: Conv 1층 (64 filters) + kp7 → Flash ≈ 459 KiB
run_exp P8-v01 "${BASE[@]}" \
    --conv-pre-layers 1 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp7

# P8-v02: Conv 2층 (32 filters) + kp7 → Flash ≈ 415 KiB
run_exp P8-v02 "${BASE[@]}" \
    --conv-pre-layers 2 --conv-pre-filters 32 --conv-pre-kernel 5 \
    --feature-set kp7

# P8-v03: Conv 1층 (64 filters) + minimal → Flash ≈ 452 KiB
run_exp P8-v03 "${BASE[@]}" \
    --conv-pre-layers 1 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set minimal

log "=== Part A complete ==="

# ── Part B: STedgeAI analyze ─────────────────────────────────────────────
log "=== Part B: STedgeAI analyze ==="
for id in P8-v01 P8-v02 P8-v03; do
    if [[ -f "$OUTROOT/$id/model.keras" ]]; then
        run_stedgeai "$OUTROOT/$id"
    else
        log "STEDGEAI SKIP $id — training failed"
    fi
done
log "=== Part B complete ==="

# ── 최종 요약 ────────────────────────────────────────────────────────────
log "=== Phase 8 Final Summary ==="
python3 - <<'PYEOF' 2>/dev/null | tee -a "$SUMMARY"
import json
from pathlib import Path

def row(exp: Path, cfg: str):
    if not (exp/"metrics.json").exists():
        print(f"{exp.name:<8} {cfg:<22}  NOT DONE")
        return
    m = json.loads((exp/"metrics.json").read_text())
    tv = m.get("metrics",{}).get("test_video",{})
    ti = m.get("metrics",{}).get("test_int8",{})
    sa = m.get("stedgeai",{}).get("analyze",{})
    fminp = tv.get("min_precision", 0)
    iminp = ti.get("min_precision") if ti else None
    flash = sa.get("weights_kib")
    ram   = sa.get("activations_kib")
    ok    = sa.get("analyze_ok", False)
    deploy = ("✓" if flash and flash <= 512 else "✗") if ok else "?"
    istr  = f"{iminp:.4f}" if iminp else "  -   "
    fstr  = f"{flash:.0f}" if flash else "  ?  "
    rstr  = f"{ram:.0f}"   if ram   else "  ?  "
    print(f"{exp.name:<8} {cfg:<22} {fminp:.4f}  {istr}  {fstr:>6} KiB  {rstr:>5} KiB  {deploy}")

hdr = f"{'ID':<8} {'Config':<22} {'Float MinP':<9} {'INT8 MinP':<8} {'Flash':>8}       {'RAM':>6}   {'OK?'}"
sep = "-" * 72
print(f"\n[Phase 8 — Conv-optimized GRU(128,64)]")
print(hdr); print(sep)

root = Path("results/gru_phase8_flash")
cfgs = {
    "P8-v01": "1×Conv64 kp7 30f",
    "P8-v02": "2×Conv32 kp7 30f",
    "P8-v03": "1×Conv64 minimal 30f",
}
for id_, cfg in cfgs.items():
    row(root / id_, cfg)

print("\n  ✓ = Flash ≤ 512 KiB  ✗ = over  ? = analyze not run")
print("\n[참고: Phase 7 best]")
ref = [
    ("results/gru_phase7_quant", "Q7-v02", "2×Conv64 kp7 30f (537 KiB est)"),
    ("results/gru_phase5_compact", "P5-v02", "2×Conv64 kp7 40f (537 KiB actual)"),
]
for r, id_, cfg in ref:
    exp = Path(r) / id_
    if (exp/"metrics.json").exists():
        m = json.loads((exp/"metrics.json").read_text())
        tv = m.get("metrics",{}).get("test_video",{})
        sa = m.get("stedgeai",{}).get("analyze",{})
        fminp = tv.get("min_precision",0)
        flash = sa.get("weights_kib")
        fstr = f"{flash:.0f}" if flash else "~537"
        print(f"  {id_:<8} {cfg:<36} MinP={fminp:.4f}  Flash={fstr} KiB")
PYEOF

log "Done. See $SUMMARY"
