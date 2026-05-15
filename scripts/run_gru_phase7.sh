#!/usr/bin/env bash
# Phase 7 — STM32 포팅 가능성 검증
#
# 목표: unidirectional GRU, STedgeAI INT8, Flash < 512 KB, MinP ≥ 0.93
#
# ── Part A: 기존 완료 uni 모델 STedgeAI analyze ─────────────────────────────
#   대상 (MinP ≥ 0.93, unidirectional):
#     P5-v02  GRU(128,64)  kp7(27f)    40f  focal  MinP=0.9495
#     P4-v05  GRU(256,128) minimal(21f) 40f        MinP=0.9513
#     P4-v02  GRU(256,128) kp12(45f)   40f  focal  MinP=0.9469
#     P4-v03  GRU(256,128) kp7(27f)    40f         MinP=0.9468
#     P4-v04  GRU(256,128) kp7(27f)    40f  focal  MinP=0.9459
#     P3-v07  GRU(256,128) kp12(45f)   30f  focal  MinP=0.9404
#     P5-v01  GRU(128,64)  kp7(27f)    40f         MinP=0.9353
#
# ── Part B: 신규 학습 실험 ─────────────────────────────────────────────────
#   GRU(128,64) + focal + filtered, 미탐색 조합
#   Q7-v01: kp12, 40f  — kp7 대비 keypoint 정보 증가
#   Q7-v02: kp7,  30f  — 윈도우 단축 경량화
#   Q7-v03: kp12, 30f  — kp12 + 짧은 윈도우
#
# ── Part C: STedgeAI analyze (Part B 신규 모델) ───────────────────────────
#   STedgeAI 내부 Python으로 compat .keras 생성 → analyze → 임시파일 삭제

set -uo pipefail

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OUTROOT="results/gru_phase7_quant"
SUMMARY="$OUTROOT/summary.log"
STEDGE_PY="/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python"
STEDGEAI="/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/stedgeai"
mkdir -p "$OUTROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

# ── STedgeAI analyze 함수 ─────────────────────────────────────────────────
# 결과를 metrics.json의 "stedgeai" 필드에 기록
run_stedgeai_analyze() {
    local exp_dir="$1"
    local id
    id=$(basename "$exp_dir")

    if [[ ! -f "$exp_dir/model.keras" ]]; then
        log "STEDGEAI SKIP $id — model.keras not found"; return
    fi

    # 이미 분석됐으면 스킵
    if python3 -c "
import json,sys
m=json.load(open('$exp_dir/metrics.json'))
sys.exit(0 if m.get('stedgeai',{}).get('analyze',{}).get('analyze_ok') else 1)
" 2>/dev/null; then
        log "STEDGEAI SKIP $id — already analyzed"
        return
    fi

    log "STEDGEAI ANALYZE $id"
    "$STEDGE_PY" scripts/util/export_stedgeai.py \
        --exp-dir "$exp_dir" \
        --target stm32n6 \
        2>&1 | grep -E "^\[stedgeai\]|weights|activations|FLASH|RAM|macc|error" \
             | tee -a "$SUMMARY" || log "STEDGEAI WARN $id — see log"
}

# ── Part A: 기존 best 모델 STedgeAI analyze ──────────────────────────────
log "=== Part A: STedgeAI analyze (existing best models) ==="

run_stedgeai_analyze "results/gru_phase5_compact/P5-v02"
run_stedgeai_analyze "results/gru_phase5_compact/P5-v01"
run_stedgeai_analyze "results/gru_phase4_uni_kp/P4-v05"
run_stedgeai_analyze "results/gru_phase4_uni_kp/P4-v02"
run_stedgeai_analyze "results/gru_phase4_uni_kp/P4-v03"
run_stedgeai_analyze "results/gru_phase4_uni_kp/P4-v04"
run_stedgeai_analyze "results/gru_phase3_2s/P3-v07"

log "=== Part A complete ==="

# Part A 중간 요약
log "--- Part A STedgeAI size summary ---"
python3 - <<'PYEOF' 2>/dev/null | tee -a "$SUMMARY"
import json
from pathlib import Path

targets = [
    ("results/gru_phase5_compact", "P5-v02", "128,64",  "kp7",     40, 0.9495),
    ("results/gru_phase5_compact", "P5-v01", "128,64",  "kp7",     40, 0.9353),
    ("results/gru_phase4_uni_kp",  "P4-v05", "256,128", "minimal", 40, 0.9513),
    ("results/gru_phase4_uni_kp",  "P4-v02", "256,128", "kp12",    40, 0.9469),
    ("results/gru_phase4_uni_kp",  "P4-v03", "256,128", "kp7",     40, 0.9468),
    ("results/gru_phase4_uni_kp",  "P4-v04", "256,128", "kp7",     40, 0.9459),
    ("results/gru_phase3_2s",      "P3-v07", "256,128", "kp12",    30, 0.9404),
]
print(f"\n{'ID':<8} {'Units':<10} {'FS':<8} {'W':<4} {'Float_MinP':<11} {'Flash_KiB':<11} {'RAM_KiB':<9} {'Deploy?'}")
print("-" * 72)
for root, id_, units, fs, w, fminp in targets:
    exp = Path(root) / id_
    m = json.loads((exp / "metrics.json").read_text()) if (exp / "metrics.json").exists() else {}
    sa = m.get("stedgeai", {}).get("analyze", {})
    flash = sa.get("weights_kib", 0)
    ram   = sa.get("activations_kib", 0)
    ok    = sa.get("analyze_ok", False)
    if not ok:
        deploy = "? ERR"
    elif flash <= 512:
        deploy = "✓ OK"
    else:
        deploy = "✗ OVER"
    fstr = f"{flash:.1f}" if flash else "  -  "
    rstr = f"{ram:.1f}"   if ram   else "  -  "
    print(f"{id_:<8} {units:<10} {fs:<8} {w:<4} {fminp:.4f}       {fstr:>7} KiB  {rstr:>5} KiB  {deploy}")
PYEOF

echo ""

# ── Part B: 신규 학습 ────────────────────────────────────────────────────
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

BASE=(
    --model-type gru
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
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
    --gru-units 128,64
)

WIN40=(--target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0)
WIN30=(--target-steps 30 --window-start-sec 3.0 --window-end-sec 9.0)

log "=== Part B: 신규 학습 ==="

run_exp Q7-v01 "${BASE[@]}" "${WIN40[@]}" --feature-set kp12
run_exp Q7-v02 "${BASE[@]}" "${WIN30[@]}" --feature-set kp7
run_exp Q7-v03 "${BASE[@]}" "${WIN30[@]}" --feature-set kp12

log "=== Part B complete ==="

# ── Part C: 신규 모델 STedgeAI analyze ──────────────────────────────────
log "=== Part C: STedgeAI analyze (new models) ==="

for id in Q7-v01 Q7-v02 Q7-v03; do
    if [[ -f "$OUTROOT/$id/model.keras" ]]; then
        run_stedgeai_analyze "$OUTROOT/$id"
    else
        log "STEDGEAI SKIP $id — training failed (no model.keras)"
    fi
done

log "=== Part C complete ==="

# ── 최종 결과 요약 ──────────────────────────────────────────────────────
log "=== Phase 7 Final Summary ==="
python3 - <<'PYEOF' 2>/dev/null | tee -a "$SUMMARY"
import json
from pathlib import Path

def row(exp: Path, units: str, fs: str, w: int):
    m = json.loads((exp / "metrics.json").read_text()) if (exp / "metrics.json").exists() else {}
    tv  = m.get("metrics", {}).get("test_video", {})
    ti  = m.get("metrics", {}).get("test_int8", {})
    sa  = m.get("stedgeai", {}).get("analyze", {})
    fminp  = tv.get("min_precision", 0)
    iminp  = ti.get("min_precision") if ti else None
    flash  = sa.get("weights_kib", 0)
    ram    = sa.get("activations_kib", 0)
    ok     = sa.get("analyze_ok", False)
    deploy = ("✓" if flash <= 512 else "✗") if ok else "?"
    istr   = f"{iminp:.4f}" if iminp else "  -   "
    fstr   = f"{flash:.0f}" if flash else "  -  "
    rstr   = f"{ram:.0f}"   if ram   else "  -  "
    return fminp, istr, fstr, rstr, deploy

hdr = f"{'ID':<8} {'Units':<10} {'FS':<8} {'W':<4} {'Float_MinP':<11} {'TFLite_MinP':<12} {'Flash_KiB':<11} {'RAM_KiB':<9} {'Deploy?'}"
sep = "-" * 80

print("\n[Part A — Existing best unidirectional models]")
print(hdr); print(sep)
for root, id_, units, fs, w in [
    ("results/gru_phase5_compact", "P5-v02", "128,64",  "kp7",     40),
    ("results/gru_phase5_compact", "P5-v01", "128,64",  "kp7",     40),
    ("results/gru_phase4_uni_kp",  "P4-v05", "256,128", "minimal", 40),
    ("results/gru_phase4_uni_kp",  "P4-v02", "256,128", "kp12",    40),
    ("results/gru_phase4_uni_kp",  "P4-v03", "256,128", "kp7",     40),
    ("results/gru_phase4_uni_kp",  "P4-v04", "256,128", "kp7",     40),
    ("results/gru_phase3_2s",      "P3-v07", "256,128", "kp12",    30),
]:
    fminp, istr, fstr, rstr, deploy = row(Path(root) / id_, units, fs, w)
    print(f"{id_:<8} {units:<10} {fs:<8} {w:<4} {fminp:.4f}       {istr:<12} {fstr:>7} KiB  {rstr:>5} KiB  {deploy}")

print("\n[Part B+C — New Phase 7 experiments]")
print(hdr); print(sep)
for id_ in ["Q7-v01", "Q7-v02", "Q7-v03"]:
    exp = Path("results/gru_phase7_quant") / id_
    if not (exp / "metrics.json").exists():
        print(f"{id_:<8}  NOT DONE"); continue
    cfg = json.loads((exp / "run_config.resolved.json").read_text()) if (exp / "run_config.resolved.json").exists() else {}
    fs  = cfg.get("feature_set", "?")
    w   = cfg.get("target_steps", "?")
    fminp, istr, fstr, rstr, deploy = row(exp, "128,64", str(fs), int(w) if str(w).isdigit() else 0)
    print(f"{id_:<8} 128,64     {str(fs):<8} {str(w):<4} {fminp:.4f}       {istr:<12} {fstr:>7} KiB  {rstr:>5} KiB  {deploy}")

print("\n  ✓ = Flash ≤ 512 KiB (STedgeAI INT8, stm32n6 target)")
print("  ✗ = Flash > 512 KiB — too large for STM32N6")
print("  ? = STedgeAI analyze not run")
PYEOF

log "Done. See $SUMMARY"
