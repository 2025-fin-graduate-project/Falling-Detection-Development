#!/usr/bin/env bash
# 상위 3 모델 재훈련 + STedgeAI generate (포팅용)
#
# 모델:
#   M1: P41-kp13-w60   — kp13, w=60, filtered, GRU(64,32)  [INT8 MinP 0.9298]
#   M2: P41-vel-kp13   — kp13+vel, w=40, filtered           [INT8 MinP 0.9211]
#   M3: P41-kp17-w40   — kp17, w=40, filtered               [INT8 MinP 0.9167]
#
# 결과: results/top3_retrain/{M1,M2,M3}/
# 포팅: results/top3_retrain/{M1,M2,M3}/generate/

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

STEDGEAI="/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/stedgeai"
OUT="$REPO/results/top3_retrain"
FILT="$REPO/dataset/splits_v2_filtered"
LOG="$OUT/run.log"
mkdir -p "$OUT"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# GPU 환경
_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

COMMON="--model-type gru --hidden-sizes 64 32
  --epochs 100 --batch-size 512 --lr 1e-3 --dropout 0.3
  --focal-alpha 0.65 --focal-gamma 2.0
  --fall-stride 1 --nfall-stride 5 --seed 42
  --pure-window --pure-margin 5
  --out-root $OUT"

run_exp() {
    local id="$1"; shift
    local exp_dir="$OUT/$id"
    if [[ -f "$exp_dir/metrics.json" ]]; then
        log "SKIP $id (already done)"
        return 0
    fi
    log "TRAIN → $id"
    uv run python -u "$REPO/scripts/train_window_phase37.py" \
        --exp-id "$id" $COMMON "$@" \
        2>&1 | tee "$OUT/${id}.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { log "FAIL $id"; return 1; }
    log "DONE $id"
}

analyze_and_generate() {
    local id="$1"
    local exp_dir="$OUT/$id"
    local model="$exp_dir/model_best.keras"
    [[ -f "$model" ]] || model="$exp_dir/model.keras"
    [[ -f "$model" ]] || { log "WARN: no model.keras for $id"; return; }

    # STedgeAI analyze (Flash/MACC 추출)
    log "ANALYZE → $id"
    /home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python \
        "$REPO/scripts/util/export_stedgeai.py" \
        --exp-dir "$exp_dir" --target stm32n6 2>&1 | tee -a "$LOG" || true

    # 1) Conv submodel + stateful GRU 분리 (STedgeAI Python / Keras 3.7)
    log "SPLIT → $id"
    /home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python \
        "$REPO/scripts/util/split_stateful_k37.py" \
        --exp-dir "$exp_dir" 2>&1 | tee -a "$LOG" || true

    # 2) Stateful GRU → ONNX → STedgeAI generate --type onnx
    log "GRU ONNX GENERATE → $id"
    uv run python "$REPO/scripts/util/export_gru_onnx.py" \
        --exp-dir "$exp_dir" --skip-split \
        2>&1 | tee -a "$LOG" || true

    # 3) Conv submodel full model generate (fallback / 비교용)
    local gen_dir="$exp_dir/generate"
    mkdir -p "$gen_dir"
    log "FULL MODEL GENERATE → $id"
    "$STEDGEAI" generate \
        --target stm32n6 \
        --model "$model" \
        --type keras \
        --name "${id//-/_}" \
        --output "$gen_dir" \
        --workspace "$gen_dir/workspace" \
        --compression lossless \
        --allocate-states \
        --verbosity 1 2>&1 | tee -a "$LOG" || true
}

print_result() {
    local id="$1"
    local exp_dir="$OUT/$id"
    python3 - "$exp_dir" "$id" << 'PYEOF' 2>/dev/null || true
import json, sys
from pathlib import Path
exp_dir = Path(sys.argv[1]); exp_id = sys.argv[2]
mf = exp_dir / "metrics.json"
if not mf.exists(): print(f"  {exp_id}: no metrics.json yet"); sys.exit(0)
m = json.loads(mf.read_text())
ev = m.get("unified_eval",{}).get("event",{})
ev_mp = ev.get("min_pr","?")
thr = m.get("threshold","?")
st = m.get("stedgeai",{}).get("analyze",{})
print(f"  {exp_id}: ev_minpr={ev_mp} thr={thr}  Flash={st.get('weights_kib','?')}KiB MACC={st.get('macc','?')}")
PYEOF
}

# ══════════════════════════════════════════════════════════════════
log "=== Top-3 모델 재훈련 시작 ==="

# M1: kp13, w=60 (최고 INT8 성능)
run_exp "M1-kp13-w60" \
    --feature-set kp13 --window-size 60 \
    --data-dir "$FILT"
print_result "M1-kp13-w60"

# M2: kp13 + velocity, w=40
run_exp "M2-vel-kp13-w40" \
    --feature-set kp13 --window-size 40 --use-velocity \
    --data-dir "$FILT"
print_result "M2-vel-kp13-w40"

# M3: kp17, w=40
run_exp "M3-kp17-w40" \
    --feature-set kp17 --window-size 40 \
    --data-dir "$FILT"
print_result "M3-kp17-w40"

log "=== 훈련 완료. STedgeAI analyze + generate ==="
for id in M1-kp13-w60 M2-vel-kp13-w40 M3-kp17-w40; do
    analyze_and_generate "$id"
done

log "=== 결과 요약 ==="
for id in M1-kp13-w60 M2-vel-kp13-w40 M3-kp17-w40; do
    print_result "$id"
done

log ""
log "생성된 C 코드 위치:"
for id in M1-kp13-w60 M2-vel-kp13-w40 M3-kp17-w40; do
    ls "$OUT/$id/generate/"*.c 2>/dev/null | head -2 | while read f; do echo "  $f"; done
done
