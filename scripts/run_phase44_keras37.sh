#!/usr/bin/env bash
# Phase 44 — Keras 3.7 재학습 (STedgeAI stateful GRU generate PoC)
#
# 목표: 상위 3 모델을 Keras 3.7로 재학습 → STedgeAI generate 호환성 검증
#   P44-kp13-w60  : kp12 filtered w=60  (INT8 MinP 최고: 0.9298)
#   P44-kp17-w40  : all  filtered w=40  (INT8 MinP: 0.9167)
#   P44-vel-kp13-w40: (vel 피처셋 준비 후 추가)
#
# 순서:
#   1. build_train_npy.py  (uv run python — pandas 필요)
#   2. train_k37.py        (STedgeAI Python = Keras 3.7)
#   3. export_stedgeai.py  (STedgeAI Python — analyze)
#   4. split_stateful_k37.py (STedgeAI Python — stateful generate PoC)
#
# PoC 우선: P44-kp13-w60만 완료되면 generate 테스트 진행.
#
# 사전 조건:
#   dataset/splits_v2_filtered/{train,val,test}.csv  존재

set -uo pipefail
cd "$(dirname "$0")/.."

STEDGE_PY="/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python"
OUTROOT="results/phase44_keras37"
LOG="$OUTROOT/phase44.log"
SEED=42

# ── GPU 환경 변수 (uv 학습 단계용) ──────────────────────────────────────────
_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTROOT" /tmp/phase44_npy

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# ── 데이터셋 확인 ─────────────────────────────────────────────────────────────
check_dataset() {
    local split_dir="$1"
    for f in train.csv val.csv test.csv; do
        if [[ ! -f "$split_dir/$f" ]]; then
            log "ERROR: $split_dir/$f not found — rebuild dataset first"
            log "  → uv run python scripts/util/build_v2_dataset.py"
            log "  → uv run python scripts/util/build_filtered_v2_splits.py"
            exit 1
        fi
    done
}

# ── 데이터 준비 ───────────────────────────────────────────────────────────────
build_npy() {
    local npy_dir="$1"; local feat_set="$2"; local steps="$3"; local preprocessing="$4"
    local add_vel="${5:-}"
    if [[ -f "$npy_dir/data_config.json" ]]; then
        log "SKIP build_npy $npy_dir (already exists)"
        return 0
    fi
    log "BUILD NPY → $npy_dir (feat=$feat_set steps=$steps pp=$preprocessing)"
    local vel_arg=""
    [[ "$add_vel" == "vel" ]] && vel_arg="--add-velocity"
    uv run python scripts/util/build_train_npy.py \
        --train-csv "dataset/splits_v2_filtered/train.csv" \
        --val-csv   "dataset/splits_v2_filtered/val.csv" \
        --test-csv  "dataset/splits_v2_filtered/test.csv" \
        --feature-set "$feat_set" --preprocessing "$preprocessing" \
        --target-steps "$steps" \
        --window-start-sec 3.0 --window-end-sec 9.0 \
        --label-column label \
        --eval-stride 1 --train-negative-stride 2 \
        --output-dir "$npy_dir" \
        $vel_arg \
        2>&1 | tee -a "$LOG"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { log "FAIL build_npy $npy_dir"; exit 1; }
}

# ── 학습 (Keras 3.7) ──────────────────────────────────────────────────────────
train_k37() {
    local npy_dir="$1"; local exp_id="$2"
    local exp_dir="$OUTROOT/$exp_id"
    if [[ -f "$exp_dir/metrics.json" ]]; then
        log "SKIP train $exp_id (metrics.json exists)"
        return 0
    fi
    log "TRAIN K37 → $exp_id"
    "$STEDGE_PY" scripts/util/train_k37.py \
        --npy-dir    "$npy_dir" \
        --output-dir "$exp_dir" \
        --gru-units  64,32 \
        --conv-filters 64 --conv-kernel 5 --conv-layers 2 \
        --focal-alpha 0.25 --focal-gamma 2.0 \
        --dropout-rate 0.3 --noise-std 0.02 \
        --batch-size 512 --epochs 100 --patience 15 \
        --seed "$SEED" \
        2>&1 | tee -a "$LOG"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { log "FAIL train $exp_id"; return 1; }
    log "DONE train $exp_id"
}

# ── STedgeAI analyze ──────────────────────────────────────────────────────────
analyze() {
    local exp_id="$1"
    local exp_dir="$OUTROOT/$exp_id"
    local analyze_ok
    analyze_ok=$(python3 -c "
import json; m=json.load(open('$exp_dir/metrics.json'))
print(m.get('stedgeai',{}).get('analyze',{}).get('analyze_ok',''))
" 2>/dev/null || true)
    if [[ "$analyze_ok" == "True" ]]; then
        log "SKIP analyze $exp_id (already done)"
        return 0
    fi
    log "ANALYZE → $exp_id"
    "$STEDGE_PY" scripts/util/export_stedgeai.py \
        --exp-dir "$exp_dir" --target stm32n6 \
        2>&1 | tee -a "$LOG"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || log "WARN analyze $exp_id failed"
}

# ── Stateful generate PoC ─────────────────────────────────────────────────────
stateful_poc() {
    local exp_id="$1"
    local exp_dir="$OUTROOT/$exp_id"
    local result_file="$exp_dir/submodels/split_generate_result.json"
    if [[ -f "$result_file" ]]; then
        log "SKIP stateful PoC $exp_id (result exists)"
        python3 -c "
import json
r = json.load(open('$result_file'))
print(f'  Conv OK: {r.get(\"conv_generate\",{}).get(\"ok\",\"?\")}')
print(f'  GRU stateful OK: {r.get(\"gru_stateful_generate\",{}).get(\"ok\",\"?\")}')
print(f'  Numerical OK: {r.get(\"numerical_ok\",\"?\")}')
" 2>/dev/null || true
        return 0
    fi
    if [[ ! -f "$exp_dir/model.keras" ]]; then
        log "SKIP stateful PoC $exp_id (model.keras missing — train first)"
        return 0
    fi
    log "STATEFUL POC → $exp_id"
    "$STEDGE_PY" scripts/util/split_stateful_k37.py \
        --exp-dir "$exp_dir" \
        2>&1 | tee -a "$LOG"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || log "WARN stateful PoC $exp_id failed"
}

# ── print result summary ──────────────────────────────────────────────────────
print_result() {
    local exp_id="$1"
    local exp_dir="$OUTROOT/$exp_id"
    python3 - "$exp_dir" "$exp_id" << 'PYEOF' 2>/dev/null || true
import json, sys
from pathlib import Path
exp_dir = Path(sys.argv[1]); exp_id = sys.argv[2]
mf = exp_dir / "metrics.json"
if not mf.exists(): print(f"  {exp_id}: no metrics.json yet"); sys.exit(0)
m = json.loads(mf.read_text())
tv = m.get("metrics",{}).get("test_video",{})
minp = tv.get("min_precision") or tv.get("min_pr", "?")
thr = m.get("threshold_selection",{}).get("threshold","?")
mc  = m.get("threshold_selection",{}).get("min_consecutive","?")
st  = m.get("stedgeai",{}).get("analyze",{})
print(f"  {exp_id}: test_minpr={minp} thr={thr} mc={mc}  Flash={st.get('weights_kib','?')}KiB MACC={st.get('macc','?')}")

rf = exp_dir / "submodels" / "split_generate_result.json"
if rf.exists():
    r = json.loads(rf.read_text())
    conv_ok = r.get("conv_generate",{}).get("ok","?")
    gru_ok  = r.get("gru_stateful_generate",{}).get("ok","?")
    num_ok  = r.get("numerical_ok","?")
    print(f"    PoC: conv={conv_ok} gru_stateful={gru_ok} numerical={num_ok}")
PYEOF
}

# ════════════════════════════════════════════════════════════════════════════
log "=== Phase 44 Keras 3.7 재학습 ==="

check_dataset "dataset/splits_v2_filtered"

# ── P44-kp13-w60 (PoC 우선) ─────────────────────────────────────────────────
NPY_KP13_W60="/tmp/phase44_npy/kp13_w60"
build_npy "$NPY_KP13_W60" kp12 60 filtered
train_k37 "$NPY_KP13_W60" "P44-kp13-w60"
print_result "P44-kp13-w60"
analyze     "P44-kp13-w60"

log "--- PoC: stateful generate test ---"
stateful_poc "P44-kp13-w60"
print_result "P44-kp13-w60"

# ── P44-kp17-w40 ─────────────────────────────────────────────────────────────
NPY_KP17_W40="/tmp/phase44_npy/kp17_w40"
build_npy "$NPY_KP17_W40" all 40 filtered
train_k37 "$NPY_KP17_W40" "P44-kp17-w40"
print_result "P44-kp17-w40"
analyze     "P44-kp17-w40"

# ════════════════════════════════════════════════════════════════════════════
log "=== Phase 44 완료 ==="
log ""
log "결과 요약:"
for id in P44-kp13-w60 P44-kp17-w40; do
    print_result "$id"
done

log ""
log "PoC 목표: P44-kp13-w60 stateful GRU generate"
python3 - "$OUTROOT/P44-kp13-w60" << 'PYEOF' 2>/dev/null || true
import json
from pathlib import Path
rf = Path(sys.argv[1]) / "submodels" / "split_generate_result.json" if False else Path("results/phase44_keras37/P44-kp13-w60/submodels/split_generate_result.json")
if rf.exists():
    r = json.loads(rf.read_text())
    gru_ok = r.get("gru_stateful_generate",{}).get("ok",False)
    print(f">>> GRU stateful generate: {'✅ PASSED' if gru_ok else '❌ FAILED'}")
    if not gru_ok:
        print(r.get("gru_stateful_generate",{}).get("stderr_tail","")[-500:])
else:
    print(">>> PoC result file not found")
PYEOF
