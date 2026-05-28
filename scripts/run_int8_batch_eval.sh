#!/usr/bin/env bash
# INT8 배치 평가 파이프라인
# 대상: float EventMinPR ≥ 0.90 + STM32 포팅 가능 모델 (GRU/LSTM, no bidir/attn)
# 순서: STedgeAI analyze → host eval (stm32h7 proxy)
# 결과: results/int8_eval_summary.tsv

set -uo pipefail
cd "$(dirname "$0")/.."

SUMMARY="results/quantization/int8_eval_summary.tsv"
STEDGE_PY="/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python"
EVAL_STRIDE="${EVAL_STRIDE:-10}"   # stride=10 → ~7K windows, ~20min/모델

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── 대상 모델 목록 (float EventMinPR 내림차순) ─────────────────────────────
EXP_DIRS=(
    results/phase27_seed_sweep/P27-vm0
    results/training/phase21_event_minpr_checkpoint/P21-v01
    results/training/phase30_val_loss_chain/P30-lstm
    results/training/phase20_class_balanced_gru/P20-v03
    results/training/phase32_gru_vl_explore/P32-ce-42
    results/training/phase20_class_balanced_gru/P20-v01
    results/training/phase32_gru_vl_explore/P32-fl-1
    results/phase27_seed_sweep/P27-s1
    results/training/phase26_full_patience/P26-v01
    results/training/phase30_val_loss_chain/P30-gru
    results/phase35_ncw/P35-ncw
    results/training/phase34_aug_alpha/P34-a35-fm10
    results/phase27_seed_sweep/P27-s2
    results/training/phase23_gru256_event/P23-v01
    results/phase27_seed_sweep/P27-s0
    results/training/phase28_seed_probe/P28-vm7
    results/training/phase33_ce_vm_explore/P33-gce-vm0
    results/training/phase31_lstm_ckpt/P31-vm42
    results/training/phase19_gru256/P19-v02
    results/training/phase28_seed_probe/P28-vm77
    results/training/phase29_alpha_sweep/P29-a35
    results/training/phase28_seed_probe/P28-vm123
)

mkdir -p results/quantization
# TSV 헤더
if [[ ! -f "$SUMMARY" ]]; then
    echo -e "exp_id\tfloat_event_minpr\tint8_event_minpr\tint8_fall_pr\tint8_nfall_pr\tint8_fn\tint8_fp\tweights_kib\tmacc\tanalyze_ok\tstatus" > "$SUMMARY"
fi

for exp_dir in "${EXP_DIRS[@]}"; do
    exp_id=$(basename "$exp_dir")

    # 이미 평가됐으면 스킵
    if grep -q "^${exp_id}	" "$SUMMARY" 2>/dev/null; then
        log "SKIP $exp_id (already in summary)"
        continue
    fi

    # model.keras 존재 확인
    if [[ ! -f "$exp_dir/model.keras" ]]; then
        log "SKIP $exp_id — model.keras not found"
        echo -e "${exp_id}\t-\t-\t-\t-\t-\t-\t-\t-\t-\tno_model" >> "$SUMMARY"
        continue
    fi

    log "=== $exp_id ==="

    # ── 1. STedgeAI analyze ────────────────────────────────────────────────
    analyze_ok=$(python3 -c "
import json; m=json.load(open('$exp_dir/metrics.json'))
st=m.get('stedgeai',{}).get('analyze',{})
print(st.get('analyze_ok',''))
" 2>/dev/null)

    if [[ "$analyze_ok" != "True" ]]; then
        log "  analyze: running..."
        "$STEDGE_PY" scripts/util/export_stedgeai.py \
            --exp-dir "$exp_dir" --target stm32n6 \
            2>&1 | tail -5
    else
        log "  analyze: already done"
    fi

    # analyze 결과 읽기
    analyze_info=$(python3 -c "
import json; m=json.load(open('$exp_dir/metrics.json'))
st=m.get('stedgeai',{}).get('analyze',{})
print(st.get('analyze_ok','?'), st.get('weights_kib','?'), st.get('macc','?'))
" 2>/dev/null)
    read -r a_ok a_wkib a_macc <<< "$analyze_info"

    # ── 2. STedgeAI host eval ─────────────────────────────────────────────
    log "  host eval: stride=${EVAL_STRIDE}..."
    eval_out=$(uv run python scripts/util/eval_stedgeai_host.py \
        --exp-dir "$exp_dir" \
        --eval-stride "$EVAL_STRIDE" \
        --reselect-threshold \
        2>&1)
    eval_rc=$?

    # float MinPR
    float_ev=$(python3 -c "
import json; m=json.load(open('$exp_dir/metrics.json'))
te=m.get('metrics',{}).get('test_event_video',{})
print(f\"{te.get('min_pr',0):.4f}\")
" 2>/dev/null)

    # INT8 결과
    int8_info=$(python3 -c "
import json; m=json.load(open('$exp_dir/metrics.json'))
he=m.get('stedgeai_host_eval',{})
if not he.get('eval_ok'):
    print('FAIL - - - - -')
else:
    mp=he.get('min_precision',0)
    fp_pr=he.get('fall_precision',0)
    nfp_pr=he.get('nfall_precision',0)
    cm=he.get('confusion_matrix')
    fn=cm[1][0] if cm else '?'
    fp_n=cm[0][1] if cm else '?'
    print(f'{mp:.4f} {fp_pr:.4f} {nfp_pr:.4f} {fn} {fp_n}')
" 2>/dev/null)
    read -r i_mp i_fpr i_nfpr i_fn i_fp <<< "$int8_info"

    if [[ "$i_mp" == "FAIL" ]]; then
        log "  host eval FAILED"
        echo -e "${exp_id}\t${float_ev}\tFAIL\t-\t-\t-\t-\t${a_wkib}\t${a_macc}\t${a_ok}\tfail" >> "$SUMMARY"
    else
        log "  float=${float_ev}  INT8=${i_mp}  FN=${i_fn} FP=${i_fp}"
        echo -e "${exp_id}\t${float_ev}\t${i_mp}\t${i_fpr}\t${i_nfpr}\t${i_fn}\t${i_fp}\t${a_wkib}\t${a_macc}\t${a_ok}\tok" >> "$SUMMARY"
    fi
done

log "=== 배치 완료 ==="
echo ""
echo "=== INT8 평가 요약 (float MinPR 기준 정렬) ==="
python3 << 'PYEOF'
import csv, sys
from pathlib import Path

rows = []
with open("results/int8_eval_summary.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)

rows.sort(key=lambda r: float(r.get("float_event_minpr") or 0), reverse=True)
print(f"{'ID':<25} {'Float':>7} {'INT8':>7} {'FN':>4} {'FP':>4}  {'Flash(KiB)':>10}  {'MACC':>9}  상태")
print("-" * 85)
for r in rows:
    flag = " *** ≥0.90!" if r.get("int8_event_minpr","") not in ("FAIL","-","") and float(r.get("int8_event_minpr",0) or 0) >= 0.90 else ""
    print(f"{r['exp_id']:<25} {r['float_event_minpr']:>7} {r.get('int8_event_minpr','-'):>7} "
          f"{r.get('int8_fn','-'):>4} {r.get('int8_fp','-'):>4}  "
          f"{r.get('weights_kib','-'):>10}  {r.get('macc','-'):>9}  {r.get('status','-')}{flag}")
PYEOF
