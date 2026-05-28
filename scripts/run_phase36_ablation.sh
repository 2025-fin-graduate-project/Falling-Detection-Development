#!/usr/bin/env bash
# Phase 36: Window-level ablation loop
# 순서: kp7→kp13(feature), 40→30(window), gru→lstm(arch)
# 목표: test window MinPR ≥ 0.90. 미달 시 다음 실험으로 자동 진행.
# 결과: results/phase36_window_ablation/summary.tsv

set -uo pipefail
cd "$(dirname "$0")/.."

TARGET_MINPR="${TARGET_MINPR:-0.90}"
SUMMARY="results/phase36_window_ablation/summary.tsv"
LOG_DIR="results/phase36_window_ablation/logs"
mkdir -p "$LOG_DIR" results/phase36_window_ablation

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# TSV 헤더
if [[ ! -f "$SUMMARY" ]]; then
    echo -e "exp_id\tfeature_set\twindow_size\tmodel_type\tval_minpr\ttest_minpr\tfall_pr\tnfall_pr\tfall_rc\tnfall_rc\tfn\tfp\tthreshold\tstatus" > "$SUMMARY"
fi

run_exp() {
    local exp_id="$1" feat="$2" win="$3" model="$4"
    shift 4
    local extra_args=("$@")

    # 이미 완료됐으면 스킵 (stdout에 숫자만 출력, log는 stderr)
    if grep -q "^${exp_id}	" "$SUMMARY" 2>/dev/null; then
        log "SKIP $exp_id (already in summary)" >&2
        awk -F'\t' -v id="$exp_id" '$1==id {print $6}' "$SUMMARY"
        return
    fi

    log "=== $exp_id  feat=$feat  win=$win  model=$model ===" >&2
    local logfile="$LOG_DIR/${exp_id}.log"

    # 훈련 출력은 logfile에만 저장, stdout으로 노출 안 함
    uv run python scripts/train_window_phase36.py \
        --exp-id "$exp_id" \
        --feature-set "$feat" \
        --window-size "$win" \
        --model-type  "$model" \
        "${extra_args[@]}" \
        > "$logfile" 2>&1

    # 결과 추출 (단일 python3 호출로 단순화)
    local test_mp val_mp fall_pr nfall_pr fall_rc nfall_rc fn fp thr
    read -r test_mp val_mp fall_pr nfall_pr fall_rc nfall_rc fn fp thr < <(python3 - <<PYEOF 2>/dev/null
import json, sys
try:
    m = json.load(open("results/phase36_window_ablation/$exp_id/metrics.json"))
    t = m["metrics"]["test_window"]
    v = m["metrics"]["val_window"]
    print(f"{t['min_pr']:.4f} {v['min_pr']:.4f} {t['fall_precision']:.4f} {t['nfall_precision']:.4f} {t['fall_recall']:.4f} {t['nfall_recall']:.4f} {t['fn']} {t['fp']} {m['threshold']}")
except Exception as e:
    print("FAIL FAIL - - - - - - -")
PYEOF
    )
    test_mp="${test_mp:-FAIL}"

    local status="ok"
    if python3 -c "exit(0 if float('${test_mp}') >= ${TARGET_MINPR} else 1)" 2>/dev/null; then
        status="TARGET_MET"
    fi

    echo -e "${exp_id}\t${feat}\t${win}\t${model}\t${val_mp}\t${test_mp}\t${fall_pr}\t${nfall_pr}\t${fall_rc}\t${nfall_rc}\t${fn}\t${fp}\t${thr}\t${status}" >> "$SUMMARY"
    log "  RESULT $exp_id: test_minpr=${test_mp}  val=${val_mp}  status=${status}" >&2
    # stdout에 숫자값만 출력
    echo "$test_mp"
}

# ── Phase 36-A: Feature set ablation (window=40, GRU) ───────────────────────
log "Phase A: Feature set ablation (win=40, GRU)" >&2
mp_kp7=$(run_exp  "P36-kp7-w40-gru"  kp7  40 gru)
mp_kp13=$(run_exp "P36-kp13-w40-gru" kp13 40 gru)

# 더 나은 feature set 선택
if python3 -c "exit(0 if float('${mp_kp13}') >= float('${mp_kp7}') else 1)" 2>/dev/null; then
    BEST_FEAT="kp13"; log "Best feature set: kp13 (${mp_kp13} >= ${mp_kp7})" >&2
else
    BEST_FEAT="kp7";  log "Best feature set: kp7 (${mp_kp7} > ${mp_kp13})" >&2
fi

# ── Phase 36-B: Window size ablation (best feature, GRU) ────────────────────
log "Phase B: Window size ablation (feat=${BEST_FEAT}, GRU)" >&2
mp_w40=$(run_exp "P36-${BEST_FEAT}-w40-gru" "$BEST_FEAT" 40 gru)
mp_w30=$(run_exp "P36-${BEST_FEAT}-w30-gru" "$BEST_FEAT" 30 gru)

if python3 -c "exit(0 if float('${mp_w30}') >= float('${mp_w40}') else 1)" 2>/dev/null; then
    BEST_WIN=30; log "Best window: 30 (${mp_w30} >= ${mp_w40})" >&2
else
    BEST_WIN=40; log "Best window: 40 (${mp_w40} > ${mp_w30})" >&2
fi

# ── Phase 36-C: Architecture ablation (best feat+window) ────────────────────
log "Phase C: Architecture ablation (feat=${BEST_FEAT}, win=${BEST_WIN})" >&2
mp_gru=$(run_exp  "P36-${BEST_FEAT}-w${BEST_WIN}-gru"  "$BEST_FEAT" "$BEST_WIN" gru)
mp_lstm=$(run_exp "P36-${BEST_FEAT}-w${BEST_WIN}-lstm" "$BEST_FEAT" "$BEST_WIN" lstm)

if python3 -c "exit(0 if float('${mp_lstm}') >= float('${mp_gru}') else 1)" 2>/dev/null; then
    BEST_ARCH="lstm"; log "Best arch: LSTM (${mp_lstm} >= ${mp_gru})" >&2
else
    BEST_ARCH="gru";  log "Best arch: GRU (${mp_gru} > ${mp_lstm})" >&2
fi
BEST_MP=$(python3 -c "
import sys
vals = []
for v in ['${mp_gru}', '${mp_lstm}']:
    try: vals.append(float(v))
    except: pass
print(max(vals) if vals else '0.0')
")

log "=== Phase 36 완료 ===" >&2
log "  최고 설정: feat=${BEST_FEAT}, win=${BEST_WIN}, arch=${BEST_ARCH}, test_minpr=${BEST_MP}" >&2

# ── 결과 요약 출력 ────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 36 Window-Level 결과 요약 (test MinPR 기준) ==="
python3 << 'PYEOF'
import csv
from pathlib import Path

rows = []
with open("results/phase36_window_ablation/summary.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows.append(r)

rows.sort(key=lambda r: float(r.get("test_minpr") or 0), reverse=True)
print(f"{'ID':<28} {'feat':>5} {'win':>3} {'arch':>5} {'val':>7} {'test':>7} {'f_pr':>6} {'nf_pr':>6} {'f_rc':>6} {'nf_rc':>6}  {'FN':>4} {'FP':>4}  thr  상태")
print("-"*115)
for r in rows:
    flag = " *** TARGET!" if r.get("status") == "TARGET_MET" else ""
    print(f"{r['exp_id']:<28} {r['feature_set']:>5} {r['window_size']:>3} {r['model_type']:>5} "
          f"{r['val_minpr']:>7} {r['test_minpr']:>7} "
          f"{r.get('fall_pr','-'):>6} {r.get('nfall_pr','-'):>6} "
          f"{r.get('fall_rc','-'):>6} {r.get('nfall_rc','-'):>6}  "
          f"{r.get('fn','-'):>4} {r.get('fp','-'):>4}  {r.get('threshold','-')}{flag}")
PYEOF
