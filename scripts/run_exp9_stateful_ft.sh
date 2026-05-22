#!/usr/bin/env bash
# Experiment 9: Stateless → Stateful Fine-tuning 재검증
#
# Phase 41/42 상위 3개 경량 후보를 stateful fine-tuning 후
# 4-metric min_pr(FallPrec/FallRec/NFallPrec/NFallRec 최솟값) 기준으로 재평가.
#
# 대상 모델:
#   kp7-w40-raw        : results/phase41_ablation/P41-raw-kp7-w40
#   kp13-w60-filtered  : results/phase41_ablation/P41-kp13-w60
#   kp17-w60-filtered  : results/phase42_cross/P42-kp17-w60
#
# 비교 기준선 (Phase 40, GRU(128,64)):
#   P40-nv-a65-stateful: ev_minpr=0.9543 (구형 2-metric)
#
# 출력: results/additional_report_experiments/exp9_stateful_finetune/

set -e
OUT_ROOT="results/additional_report_experiments/exp9_stateful_finetune"
mkdir -p "$OUT_ROOT/logs"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_exp() {
    local exp_id="$1"
    local base_dir="$2"
    local data_dir="$3"
    local window_size="$4"

    if [ -f "${OUT_ROOT}/${exp_id}/metrics.json" ]; then
        log "SKIP $exp_id (already done)"
        return 0
    fi

    local base_model="${base_dir}/model.keras"
    if [ ! -f "$base_model" ]; then
        log "SKIP $exp_id — $base_model not found"
        return 1
    fi

    log "=== START $exp_id ==="
    uv run python scripts/train_stateful_finetune.py \
        --base-model  "$base_model" \
        --base-dir    "$base_dir" \
        --exp-id      "$exp_id" \
        --out-root    "$OUT_ROOT" \
        --train-csv   "${data_dir}/train.csv" \
        --val-csv     "${data_dir}/val.csv" \
        --test-csv    "${data_dir}/test.csv" \
        --window-size "$window_size" \
        --epochs 15 \
        --lr 5e-5 \
        --focal-alpha 0.65 \
        --freeze-conv \
        --pure-window \
        --pure-margin 5 \
        --nfall-stride 5 \
        --seed 42 \
        2>&1 | tee "${OUT_ROOT}/logs/${exp_id}.log"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        log "FAILED $exp_id (exit $exit_code)"
        return 1
    fi
    log "Done: $exp_id"
}

log "=== Experiment 9: Stateless → Stateful Fine-tuning ==="

# ── 1. kp7-w40-raw (최경량, CPU 우선) ────────────────────────────────────────
run_exp \
    "exp9-kp7-w40-raw" \
    "results/phase41_ablation/P41-raw-kp7-w40" \
    "dataset/splits_v2_class_balanced" \
    40

# ── 2. kp13-w60-filtered (포팅 균형) ─────────────────────────────────────────
run_exp \
    "exp9-kp13-w60" \
    "results/phase41_ablation/P41-kp13-w60" \
    "dataset/splits_v2_class_balanced_filtered" \
    60

# ── 3. kp17-w60-filtered (최고 성능) ─────────────────────────────────────────
run_exp \
    "exp9-kp17-w60" \
    "results/phase42_cross/P42-kp17-w60" \
    "dataset/splits_v2_class_balanced_filtered" \
    60

log "=== Exp 9 완료 ==="
python3 -c "
import json, pathlib
out = pathlib.Path('results/additional_report_experiments/exp9_stateful_finetune')
print(f'  {\"ID\":<25} {\"base_val\":>9} {\"test_ev\":>9} {\"FP\":>4} {\"FN\":>4}')
print('  ' + '-'*55)
for mf in sorted(out.glob('*/metrics.json')):
    try:
        m = json.loads(mf.read_text())
        base = m.get('baseline_val_ev', 0)
        test = m.get('metrics', {}).get('test_event_vote', {})
        print(f'  {mf.parent.name:<25} {base:>9.4f} {test.get(\"min_pr\",0):>9.4f} {test.get(\"fp\",0):>4} {test.get(\"fn\",0):>4}')
    except: pass
" 2>/dev/null || true
