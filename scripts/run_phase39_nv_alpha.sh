#!/usr/bin/env bash
# Phase 39: nv+α0.65 베이스라인 추가 실험
#
# W2-NV 발견: velocity 없이 α=0.65만으로 event_vote 0.9079, FP=21 (신기록)
# → nv+α0.65를 새 베이스로 삼아 FP 21→19 이하 (ev ≥ 0.915) 달성 탐색
#
# 베이스: no-vel, kp13, w40, GRU(128,64), pure-window, margin=5, α=0.65
#
# 실험 순서 (기대 효과 순):
#   A: nv+α0.60       — α 추가 감소 → nfall FP 구분 극대화
#   B: nv+α0.65+w30   — w30이 순수 모델에서 ev=0.9035 달성; nv base에 적용
#   C: nv+α0.65+h256  — 큰 모델로 표현력↑, window MinP 유지하며 FP 억제
#   D: nv+α0.65+w35   — w30/w40 중간값; FN 악화 없이 FP 억제 탐색
#   E: nv+α0.65+seed2 — 재현성 확인 (seed=123)

set -uo pipefail
cd "$(dirname "$0")/.."

SUMMARY="results/phase36_window_ablation/summary_phase37.tsv"
LOG_DIR="results/phase36_window_ablation/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_exp() {
    local exp_id="$1" group="$2" var_desc="$3"
    shift 3
    local train_args=("$@")

    if grep -q "^${exp_id}	" "$SUMMARY" 2>/dev/null; then
        log "SKIP $exp_id (already done)"
        awk -F'\t' -v id="$exp_id" '$1==id {print $5}' "$SUMMARY"
        return 0
    fi

    log "=== START $exp_id  [$group: $var_desc] ==="
    local logfile="$LOG_DIR/${exp_id}.log"

    uv run python scripts/train_window_phase37.py \
        --exp-id "$exp_id" \
        "${train_args[@]}" \
        > "$logfile" 2>&1

    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "FAILED $exp_id (exit $rc) — check $logfile"
        printf "%s\t%s\t%s\tFAIL\tFAIL\t-\t-\t-\t-\t-\t-\t-\t-\t-\tfail\n" \
            "$exp_id" "$group" "$var_desc" >> "$SUMMARY"
        echo "FAIL"
        return 1
    fi

    local metrics
    metrics=$(python3 - << PYEOF 2>/dev/null
import json, sys
try:
    m = json.load(open("results/phase36_window_ablation/$exp_id/metrics.json"))
    t  = m["metrics"]["test_window"]
    v  = m["metrics"]["val_window"]
    ev = m["metrics"].get("test_event_vote", {})
    er = m["metrics"].get("test_event_raw",  {})
    ws = m.get("window_stats", {})
    evv = ev.get("min_pr","?")
    evr = er.get("min_pr","?")
    fp_v = ev.get("fp","?"); fn_v = ev.get("fn","?")
    print(
        f"{v['min_pr']:.4f}\t{t['min_pr']:.4f}\t"
        f"{t['fall_precision']:.4f}\t{t['fall_recall']:.4f}\t"
        f"{t['nfall_precision']:.4f}\t{t['nfall_recall']:.4f}\t"
        f"{t['fn']}\t{t['fp']}\t{m['threshold']}\t"
        f"{ws.get('train_fall','?')}\t{ws.get('train_nfall','?')}"
    )
    print(f"  → event_vote={evv}  event_raw={evr}  FP_vid={fp_v}  FN_vid={fn_v}", file=sys.stderr)
except Exception as e:
    print(f"# error: {e}", file=sys.stderr)
    print("FAIL\tFAIL\t-\t-\t-\t-\t-\t-\t-\t-\t-")
PYEOF
    )

    local test_mp
    test_mp=$(echo "$metrics" | cut -f2)

    printf "%s\t%s\t%s\t%s\tok\n" \
        "$exp_id" "$group" "$var_desc" "$metrics" >> "$SUMMARY"

    local ev_vote ev_raw fp_vid fn_vid
    ev_vote=$(python3 -c "import json; m=json.load(open('results/phase36_window_ablation/$exp_id/metrics.json')); print(m['metrics'].get('test_event_vote',{}).get('min_pr','?'))" 2>/dev/null)
    ev_raw=$(python3 -c  "import json; m=json.load(open('results/phase36_window_ablation/$exp_id/metrics.json')); print(m['metrics'].get('test_event_raw',{}).get('min_pr','?'))"  2>/dev/null)
    fp_vid=$(python3 -c  "import json; m=json.load(open('results/phase36_window_ablation/$exp_id/metrics.json')); print(m['metrics'].get('test_event_vote',{}).get('fp','?'))"     2>/dev/null)
    fn_vid=$(python3 -c  "import json; m=json.load(open('results/phase36_window_ablation/$exp_id/metrics.json')); print(m['metrics'].get('test_event_vote',{}).get('fn','?'))"     2>/dev/null)

    log "  RESULT $exp_id: window=${test_mp}  event_vote=${ev_vote}  event_raw=${ev_raw}  FP_vid=${fp_vid}  FN_vid=${fn_vid}"

    if python3 -c "
t=float('${test_mp}'.replace('FAIL','0') or 0)
ev=float('${ev_vote}'.replace('?','0') or 0)
assert t >= 0.93 and ev >= 0.915, 'not yet'
" 2>/dev/null; then
        log "  ✅ 목표 달성! window(${test_mp}) >= 0.93 AND event_vote(${ev_vote}) >= 0.915"
    fi

    echo "$test_mp"
}

log "=== Phase 39: nv+α0.65 베이스 실험 시작 ==="
log "    베이스 (W2-NV): win=0.9345  ev=0.9079  FP=21  FN=8"
log "    목표: ev >= 0.915 (FP ≤ 19)"

# ── H39-A: nv + α0.60 ────────────────────────────────────────────────────────
# α 추가 감소 (0.65→0.60): nfall FP 구분 극대화
# risk: fall_recall 하락, FN↑
run_exp "P39-nv-a60" "H39_A" "nv+α=0.60" \
    --feature-set kp13 --window-size 40 --model-type gru \
    --pure-window --pure-margin 5 \
    --focal-alpha 0.60

# ── H39-B: nv + α0.65 + w30 ─────────────────────────────────────────────────
# w30이 순수 모델에서 ev=0.9035 달성(FP=22); nv+α0.65 base에 적용
# 기대: α0.65 nfall 구분 + w30 FP 억제 = FP ≤ 19
# risk: FN↑ (w30 단독 FN=14 전례)
run_exp "P39-nv-a65-w30" "H39_B" "nv+α=0.65+w30" \
    --feature-set kp13 --window-size 30 --model-type gru \
    --pure-window --pure-margin 5 \
    --focal-alpha 0.65

# ── H39-C: nv + α0.65 + h256 ─────────────────────────────────────────────────
# 큰 모델 (128,64→256,128): window MinP 유지 + 표현력으로 FP 추가 억제
# h256 단독: event_vote=0.8904; nv+α0.65 base에 추가 시 시너지 기대
run_exp "P39-nv-a65-h256" "H39_C" "nv+α=0.65+h256" \
    --feature-set kp13 --window-size 40 --model-type gru \
    --pure-window --pure-margin 5 \
    --focal-alpha 0.65 \
    --hidden-sizes 256 128

# ── H39-D: nv + α0.65 + w35 ──────────────────────────────────────────────────
# 중간 윈도우 (40→35): w30의 FN 악화 없이 FP 억제 탐색
# vel+w35 결과와 비교: nv base에서 w35가 더 효과적인지 확인
run_exp "P39-nv-a65-w35" "H39_D" "nv+α=0.65+w35" \
    --feature-set kp13 --window-size 35 --model-type gru \
    --pure-window --pure-margin 5 \
    --focal-alpha 0.65

# ── H39-E: nv + α0.65 + seed2 ────────────────────────────────────────────────
# 베이스라인 재현성 확인 (seed=123)
# W2-NV가 운좋은 초기화인지 체크
run_exp "P39-nv-a65-seed2" "H39_E" "nv+α=0.65+seed=123" \
    --feature-set kp13 --window-size 40 --model-type gru \
    --pure-window --pure-margin 5 \
    --focal-alpha 0.65 \
    --seed 123

# ── 최종 요약 ─────────────────────────────────────────────────────────────────
log "=== Phase 39 완료 ==="
python3 - << 'PYEOF'
import json
from pathlib import Path

baseline = "P38-nv-a65-gru"
p39_exps = [
    "P39-nv-a60",
    "P39-nv-a65-w30",
    "P39-nv-a65-h256",
    "P39-nv-a65-w35",
    "P39-nv-a65-seed2",
]

print(f"\n{'=== Phase 39 결과 요약 ==='}")
print(f"{'ID':<28} {'win':>7} {'ev_vote':>8} {'ev_raw':>8} {'FP':>5} {'FN':>5} {'thr':>6}")
print("-" * 75)

for eid in [baseline] + p39_exps:
    p = Path(f"results/phase36_window_ablation/{eid}/metrics.json")
    tag = "(base)" if eid == baseline else ""
    if not p.exists():
        print(f"{eid:<28} {'N/A':>7} {tag}")
        continue
    m = json.loads(p.read_text())
    tw  = m["metrics"].get("test_window",     {}).get("min_pr")
    ev  = m["metrics"].get("test_event_vote", {}).get("min_pr")
    er  = m["metrics"].get("test_event_raw",  {}).get("min_pr","?")
    fpv = m["metrics"].get("test_event_vote", {}).get("fp","?")
    fnv = m["metrics"].get("test_event_vote", {}).get("fn","?")
    thr = m.get("threshold","?")
    tw_s = f"{tw:.4f}" if tw else "?"
    ev_s = f"{ev:.4f}" if ev else "?"
    mark = " ✅" if tw and ev and tw >= 0.93 and ev >= 0.915 else ""
    print(f"{eid:<28} {tw_s:>7} {ev_s:>8} {str(er):>8} {str(fpv):>5} {str(fnv):>5} {str(thr):>6}{mark} {tag}")
PYEOF

# ── Phase 39 완료 후 Phase 40 자동 시작 ──────────────────────────────────────
log "=== Phase 39 완료 → Phase 40 (Stateful 파인튜닝) 시작 ==="
chmod +x scripts/run_phase40_stateful.sh
bash scripts/run_phase40_stateful.sh
