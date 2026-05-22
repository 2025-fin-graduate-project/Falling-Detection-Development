#!/usr/bin/env bash
# Phase 37 순차 ablation 실험 (보고서용)
# 결과: results/phase36_window_ablation/summary_phase37.tsv
# 순서:
#   1. P37-pure-kp7-w40-gru       labeling × feat
#   2. P37-pure-kp13-w30-gru      window size
#   3. P37-pure-kp13-w40-lstm     architecture
#   4. P37-pure-m0-kp13-w40-gru   margin=0
#   5. P37-pure-m10-kp13-w40-gru  margin=10
#   6. P37-pure-vel-kp13-w40-gru  velocity features
#   7. P37-pure-h256-kp13-w40-gru hidden=256,128
#   8. P37-pure-h64-kp13-w40-gru  hidden=64,32

set -uo pipefail
cd "$(dirname "$0")/.."

SUMMARY="results/phase36_window_ablation/summary_phase37.tsv"
LOG_DIR="results/phase36_window_ablation/logs"
mkdir -p "$LOG_DIR" results/phase36_window_ablation

log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }

# TSV 헤더
if [[ ! -f "$SUMMARY" ]]; then
    printf "exp_id\tgroup\tvar\tval_minpr\ttest_minpr\tfall_pr\tfall_rc\tnfall_pr\tnfall_rc\tfn\tfp\tthr\ttrain_fall\ttrain_nfall\tstatus\n" > "$SUMMARY"
fi

run_exp() {
    local exp_id="$1" group="$2" var_desc="$3"
    shift 3
    local train_args=("$@")

    # 이미 완료됐으면 스킵, stdout에 test_minpr만 반환
    if grep -q "^${exp_id}	" "$SUMMARY" 2>/dev/null; then
        log "SKIP $exp_id (already in summary)"
        awk -F'\t' -v id="$exp_id" '$1==id {print $5}' "$SUMMARY"
        return 0
    fi

    log "=== $exp_id  [$group: $var_desc] ==="
    local logfile="$LOG_DIR/${exp_id}.log"

    uv run python scripts/train_window_phase37.py \
        --exp-id "$exp_id" \
        "${train_args[@]}" \
        > "$logfile" 2>&1

    # 결과 추출
    local metrics
    metrics=$(python3 - << PYEOF 2>/dev/null
import json, sys
try:
    m = json.load(open("results/phase36_window_ablation/$exp_id/metrics.json"))
    t = m["metrics"]["test_window"]
    v = m["metrics"]["val_window"]
    ws = m["window_stats"]
    print(f"{v['min_pr']:.4f}\t{t['min_pr']:.4f}\t{t['fall_precision']:.4f}\t{t['fall_recall']:.4f}\t{t['nfall_precision']:.4f}\t{t['nfall_recall']:.4f}\t{t['fn']}\t{t['fp']}\t{m['threshold']}\t{ws['train_fall']}\t{ws['train_nfall']}")
except Exception as e:
    print("FAIL\tFAIL\t-\t-\t-\t-\t-\t-\t-\t-\t-")
PYEOF
    )

    local test_mp
    test_mp=$(echo "$metrics" | cut -f2)

    printf "%s\t%s\t%s\t%s\tok\n" \
        "$exp_id" "$group" "$var_desc" "$metrics" >> "$SUMMARY"

    log "  RESULT $exp_id: test_minpr=${test_mp}"
    echo "$test_mp"
}

# ── 1. labeling×feat: pure-window + kp7 ──────────────────────────────────────
run_exp "P37-pure-kp7-w40-gru" "G1_labeling_feat" "pure+kp7" \
    --feature-set kp7 --window-size 40 --model-type gru \
    --pure-window --pure-margin 5

# ── 2. window size: pure + kp13 + win=30 ─────────────────────────────────────
run_exp "P37-pure-kp13-w30-gru" "G3_window_size" "win=30" \
    --feature-set kp13 --window-size 30 --model-type gru \
    --pure-window --pure-margin 5

# ── 3. architecture: pure + kp13 + w40 + lstm ────────────────────────────────
run_exp "P37-pure-kp13-w40-lstm" "G4_arch" "LSTM" \
    --feature-set kp13 --window-size 40 --model-type lstm \
    --pure-window --pure-margin 5

# ── 4. margin=0: 완전 엄격 ───────────────────────────────────────────────────
run_exp "P37-pure-m0-kp13-w40-gru" "G5_margin" "margin=0" \
    --feature-set kp13 --window-size 40 --model-type gru \
    --pure-window --pure-margin 0

# ── 5. margin=10: 완화 ───────────────────────────────────────────────────────
run_exp "P37-pure-m10-kp13-w40-gru" "G5_margin" "margin=10" \
    --feature-set kp13 --window-size 40 --model-type gru \
    --pure-window --pure-margin 10

# ── 6. velocity features ─────────────────────────────────────────────────────
run_exp "P37-pure-vel-kp13-w40-gru" "G6_features" "velocity" \
    --feature-set kp13 --window-size 40 --model-type gru \
    --pure-window --pure-margin 5 --use-velocity

# ── 7. hidden=256,128 (대형) ──────────────────────────────────────────────────
run_exp "P37-pure-h256-kp13-w40-gru" "G7_hidden" "hidden=256,128" \
    --feature-set kp13 --window-size 40 --model-type gru \
    --pure-window --pure-margin 5 --hidden-sizes 256 128

# ── 8. hidden=64,32 (경량) ───────────────────────────────────────────────────
run_exp "P37-pure-h64-kp13-w40-gru" "G7_hidden" "hidden=64,32" \
    --feature-set kp13 --window-size 40 --model-type gru \
    --pure-window --pure-margin 5 --hidden-sizes 64 32

# ── 최종 요약 ─────────────────────────────────────────────────────────────────
log "=== Phase 37 전체 완료 ==="
echo ""
echo "=== Phase 37 Ablation 결과 요약 ==="
python3 - << 'PYEOF'
import csv

# Phase 36 기준선 포함
baselines = [
    ("P36-kp7-w40-gru",   "G0_baseline", "last-frame+kp7",  "-",      "0.7771", "0.7878", "0.7771", "0.9769", "0.9783"),
    ("P36-kp13-w40-gru",  "G0_baseline", "last-frame+kp13", "-",      "0.7843", "0.7985", "0.7843", "0.9777", "0.9795"),
    ("P37-pure-kp13-w40-gru", "G0_best", "pure+kp13(base)", "0.9477", "0.9404", "0.9404", "0.9421", "0.9911", "0.9908"),
]

rows = []
with open("results/phase36_window_ablation/summary_phase37.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows.append(r)

print(f"{'ID':<35} {'group':<18} {'var':<18} {'val':>7} {'test':>7} {'f_pr':>6} {'f_rc':>6} {'nf_pr':>6} {'nf_rc':>6}")
print("-"*115)

# 기준선
for b in baselines:
    print(f"{b[0]:<35} {b[1]:<18} {b[2]:<18} {b[3]:>7} {b[4]:>7} {b[5]:>6} {b[6]:>6} {b[7]:>6} {b[8]:>6}")
print()

for r in rows:
    val = r.get("val_minpr","-"); test = r.get("test_minpr","-")
    flag = " *** BEST" if float(test) >= 0.94 else "" if test not in ("FAIL","-") else ""
    print(f"{r['exp_id']:<35} {r['group']:<18} {r['var']:<18} "
          f"{val:>7} {test:>7} "
          f"{r.get('fall_pr','-'):>6} {r.get('fall_rc','-'):>6} "
          f"{r.get('nfall_pr','-'):>6} {r.get('nfall_rc','-'):>6}{flag}")
PYEOF
