#!/usr/bin/env bash
# Phase 9 MinPR full run - retrain top screen candidates and record deploy checks.

set -uo pipefail

TARGET_MIN_PR="${TARGET_MIN_PR:-0.92}"
SCREEN_ROOT="${SCREEN_ROOT:-results/phase9_minpr_screen}"
OUTROOT="${OUTROOT:-results/phase9_minpr_full}"
SUMMARY="$OUTROOT/summary.log"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_STRIDE="${EVAL_STRIDE:-1}"
TOP_N="${TOP_N:-3}"
STEDGE_PY="${STEDGE_PY:-/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python}"

if [[ "${TRAIN_DEVICE:-cpu}" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTROOT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

COMMON=(
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label
    --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --eval-stride "$EVAL_STRIDE"
    --epochs "$EPOCHS" --early-stop-patience "$PATIENCE" --batch-size "$BATCH_SIZE"
    --min-val-min-pr "$TARGET_MIN_PR"
    --min-consecutive-values 1,3,5
)

FOCAL=(--focal-loss --focal-gamma 2.0 --focal-alpha 0.25)
WIN30=(--target-steps 30 --window-start-sec 3.0 --window-end-sec 9.0)
WIN40=(--target-steps 40 --window-start-sec 3.0 --window-end-sec 9.0)

screen_to_full_id() {
    case "$1" in
        P9S-v01) echo "P9F-v01" ;;
        P9S-v02) echo "P9F-v02" ;;
        P9S-v03) echo "P9F-v03" ;;
        P9S-v04) echo "P9F-v04" ;;
        P9S-v05) echo "P9F-v05" ;;
        P9S-v06) echo "P9F-v06" ;;
        P9S-v07) echo "P9F-v07" ;;
        *) return 1 ;;
    esac
}

run_candidate() {
    local screen_id="$1"
    local id
    id=$(screen_to_full_id "$screen_id") || return 1
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id ($screen_id) - already complete"
        return 0
    fi
    log "START $id from $screen_id"
    case "$screen_id" in
        P9S-v01)
            uv run python scripts/train_baseline.py --experiment-id "$id" --output-root "$OUTROOT" --quiet "${COMMON[@]}" --model-type gru --gru-units 128,64 "${WIN40[@]}" --feature-set kp7 --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 "${FOCAL[@]}" 2>&1 | tee "$logfile"
            ;;
        P9S-v02)
            uv run python scripts/train_baseline.py --experiment-id "$id" --output-root "$OUTROOT" --quiet "${COMMON[@]}" --model-type gru --gru-units 128,64 "${WIN30[@]}" --feature-set kp7 --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 "${FOCAL[@]}" 2>&1 | tee "$logfile"
            ;;
        P9S-v03)
            uv run python scripts/train_baseline.py --experiment-id "$id" --output-root "$OUTROOT" --quiet "${COMMON[@]}" --model-type gru --gru-units 128,64 "${WIN30[@]}" --feature-set kp12 --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 "${FOCAL[@]}" 2>&1 | tee "$logfile"
            ;;
        P9S-v04)
            uv run python scripts/train_baseline.py --experiment-id "$id" --output-root "$OUTROOT" --quiet "${COMMON[@]}" --model-type gru --gru-units 256,128 "${WIN40[@]}" --feature-set minimal "${FOCAL[@]}" 2>&1 | tee "$logfile"
            ;;
        P9S-v05)
            uv run python scripts/train_baseline.py --experiment-id "$id" --output-root "$OUTROOT" --quiet "${COMMON[@]}" --model-type tcn --tcn-channels 32,64,64,96 "${WIN30[@]}" --feature-set kp7 2>&1 | tee "$logfile"
            ;;
        P9S-v06)
            uv run python scripts/train_baseline.py --experiment-id "$id" --output-root "$OUTROOT" --quiet "${COMMON[@]}" --model-type tcn --tcn-channels 32,64,64,96 "${WIN40[@]}" --feature-set kp7 2>&1 | tee "$logfile"
            ;;
        P9S-v07)
            uv run python scripts/train_baseline.py --experiment-id "$id" --output-root "$OUTROOT" --quiet "${COMMON[@]}" --model-type tcn --tcn-channels 64,64,128,128 "${WIN30[@]}" --feature-set kp7 2>&1 | tee "$logfile"
            ;;
    esac
    local status=${PIPESTATUS[0]}
    if [[ "$status" -eq 0 ]]; then
        log "OK    $id"
    else
        log "FAIL  $id"
    fi
    return "$status"
}

run_stedgeai_for_passes() {
    local exp_dir="$1"
    local id
    id=$(basename "$exp_dir")
    if [[ ! -f "$exp_dir/metrics.json" ]]; then
        return
    fi
    if ! python3 - "$exp_dir/metrics.json" "$TARGET_MIN_PR" <<'PYEOF'
import json
import sys
metrics = json.loads(open(sys.argv[1], encoding="utf-8").read())
target = float(sys.argv[2])
test_video = metrics.get("metrics", {}).get("test_video", {})
sys.exit(0 if float(test_video.get("min_pr", 0.0)) >= target else 1)
PYEOF
    then
        log "STEDGEAI SKIP $id - float minPR below ${TARGET_MIN_PR}"
        return
    fi
    log "STEDGEAI analyze $id"
    "$STEDGE_PY" scripts/util/export_stedgeai.py \
        --exp-dir "$exp_dir" \
        --target stm32n6 \
        2>&1 | tee -a "$SUMMARY" || log "STEDGEAI analyze WARN $id"

    log "STEDGEAI host eval $id"
    uv run python scripts/util/eval_stedgeai_host.py \
        --exp-dir "$exp_dir" \
        --reselect-threshold \
        --eval-stride "$EVAL_STRIDE" \
        2>&1 | tee -a "$SUMMARY" || log "STEDGEAI host WARN $id"
}

if [[ -n "${PHASE9_FULL_IDS:-}" ]]; then
    mapfile -t SELECTED < <(printf "%s\n" "$PHASE9_FULL_IDS" | tr ',' '\n' | sed '/^$/d')
else
    mapfile -t SELECTED < <(python3 scripts/report_phase9_minpr_status.py --output-root "$SCREEN_ROOT" --top "$TOP_N" --ids-only)
fi

log "=== Phase 9 MinPR full: selected=${SELECTED[*]:-none}, target=${TARGET_MIN_PR}, epochs=${EPOCHS}, eval_stride=${EVAL_STRIDE} ==="
for screen_id in "${SELECTED[@]}"; do
    run_candidate "$screen_id"
done

log "=== Phase 9 deploy checks for float minPR passes ==="
for screen_id in "${SELECTED[@]}"; do
    full_id=$(screen_to_full_id "$screen_id") || continue
    run_stedgeai_for_passes "$OUTROOT/$full_id"
done

log "=== Phase 9 MinPR full summary ==="
python3 scripts/report_phase9_minpr_status.py \
    --output-root "$OUTROOT" \
    --target-min-pr "$TARGET_MIN_PR" \
    2>&1 | tee -a "$SUMMARY"
