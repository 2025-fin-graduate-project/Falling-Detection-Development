#!/usr/bin/env bash
# LB-3 (3-class label) experiments: v26~v30
# Waits for the attention runner (v25) to finish, then runs LB-3 variants.
# All experiments use dataset/lb3_v2 (splits_v2 + label_3class) with data-scope=all.
set -uo pipefail

OUTROOT="results/baselines_phase0"
SUMMARY="$OUTROOT/run_all_summary.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"
}

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id — already complete"
        return 0
    fi
    log "START $id"
    if uv run python scripts/train_baseline.py \
            --experiment-id "$id" \
            --output-root  "$OUTROOT" \
            --quiet \
            "$@" \
            2>&1 | tee "$logfile"; then
        log "OK    $id"
    else
        log "FAIL  $id"
    fi
}

# Wait for attention runner to finish (v25)
log "Waiting for B-GRU-raw-v25 to complete..."
while [[ ! -f "$OUTROOT/B-GRU-raw-v25/metrics.json" ]]; do
    sleep 30
done
log "B-GRU-raw-v25 done — starting LB-3 experiments"

LB3_DATA=(
    --train-csv dataset/lb3_v2/train.csv
    --val-csv   dataset/lb3_v2/val.csv
    --test-csv  dataset/lb3_v2/test.csv
    --label-column label_3class
    --positive-labels 1,2
    --num-classes 3
    --data-scope all
)

# ════════════════════════════════════════════════════════════════════════════
# v26  LB-3 baseline: GRU(128,64) + splits_v2 + all
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v26 \
    --model-type gru --preprocessing raw \
    --gru-units 128,64 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    "${LB3_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v27  LB-3 + GRU(256,128)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v27 \
    --model-type gru --preprocessing raw \
    --gru-units 256,128 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    "${LB3_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v28  LB-3 + Bidirectional GRU
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v28 \
    --model-type gru --preprocessing raw \
    --gru-units 128,64 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    --bidirectional \
    "${LB3_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v29  LB-3 + Temporal Attention
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v29 \
    --model-type gru --preprocessing raw \
    --gru-units 128,64 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    --temporal-attention \
    "${LB3_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v30  LB-3 + GRU(256,128) + Bidirectional + Temporal Attention  (ultimate)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v30 \
    --model-type gru --preprocessing raw \
    --gru-units 256,128 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    --bidirectional \
    --temporal-attention \
    "${LB3_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# Final summary
# ════════════════════════════════════════════════════════════════════════════
log "LB-3 experiments complete. Summary:"
echo "" | tee -a "$SUMMARY"
echo "=== LB-3 RESULTS ===" | tee -a "$SUMMARY"
for id in B-GRU-raw-v26 B-GRU-raw-v27 B-GRU-raw-v28 B-GRU-raw-v29 B-GRU-raw-v30; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json
m = json.load(open('$mfile'))
tv = m['metrics'].get('test_video', {})
vv = m['metrics'].get('val_video', {})
print(f'$id  testF1={tv.get(\"f1\",0):.4f}  Rec={tv.get(\"recall\",0):.4f}  FallPrec={tv.get(\"precision\",0):.4f}  valF1={vv.get(\"f1\",0):.4f}')
" 2>/dev/null || echo "$id  (parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "All LB-3 experiments done."
