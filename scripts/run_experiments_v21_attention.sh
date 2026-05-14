#!/usr/bin/env bash
# Temporal Attention experiments: v21~v25
# Waits for the v4-v20 runner to finish, then runs attention variants.
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

# Wait for v20 to finish
log "Waiting for B-GRU-raw-v20 to complete..."
while [[ ! -f "$OUTROOT/B-GRU-raw-v20/metrics.json" ]]; do
    sleep 30
done
log "B-GRU-raw-v20 done — starting attention experiments"

V2_DATA=(
    --train-csv dataset/splits_v2/train.csv
    --val-csv   dataset/splits_v2/val.csv
    --test-csv  dataset/splits_v2/test.csv
    --data-scope all
)
FILT_DATA=(
    --train-csv dataset/train_filtered.csv
    --val-csv   dataset/val_filtered.csv
    --test-csv  dataset/test_filtered.csv
    --data-scope no_by
)

# ════════════════════════════════════════════════════════════════════════════
# v21  GRU(128,64) + Temporal Attention + splits_v2+all
#      (v5 + attention)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v21 \
    --model-type gru --preprocessing raw \
    --gru-units 128,64 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    --temporal-attention \
    "${V2_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v22  GRU(256,128) + Temporal Attention + splits_v2+all
#      (v12 + attention — best F1 base)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v22 \
    --model-type gru --preprocessing raw \
    --gru-units 256,128 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    --temporal-attention \
    "${V2_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v23  Bidirectional + Temporal Attention + splits_v2+all
#      (v11 + attention — best FallPrec base)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v23 \
    --model-type gru --preprocessing raw \
    --gru-units 128,64 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    --bidirectional \
    --temporal-attention \
    "${V2_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v24  GRU(256,128) + Bidirectional + Temporal Attention + splits_v2+all
#      (ultimate combo)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v24 \
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
    "${V2_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v25  PP-D filtered + Temporal Attention
#      (v13 + attention — best AUC base)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v25 \
    --model-type gru --preprocessing filtered \
    --gru-units 128,64 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    --temporal-attention \
    "${FILT_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# Final summary
# ════════════════════════════════════════════════════════════════════════════
log "Attention experiments complete. Summary:"
echo "" | tee -a "$SUMMARY"
echo "=== ATTENTION RESULTS ===" | tee -a "$SUMMARY"
for id in B-GRU-raw-v21 B-GRU-raw-v22 B-GRU-raw-v23 B-GRU-raw-v24 B-GRU-raw-v25; do
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

log "All done."
