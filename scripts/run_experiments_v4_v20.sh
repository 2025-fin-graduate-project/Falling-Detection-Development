#!/usr/bin/env bash
# Sequential experiment runner: B-GRU-raw-v5 ~ v20
# v4 is assumed to be already running; this script waits for it then runs v5-v20.
set -uo pipefail

OUTROOT="results/baselines_phase0"
SUMMARY="$OUTROOT/run_all_summary.log"
mkdir -p "$OUTROOT"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"
}

# ── run one experiment ────────────────────────────────────────────────────────
# Usage: run_exp <id> [extra args...]
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
        log "FAIL  $id (see $logfile)"
    fi
}

# ── wait for v4 ──────────────────────────────────────────────────────────────
log "Waiting for B-GRU-raw-v4 to complete..."
while [[ ! -f "$OUTROOT/B-GRU-raw-v4/metrics.json" ]]; do
    sleep 30
done
log "B-GRU-raw-v4 done — proceeding"

# ════════════════════════════════════════════════════════════════════════════
# Shared flag groups (used by name below for readability)
# ════════════════════════════════════════════════════════════════════════════
BASE_DATA=(
    --train-csv dataset/train.csv
    --val-csv   dataset/val.csv
    --test-csv  dataset/test.csv
    --data-scope no_by
)
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
GRU_ARCH=(
    --model-type gru --preprocessing raw
    --gru-units 128,64
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --feature-set kp12
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --early-stop-patience 15 --epochs 100
    --min-val-precision 0.90
)
TCN_ARCH=(
    --model-type tcn --preprocessing raw
    --tcn-channels 64,64,64,64 --tcn-dilations 1,2,4,8 --tcn-kernel-size 3
    --feature-set kp12
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --early-stop-patience 15 --epochs 100
    --min-val-precision 0.90
)

# ════════════════════════════════════════════════════════════════════════════
# v5  splits_v2 + all directions  (broader, cleaner training data)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v5 \
    "${GRU_ARCH[@]}" "${V2_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v6  Bidirectional GRU  (arch variation, base data)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v6 \
    "${GRU_ARCH[@]}" "${BASE_DATA[@]}" \
    --bidirectional

# ════════════════════════════════════════════════════════════════════════════
# v7  Focal loss alpha=0.5  (loss variation, base data)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v7 \
    "${GRU_ARCH[@]}" "${BASE_DATA[@]}" \
    --focal-loss --focal-alpha 0.5 --focal-gamma 2.0

# ════════════════════════════════════════════════════════════════════════════
# v8  GRU(256,128)  (larger model, base data)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v8 \
    --model-type gru --preprocessing raw \
    --gru-units 256,128 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    "${BASE_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v9  patience=20 + epochs=150  (longer training, base data)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v9 \
    --model-type gru --preprocessing raw \
    --gru-units 128,64 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 20 --epochs 150 \
    --min-val-precision 0.90 \
    "${BASE_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v10  splits_v2 + all + stride=1  (combine v4 + v5)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v10 \
    "${GRU_ARCH[@]}" "${V2_DATA[@]}" \
    --train-negative-stride 1

# ════════════════════════════════════════════════════════════════════════════
# v11  splits_v2 + all + Bidirectional  (combine v5 + v6)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v11 \
    "${GRU_ARCH[@]}" "${V2_DATA[@]}" \
    --bidirectional

# ════════════════════════════════════════════════════════════════════════════
# v12  splits_v2 + all + GRU(256,128)  (combine v5 + v8)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v12 \
    --model-type gru --preprocessing raw \
    --gru-units 256,128 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    "${V2_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v13  PP-D filtered preprocessing  (Phase 0: B-GRU-D)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v13 \
    --model-type gru --preprocessing filtered \
    --gru-units 128,64 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    "${FILT_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v14  TCN baseline raw  (Phase 0: B-TCN-raw)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v14 \
    "${TCN_ARCH[@]}" "${BASE_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v15  TCN + PP-D filtered  (Phase 0: B-TCN-D)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v15 \
    --model-type tcn --preprocessing filtered \
    --tcn-channels 64,64,64,64 --tcn-dilations 1,2,4,8 --tcn-kernel-size 3 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    "${FILT_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v16  KP-8 keypoints + splits_v2 + all  (Phase 2 preview: fewer features)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v16 \
    --model-type gru --preprocessing raw \
    --gru-units 128,64 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp8 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    "${V2_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v17  splits_v2 + all + GRU(256,128) + Bidirectional  (combine v12 + v11)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v17 \
    --model-type gru --preprocessing raw \
    --gru-units 256,128 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    "${V2_DATA[@]}" \
    --bidirectional

# ════════════════════════════════════════════════════════════════════════════
# v18  splits_v2 + all + Bidirectional + stride=1  (ultimate precision push)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v18 \
    "${GRU_ARCH[@]}" "${V2_DATA[@]}" \
    --bidirectional \
    --train-negative-stride 1

# ════════════════════════════════════════════════════════════════════════════
# v19  patience=25 + epochs=200 + splits_v2 + all  (long convergence)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v19 \
    --model-type gru --preprocessing raw \
    --gru-units 128,64 \
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 25 --epochs 200 \
    --min-val-precision 0.90 \
    "${V2_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# v20  TCN + splits_v2 + all  (TCN on clean data)
# ════════════════════════════════════════════════════════════════════════════
run_exp B-GRU-raw-v20 \
    --model-type tcn --preprocessing raw \
    --tcn-channels 64,64,64,64 --tcn-dilations 1,2,4,8 --tcn-kernel-size 3 \
    --feature-set kp12 \
    --dropout-rate 0.3 --noise-std 0.02 \
    --train-negative-stride 2 \
    --early-stop-patience 15 --epochs 100 \
    --min-val-precision 0.90 \
    "${V2_DATA[@]}"

# ════════════════════════════════════════════════════════════════════════════
# Final summary
# ════════════════════════════════════════════════════════════════════════════
log "All experiments complete. Collecting results..."
echo "" | tee -a "$SUMMARY"
echo "=== RESULTS SUMMARY ===" | tee -a "$SUMMARY"
for id in B-GRU-raw-v4 B-GRU-raw-v5 B-GRU-raw-v6 B-GRU-raw-v7 B-GRU-raw-v8 \
          B-GRU-raw-v9 B-GRU-raw-v10 B-GRU-raw-v11 B-GRU-raw-v12 B-GRU-raw-v13 \
          B-GRU-raw-v14 B-GRU-raw-v15 B-GRU-raw-v16 B-GRU-raw-v17 B-GRU-raw-v18 \
          B-GRU-raw-v19 B-GRU-raw-v20; do
    mfile="$OUTROOT/$id/metrics.json"
    if [[ -f "$mfile" ]]; then
        python3 -c "
import json, sys
m = json.load(open('$mfile'))
tv = m.get('test_video', m.get('test', {}))
vv = m.get('val_video', m.get('val', {}))
print(f'$id  test_F1={tv.get(\"f1\",0):.4f}  test_Rec={tv.get(\"recall\",0):.4f}  test_FallPrec={tv.get(\"precision\",0):.4f}  val_F1={vv.get(\"f1\",0):.4f}')
" 2>/dev/null || echo "$id  (metrics parse error)"
    else
        echo "$id  NOT DONE"
    fi
done | tee -a "$SUMMARY"

log "Done. See $SUMMARY"
