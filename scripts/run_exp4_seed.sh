#!/usr/bin/env bash
# Experiment 4: Seed stability — kp7-w40-raw only (GPU unavailable, CPU training)
# kp7-w40 uses ~700MB window data vs ~10GB for kp17-w60 — feasible on CPU

set -e
SEEDS=(0 7 13 21 37 99)
OUT_ROOT="results/additional_report_experiments/exp4_seed_stability"
RAW_DIR="dataset/splits_v2_class_balanced"

echo "=== Experiment 4b: Seed Stability (kp7-w40-raw, CPU training) ==="
echo "Seeds: ${SEEDS[*]}"

for SEED in "${SEEDS[@]}"; do
    EXP_ID="seed-kp7-w40-raw-s${SEED}"
    if [ -f "${OUT_ROOT}/${EXP_ID}/metrics.json" ]; then
        echo "SKIP $EXP_ID (already exists)"
        continue
    fi
    echo "[$(date +%H:%M)] Running $EXP_ID ..."
    uv run python scripts/train_window_phase37.py \
        --exp-id "$EXP_ID" \
        --feature-set kp7 \
        --window-size 40 \
        --model-type gru \
        --hidden-sizes 64 32 \
        --epochs 80 \
        --batch-size 512 \
        --lr 0.001 \
        --dropout 0.3 \
        --fall-stride 1 \
        --nfall-stride 5 \
        --focal-alpha 0.65 \
        --focal-gamma 2.0 \
        --pure-window \
        --pure-margin 5 \
        --seed "$SEED" \
        --data-dir "$RAW_DIR" \
        --out-root "$OUT_ROOT"
    echo "[$(date +%H:%M)] Done: $EXP_ID"
done

echo "=== kp7 seed runs complete ==="
