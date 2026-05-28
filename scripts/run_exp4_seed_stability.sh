#!/usr/bin/env bash
# Experiment 4: Seed stability — 6 seeds × 3 top models = 18 training runs
#
# Seeds: 0, 7, 13, 21, 37, 99 (42 was the original seed, excluded to avoid duplication)
# Models:
#   A) P42-kp17-w60:   kp17, w=60, filtered, no-vel, focal_alpha=0.65
#   B) P41-kp13-w60:   kp13, w=60, filtered, no-vel, focal_alpha=0.65
#   C) P41-raw-kp7-w40: kp7, w=40, raw,      no-vel, focal_alpha=0.65
#
# Output: results/additional_report_experiments/exp4_seed_stability/

set -e
SEEDS=(0 7 13 21 37 99)
OUT_ROOT="results/additional_report_experiments/exp4_seed_stability"
FILTERED_DIR="dataset/splits_v2_class_balanced_filtered"
RAW_DIR="dataset/splits_v2_class_balanced"

echo "=== Experiment 4: Seed Stability ==="
echo "Seeds: ${SEEDS[*]}"
echo "Output: $OUT_ROOT"

for SEED in "${SEEDS[@]}"; do
    # Model A: kp17, w=60, filtered
    EXP_ID="seed-kp17-w60-s${SEED}"
    if [ ! -f "${OUT_ROOT}/${EXP_ID}/metrics.json" ]; then
        echo "[$(date +%H:%M)] Running $EXP_ID ..."
        uv run python scripts/train_window_phase37.py \
            --exp-id "$EXP_ID" \
            --feature-set kp17 \
            --window-size 60 \
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
            --data-dir "$FILTERED_DIR" \
            --out-root "$OUT_ROOT"
        echo "[$(date +%H:%M)] Done: $EXP_ID"
    else
        echo "SKIP $EXP_ID (already exists)"
    fi

    # Model B: kp13, w=60, filtered
    EXP_ID="seed-kp13-w60-s${SEED}"
    if [ ! -f "${OUT_ROOT}/${EXP_ID}/metrics.json" ]; then
        echo "[$(date +%H:%M)] Running $EXP_ID ..."
        uv run python scripts/train_window_phase37.py \
            --exp-id "$EXP_ID" \
            --feature-set kp13 \
            --window-size 60 \
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
            --data-dir "$FILTERED_DIR" \
            --out-root "$OUT_ROOT"
        echo "[$(date +%H:%M)] Done: $EXP_ID"
    else
        echo "SKIP $EXP_ID (already exists)"
    fi

    # Model C: kp7, w=40, raw
    EXP_ID="seed-kp7-w40-raw-s${SEED}"
    if [ ! -f "${OUT_ROOT}/${EXP_ID}/metrics.json" ]; then
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
    else
        echo "SKIP $EXP_ID (already exists)"
    fi
done

echo ""
echo "=== All seed stability runs complete ==="
echo "Run analysis with: uv run python scripts/util/analyze_seed_stability.py"
