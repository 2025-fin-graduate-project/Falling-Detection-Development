#!/usr/bin/env bash
# Experiment 2: A/B/C/D filter stage comparison
#
# Trains kp7-w40 GRU(64,32) on each filter stage dataset:
#   A: raw (splits_v2_class_balanced)
#   B: One-Euro only (splits_v2_class_balanced_filter_B)
#   C: One-Euro + EMA (splits_v2_class_balanced_filter_C)
#   D: All features (splits_v2_class_balanced_filtered)
#
# Prerequisite: run build_filter_stage_datasets.py first
# Output: results/additional_report_experiments/exp2_filter_stages/

set -e
OUT_ROOT="results/additional_report_experiments/exp2_filter_stages"
FILTERED_DIR="dataset/splits_v2_class_balanced_filtered"

declare -A STAGE_DIRS=(
    ["A"]="dataset/splits_v2_class_balanced"
    ["B"]="dataset/splits_v2_class_balanced_filter_B"
    ["C"]="dataset/splits_v2_class_balanced_filter_C"
    ["D"]="$FILTERED_DIR"
)

echo "=== Experiment 2: Filter Stage Comparison ==="

for STAGE in A B C D; do
    DATA_DIR="${STAGE_DIRS[$STAGE]}"
    EXP_ID="filter-stage-${STAGE}-kp7-w40"

    if [ ! -d "$DATA_DIR" ]; then
        echo "SKIP stage $STAGE — $DATA_DIR not found"
        continue
    fi

    if [ -f "${OUT_ROOT}/${EXP_ID}/metrics.json" ]; then
        echo "SKIP $EXP_ID (already exists)"
        continue
    fi

    echo "[$(date +%H:%M)] Running stage $STAGE: $EXP_ID ..."
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
        --seed 42 \
        --data-dir "$DATA_DIR" \
        --out-root "$OUT_ROOT"
    echo "[$(date +%H:%M)] Done: $EXP_ID"
done

echo ""
echo "=== All filter stage runs complete ==="
echo "Results in: $OUT_ROOT"
