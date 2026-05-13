# Colab Baseline Flow

This flow keeps notebooks thin and puts reproducible training behavior in `scripts/train_baseline.py`.

## Workspace

The notebook should mount Google Drive and resolve the project root from one of these candidates:

- `/content/drive/MyDrive/Graduate-Project/Falling-Model-Development`
- `/content/drive/MyDrive/Falling-Model-Development`
- `/content/drive/MyDrive/졸업 과제/Falling-Model-Development`
- `/content/drive/MyDrive/졸업 과제/Falling-Detection-Development`

Set `PROJECT_ROOT_OVERRIDE` in the notebook when the shared folder uses another path.

## Phase 0

Use one fixed notebook per assigned baseline:

- `colab/baselines/tcn_raw_baseline.ipynb`
- `colab/baselines/tcn_filtered_baseline.ipynb`
- `colab/baselines/gru_raw_baseline.ipynb`
- `colab/baselines/gru_filtered_baseline.ipynb`

Each notebook runs one fixed experiment on the cleaned split CSVs declared by `train_csv`, `val_csv`, and `test_csv` in its config, then writes to `results/baselines_phase0/{experiment_id}/`.
The current baseline configs point to `dataset/train.csv`, `dataset/val.csv`, and `dataset/test.csv`.
The direct CLI equivalent is:

```bash
python3 scripts/train_baseline.py \
  --config configs/experiments/phase0/B-TCN-D.json
```

Run a smoke test:

```bash
python3 scripts/train_baseline.py \
  --config configs/experiments/phase0/B-TCN-D.json \
  --smoke
```

Collect common plots and tables after multiple runs:

```bash
python3 scripts/collect_baseline_results.py \
  --results-root results/baselines_phase0
```

## Outputs

Each experiment writes to `results/baselines_phase0/{experiment_id}/`:

- `metrics.json`
- `run_config.resolved.json`
- `feature_columns.json`
- `normalization.json`
- `window_distribution.csv`
- `window_distribution.png`
- `training_curve.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`
- `int8_confusion_matrix.png`
- `int8_roc_curve.png`
- `int8_pr_curve.png`
- `model.keras`
- `model_fp32.tflite`
- `model_int8.tflite`
- `quantization_report.json`

Common comparison files are written to `results/baselines_phase0/common/`.

## Deployment Policy

- TCN is the primary STM32N6 INT8 deployment candidate.
- GRU is kept as a Keras/STM32 CPU baseline. INT8 export is attempted and recorded, but GRU conversion failures are reported as experiment metadata instead of blocking the suite.
