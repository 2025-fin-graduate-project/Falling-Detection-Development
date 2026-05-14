# TCN Baseline Automated Experiment Plan

## Goal

Build a reproducible TCN baseline flow for STM32N6 fall detection. The flow first searches for a stronger ST Edge AI/TFLite INT8-safe TCN layer baseline on `splits_v2` raw LB-2, then applies preprocessing and label-schema variables only after a baseline architecture is selected.

## Data Policy

- Required first dataset: `dataset/splits_v2/{train,val,test}.csv`.
- Optional uploaded datasets: `dataset/lb3_v2/{train,val,test}.csv` and `dataset/splits_v2_filtered/{train,val,test}.csv`.
- The automation does not generate or modify dataset files. Missing combinations are recorded as `pending_data`.
- `dataset/splits_v2` and derived datasets are treated as read-only ground truth.

## Execution Flow

1. Run `uv sync`.
2. Verify WSL GPU visibility:
   ```bash
   bash scripts/check_tensorflow_gpu.sh
   ```
3. Run a smoke experiment:
   ```bash
   uv run python scripts/auto_tcn_experiments.py \
     --mode smoke \
     --output-root results/tcn_auto_smoke \
     --max-experiments 1
   ```
4. Run the automated suite:
   ```bash
   mkdir -p results/tcn_auto
   nohup uv run python scripts/auto_tcn_experiments.py \
     --mode auto \
     --output-root results/tcn_auto \
     > results/tcn_auto/nohup.log 2>&1 &
   echo $! > results/tcn_auto/run.pid
   ```

## Experiment Policy

Stage 1 searches TCN layer variants on `raw + LB-2 + kp12`.

Deployment safety rules:

- Do not use temporal dilation greater than 1. ST Edge AI TFLite `CONV_2D` supports int8, but dilation factors different from 1 are not supported for int8 models.
- Do not add custom/Lambda layers, adaptive graph ops, dynamic shape ops, or attention blocks until a generated `.tflite` passes ST Edge AI validation.
- Prefer Conv/BatchNorm/ReLU/Add/Concat/Reshape/Dense/Pooling operators.
- Use explicit pooling layers instead of generic reduce-based global pooling where possible.

| ID | Layer variant | Purpose |
|---|---|---|
| `TCN-SAFE-v01` | channels `32,32,64,96`, dilations `1,1,1,1`, kernel `3` | Existing baseline capacity without dilation |
| `TCN-SAFE-v02` | channels `24,24,48,64`, dilations `1,1,1,1` | Smaller INT8-friendly shape |
| `TCN-SAFE-v03` | channels `32,64,96,128`, dilations `1,1,1,1` | Capacity ceiling |
| `TCN-SAFE-v04` | channels `32,32,64,64,96`, dilations `1,1,1,1,1` | Deeper non-dilated temporal stack |
| `TCN-SAFE-v05` | kernel `5`, dilations `1,1,1,1` | Wider local temporal receptive field |

Stage 2 applies variables using the best completed architecture from Stage 1.

Primary selection metric:

- Maximize `min(fall precision, non-fall precision)` on `test_video`.
- Strict target: `min_precision >= 0.90`.
- Minimum acceptable floor for a candidate worth tuning: `min_precision >= 0.89`.

Pass criteria:

- `test_video.f1 >= 0.91`
- `test_video.recall >= 0.90`
- `min(test_video fall precision, test_video non-fall precision) >= 0.90`
- `test_int8.f1 >= 0.90`

Variable queue:

| ID | Dataset | Purpose |
|---|---|---|
| `TCN-VAR-v01` | filtered LB-2 | Preprocessing comparison when uploaded |
| `TCN-VAR-v02` | raw LB-3 | Label schema comparison when uploaded |
| `TCN-VAR-v03` | filtered LB-3 | Combined filtered/LB-3 comparison when uploaded |

Stage 3 tuning candidates are appended only after completed metrics exist:

- Low min precision but `>= 0.89`: `--min-val-precision 0.92 --min-consecutive-values 3,5,7`.
- Low min precision below `0.89`: `--min-val-precision 0.93 --min-consecutive-values 5,7,9`.
- Low recall: `--min-consecutive-values 1,2,3,5`.
- Low INT8 F1: expand calibration with `--representative-samples 512 --quant-eval-max-windows 0`, then try `--tcn-channels 24,24,48,64`.

## Outputs

Each run writes:

- `state.json`: current and final experiment state.
- `summary.csv`: compact table of status, fall precision, non-fall precision, min precision, F1, threshold, consecutive count, and model size.
- `<experiment-id>.log`: per-experiment training log.
- `<experiment-id>/metrics.json`: authoritative metrics from `scripts/train_baseline.py`.
- `<experiment-id>/model_fp32.tflite` and `<experiment-id>/model_int8.tflite` when export succeeds.

## Notes

- TensorFlow 2.21 sees the RTX 5070 Ti under WSL when CUDA libraries from `.venv` are added to `LD_LIBRARY_PATH`.
- The first CUDA run can JIT compile kernels for compute capability 12.0a, so the first GPU check or first training step may be slower than later runs.
- The automation deliberately avoids git commit or push operations.
