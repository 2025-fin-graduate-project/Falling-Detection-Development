# CLAUDE.md — Falling Model Development

## Project Overview

Fall detection model targeting **STM32N6** deployment.
- Input: MoveNet 17-keypoint pose sequences (15 fps); window size 30~40 frames
- Architecture: **Unidirectional GRU** — STedgeAI INT8 quantizable, stateful streaming compatible
- Output: binary fall/non-fall, triggered by consecutive-window post-processing
- **MinP = min(FallPrecision, NFallPrecision)** — primary metric
  - Float target: MinP ≥ 0.93; INT8 target: MinP ≥ 0.90

---

## Environment

| Component | Version |
|---|---|
| Python | 3.12+ (`uv run python`) |
| TensorFlow / Keras | **2.21.0 / 3.13.2** |
| CUDA / cuDNN | 12.x / 9.x (nvidia-*-cu12 via uv) |
| STedgeAI | 4.0 — `/home/min/app/ST/STEdgeAI/4.0/` |
| STedgeAI internal TF / Keras | **2.18.0 / 3.7.0** |

**Keras compat**: Training emits `.keras` with Keras 3.13 format; STedgeAI 4.0 uses Keras 3.7 and rejects `quantization_config` (added in 3.10). Fix: `scripts/util/export_stedgeai.py` strips the field from the zip before analysis. Must run with STedgeAI's own Python.

**TFLite GRU export**: All GRU models use `TensorListReserve` internally — TFLite requires `unroll=True` rebuild (`scripts/util/reexport_tflite.py`). TFLite file size is inflated (≈ weights × timesteps); actual Flash via STedgeAI is 3–5× smaller.

---

## STM32N6 Deployment

**Target**: STM32N6570-DK / NUCLEO-N657X0-Q, Cortex-M55 CPU

### 추론 구조
- MoveNet (NPU) → keypoints → GRU (CPU, 1-step stateful) → fall score
- 학습: window(30~40f) 단위 stateless / 온디바이스: 매 프레임 1-step, hidden state 유지
- 낙상 알람 시 hidden state 초기화 (`network_reset()`)
- Bidirectional GRU 불가 — 미래 프레임 필요, stateful 스트리밍 불가

### 포팅 경로 (목표)
```
model.keras  →  STedgeAI generate  →  C code + .bin weights  →  flash 0x70680000
```
- **직접 경로**: `.keras` → STedgeAI (TFLite 경유 불필요)
- STedgeAI channel-wise INT8 PTQ — TFLite per-tensor보다 정확도 손실 적음
- `reexport_tflite.py`는 PC에서 window-level 정확도 추정 전용으로만 유지

### I/O 타입 전략
- **기본**: float32 I/O (MoveNet keypoint 출력이 float32이므로 자연스러운 연결)
- **병목 시**: INT8 I/O로 전환 고려 (dequant overhead 실측 후 결정)
- 온디바이스 성능 평가 후 타협점 결정 — 현재 v26은 평가 미실시

### Flash 레이아웃 (xSPI2 NOR 64 MB)
```
0x70000000  FSBL
0x70100000  Application code
0x70380000  MoveNet 256×256 (2.64 MB)
0x70680000  GRU weights  ← 여유 ~60 MB, Flash 제약 없음
```

### 현재 배포 모델 (v26 — PoC)
**목적**: 포팅 가능성 증명 (Proof of Concept). 성능 최적화 대상 아님.
| 항목 | 값 |
|---|---|
| 모델 | `gru_v26_int8.tflite` → GRU(64,32) stateful |
| 포팅 경로 | keras → TFLite INT8 → STedgeAI (구 경로, 이후 미사용) |
| 활성화 버퍼 | 2,816 B |
| MACC/frame | 37,378 |
| 임계값 | score ≥ 0.65, 인물 미감지 45f → reset |
| 온디바이스 성능 | 평가 없음 — STM32N6에서 GRU 구동 가능함을 확인하는 것이 목적 |

### 다음 포팅 대상 (Phase 7/8 완료 후)
최적 모델(GRU(128,64), MinP ≥ 0.93)을 `.keras` → STedgeAI 직접 경로로 포팅.
`Model/generate-gru-model_STM32N6570-DK.sh` 업데이트 후 온디바이스 성능 실측.

**Known STedgeAI sizes (stm32n6 analyze)**:
| Config | Flash (KiB) | Activation (KiB) |
|---|---|---|
| GRU(128,64) kp7 | 537 | ~40 |
| GRU(128,64) kp12 | 559 | ~40 |
| GRU(64,32) — v26 | ~150 | 2.8 |

STedgeAI analyze → `metrics.json["stedgeai"]["analyze"]` (weights_kib, activations_kib, analyze_ok).

---

## Dataset

Canonical: `dataset/splits_v2/` — 9,066 videos, ~1.27M rows (train). Cameras C5~C8 only; outlier-removed; pre-split.
**Do not use** top-level `dataset/*.csv` (unrefined originals).

| Path | Description | Build script |
|---|---|---|
| `dataset/splits_v2/` | Raw kp + HSSC/VHSSC/RWHC | `build_v2_dataset.py` |
| `dataset/splits_v2_filtered/` | One-Euro+EMA on kp, AHSSC/AHSSC_x recomputed | `build_filtered_v2_splits.py` |
| `dataset/lb3_v2/` | splits_v2 + `label_3class` (0=normal,1=falling,2=fallen) | `build_lb3_dataset.py --source splits_v2` |

Fresh setup: splits_v2 already exists. Run lb3 build then filtered build (filtered includes label_3class).

---

## Fixed Training Parameters

| Parameter | Value |
|---|---|
| `--data-scope` | `all` |
| `--min-val-precision` | `0.90` |
| `--early-stop-patience` | `15` |
| `--epochs` | `100` |
| `--dropout-rate` | `0.3` |
| `--noise-std` | `0.02` |
| `--train-negative-stride` | `2` |
| Post-processing sweep | `--min-consecutive-values 1,3,5` (default) |

---

## Experiment Axes

| Axis | Options | Flag |
|---|---|---|
| Preprocessing | `raw` (splits_v2) / `filtered` (splits_v2_filtered) | `--preprocessing` |
| Label | LB-2: `label` / LB-3: `label_3class --num-classes 3 --positive-labels 1,2` | `--label-column` |
| GRU units | **`128,64`** (target) / `256,128` (over Flash budget) | `--gru-units` |
| Feature set | `kp7` (27f) / `kp12` (45f, default) / `kp8` / `all` | `--feature-set` |
| Window | 30f (`--target-steps 30`) / 40f | `--target-steps` |

Bidirectional (`--bidirectional`) and attention (`--temporal-attention`) were explored in earlier phases but are STM32-incompatible or lower-priority — do not use for new deployment-targeted runs.

---

## Runner: `scripts/train_baseline.py`

```bash
uv run python scripts/train_baseline.py \
  --experiment-id Q7-v01 --output-root results/gru_phase7_quant \
  --model-type gru --gru-units 128,64 \
  --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
  --focal-loss --focal-gamma 2.0 --focal-alpha 0.25 \
  --preprocessing filtered --feature-set kp12 \
  --train-csv dataset/splits_v2_filtered/train.csv \
  --val-csv   dataset/splits_v2_filtered/val.csv \
  --test-csv  dataset/splits_v2_filtered/test.csv \
  --label-column label --data-scope all \
  --target-steps 30 --window-start-sec 3.0 --window-end-sec 9.0 \
  --dropout-rate 0.3 --noise-std 0.02 --train-negative-stride 2 \
  --early-stop-patience 15 --epochs 100 --min-val-precision 0.90 --quiet
```

Output per experiment (`{output-root}/{id}/`): `metrics.json`, `model.keras`, `model_fp32.tflite`, `model_int8.tflite`, `run_config.resolved.json`, `threshold_sweep.csv`.

`run_exp` in runner scripts skips if `metrics.json` already exists — safe to re-run after failure.

---

## STedgeAI Analysis

```bash
# Run with STedgeAI's internal Python (Keras 3.7 env)
/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python \
    scripts/util/export_stedgeai.py \
    --exp-dir results/gru_phase7_quant/Q7-v01 --target stm32n6
```

Updates `metrics.json["stedgeai"]` with analyze results. Compat `.keras` is created and deleted automatically.

---

## Experiment Phases (summary)

| Phase | Script | Focus | Status |
|---|---|---|---|
| 1 | `run_gru_phase1.sh` | PP × LB × Arch grid (GRU 256,128) | done |
| 2 | `run_gru_phase2_arch.sh` | Architecture variants | done |
| 3 | `run_gru_phase3_2s.sh` | 30f window (2s) variants | done |
| 4 | `run_gru_phase4_uni_kp.sh` | Unidirectional + KP ablation | done |
| 5 | `run_gru_phase5_compact.sh` | GRU(128,64) compact + focal | done |
| 7 | `run_gru_phase7.sh` | STM32N6 Flash verify + Q7 training | **in progress** |

Best models (MinP ≥ 0.93, unidirectional): P5-v02 (0.9495), P4-v05 (0.9513), P4-v02 (0.9469).

---

## Evaluation

Video-level: threshold → binary windows → consecutive rule → fall if any positive window.
(threshold, min_consecutive) selected by 2D val-set sweep maximising F1 subject to precision ≥ 0.90.

**Metrics hierarchy**: `test_video` (primary) > `val_video` > `test_float` > `test_int8`.

Key `metrics.json` fields: `threshold_selection.{threshold,min_consecutive}`, `metrics.test_video.{f1,recall,min_precision}`, `stedgeai.analyze.{weights_kib,activations_kib,analyze_ok}`.

---

## Multi-Agent Workflow

Each agent: own branch (`experiment/{model}-{focus}-{phase}`) + own `--output-root`. Never modify `dataset/splits_v2/`, `--data-scope all`, or `scripts/train_baseline.py` without PR coordination. Merge to `dev` when done.

---

## Glossary

| Term | Meaning |
|---|---|
| MinP | min(FallPrecision, NFallPrecision) |
| PP-raw / PP-D | raw kp / One-Euro+EMA filtered kp |
| LB-2 / LB-3 | binary / 3-class (normal, falling, fallen) |
| kp7/kp12 | 27/45 features (keypoint subsets) |
| consecutive rule | ≥ K consecutive positive windows before alarm |
