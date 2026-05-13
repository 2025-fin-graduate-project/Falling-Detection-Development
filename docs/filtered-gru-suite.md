# Filtered Dataset GRU Experiment Suite

작성일: 2026-05-13

## 목적

`final_dataset_filtered.csv` 기반으로 GRU 계열 모델 3~4개를 같은 데이터 split, 같은 후처리, 같은 시각화 파이프라인에서 비교한다. 목표는 논문용 비교표와 그림을 재현 가능하게 만들고, STM32N6570-DK의 GRU stateful 추론 구조로 옮길 수 있는 후보를 선별하는 것이다.

STM32 문서 `../Falling-Detection-STM32/Doc/GRU-Implementation-Architecture.md` 기준 배포 구조는 다음 제약을 가진다.

- 2-layer 단방향 GRU + Dense head
- 학습은 60-step sequence로 수행
- 배포는 1-frame feature와 이전 은닉 상태 `h1`, `h2`를 입력받는 stateful step 모델로 수행
- 현재 STM32 런타임은 `h1`, `h2`, `pose_feature` 입력과 `new_h1`, `fall_scores`, `new_h2` 출력을 관리한다
- 기존 C 구현은 55-dim feature 기준이므로 filtered dataset의 `AHSSC`, `AHSSC_x`까지 쓰려면 STM32 쪽 `POSE_FEATURE_COUNT`와 feature 생성부를 57-dim으로 맞춰야 한다

## 모델 후보

`scripts/train_filtered_gru_suite.py`는 STM32 stateful 변환이 가능한 4개 후보만 기본으로 학습한다.

| 모델 | 구조 | 의도 |
|---|---|---|
| `gru_64_32` | GRU(64) -> GRU(32) -> Dense(32) | 현재 STM32 GRU v26과 가장 가까운 기준 모델 |
| `gru_96_48` | GRU(96) -> GRU(48) -> Dense(48) | 성능 여유를 확인하는 중간 확장 모델 |
| `gru_128_64` | GRU(128) -> GRU(64) -> Dense(64) | 정확도 상한을 보는 고용량 모델 |
| `gru_64_32_light` | GRU(64) -> GRU(32) -> Dense(16), 낮은 dropout | 경량 head로 과적합과 배포 비용을 줄이는 후보 |

Bidirectional GRU, attention, CNN-GRU는 논문 비교용으로는 쓸 수 있지만 현재 STM32 문서의 frame-by-frame stateful 구조와 직접 호환되지 않는다. 따라서 기본 suite에서는 제외했다.

## 실행

필요 패키지:

- `tensorflow`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

```bash
python3 scripts/train_filtered_gru_suite.py \
  --csv-path dataset/final_dataset_filtered.csv \
  --output-dir artifacts/filtered_gru_suite \
  --models all \
  --epochs 40 \
  --batch-size 64 \
  --min-val-recall 0.86
```

빠른 smoke test:

```bash
python3 scripts/train_filtered_gru_suite.py \
  --csv-path dataset/final_dataset_filtered.csv \
  --output-dir artifacts/filtered_gru_suite_smoke \
  --models gru_64_32 \
  --epochs 1 \
  --max-rows 20000 \
  --quant-eval-max-windows 100
```

STM32/TFLite export를 건너뛰고 학습 비교만 하려면:

```bash
python3 scripts/train_filtered_gru_suite.py \
  --csv-path dataset/final_dataset_filtered.csv \
  --skip-stm32-export
```

## 공통 평가 방식

모든 모델은 다음 조건으로 비교한다.

- `video_id` 기준 train/val/test group split
- train window는 낙상 구간을 더 촘촘히, 정상 구간을 더 성기게 샘플링
- validation set에서 `threshold`와 `min_consecutive`를 grid search
- test set에서는 validation에서 선택한 값을 고정
- 기본 목표선은 accuracy, precision, recall, macro F1 모두 `0.90` 이상

저장 지표:

- accuracy
- balanced accuracy
- precision
- recall
- macro F1
- specificity
- false positive rate
- false negative rate
- MCC
- ROC-AUC
- PR-AUC
- confusion matrix
- classification report

## 출력 산출물

전체 suite:

| 파일 | 설명 |
|---|---|
| `suite_config.json` | 실험 설정 |
| `model_comparison.csv` | 모델별 test metric 비교표 |
| `model_comparison.json` | 비교표 JSON |
| `model_comparison.png` | 논문용 모델 비교 bar plot |

모델별 디렉터리:

| 파일 | 설명 |
|---|---|
| `<model>.keras` | 학습된 sequence 모델 |
| `<model>_stateful_step.keras` | STM32 구조와 같은 explicit-state step 모델 |
| `<model>_stateful_fp32.tflite` | stateful FP32 TFLite |
| `<model>_stateful_int8.tflite` | representative dataset 기반 INT8 TFLite |
| `run_metadata.json` | 설정, feature 목록, normalization, metric, export path |
| `threshold_sweep.csv` | validation threshold/min-consecutive 탐색 결과 |
| `history.png` | 학습 곡선 |
| `confusion_matrix_test.png` | test confusion matrix |
| `roc_pr_test.png` | test ROC/PR curve |
| `stm32_runtime_comparison.csv` | Keras stateful / TFLite FP32 / TFLite INT8 metric |
| `stm32_runtime_comparison.png` | STM32 export 런타임별 metric 비교 |

## STM32 반영 시 체크리스트

1. `final_dataset_filtered.csv`의 feature 수가 57인지 확인한다.
2. STM32 `POSE_FEATURE_COUNT`를 55에서 57로 변경한다.
3. C feature 생성부에 `AHSSC`, `AHSSC_x`를 추가한다.
4. 학습 metadata의 `normalization.mean/std` 57개를 C 정규화 테이블로 반영한다.
5. `GRU_FALL_SCORE_THRESHOLD`와 `GRU_FALL_RESET_COUNT`는 `threshold_sweep.csv`에서 선택된 `threshold`, `min_consecutive` 기준으로 업데이트한다.
6. `*_stateful_fp32.tflite`와 `*_stateful_int8.tflite` 모두 ST Edge AI 변환 가능 여부와 latency를 확인한다.
7. INT8 성능이 FP32 대비 크게 떨어지면 FP32 CPU 모델을 우선 배포하고, quantization-aware training 또는 calibration 샘플 확장을 별도 실험한다.

## 해석 기준

논문에서는 다음 순서로 결과를 제시하는 것이 좋다.

1. filtered feature 도입 전후 비교
2. 4개 GRU 후보의 Keras sequence 성능 비교
3. 선택 모델의 stateful step 변환 후 성능 유지 여부
4. FP32 TFLite와 INT8 TFLite의 성능 차이
5. STM32 메모리/latency 제약을 반영한 최종 모델 선택 근거
