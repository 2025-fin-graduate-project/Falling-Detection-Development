# Colab Shared Session Flow

모델별 단일 노트북을 기준으로 공유 Drive에서 여러 인원이 동시에 실험하는 구조다.

## Notebooks

- `colab/tcn_baseline_session.ipynb`: TCN 전용
- `colab/gru_baseline_session.ipynb`: GRU 전용
- `colab/baseline_phase0_flow.ipynb`: 전용 노트북 위치를 안내하는 index

## Session Rule

각 인원은 자기 노트북 상단에서 아래 값을 고유하게 바꾼다.

```python
SESSION_ID = "tcn_01_yourname"
OWNER = "yourname"
PREPROCESSING = "filtered"  # raw, filtered, both
SMOKE = True
```

산출물은 다음 위치에 저장된다.

```text
results/shared_sessions/{model_type}/{SESSION_ID}/
  session_manifest.json
  B-TCN-D/ 또는 B-GRU-D/
    metrics.json
    model.keras
    model_fp32.tflite
    model_int8.tflite
    training_curve.png
    threshold_sweep.png
    window_distribution.png
    confusion_matrix.png
    roc_curve.png
    pr_curve.png
  common/
    phase0_summary.csv
    compare_f1_bar.png
    compare_precision_recall.png
```

## Recommended Use

1. 각 인원에게 모델과 `SESSION_ID`를 배정한다.
2. 처음에는 `SMOKE=True`로 경로, 데이터, export 흐름을 확인한다.
3. 정상 동작하면 `SMOKE=False`로 정식 학습을 실행한다.
4. `PREPROCESSING="both"`로 raw/filter 비교를 같은 세션에 남긴다.
5. 공유할 때는 세션 폴더 경로와 `session_manifest.json`을 같이 전달한다.
