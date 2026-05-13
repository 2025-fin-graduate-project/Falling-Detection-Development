# Colab Baseline Assignment Flow

Phase 0 기준점 모델은 **실험 조합별 독립 노트북**을 기준으로 공유 Drive에서 실행한다.
입력 데이터는 각 config에 명시된 `train_csv`, `val_csv`, `test_csv`를 직접 사용한다.
현재 기준 경로는 `dataset/train.csv`, `dataset/val.csv`, `dataset/test.csv`다.
각 노트북은 하나의 고정 config만 실행하므로 `SESSION_ID`나 선택형 config 생성 단계가 필요 없다.

## Notebooks

- `colab/baselines/tcn_raw_baseline.ipynb`: `B-TCN-raw`
- `colab/baselines/tcn_filtered_baseline.ipynb`: `B-TCN-D`
- `colab/baselines/gru_raw_baseline.ipynb`: `B-GRU-raw`
- `colab/baselines/gru_filtered_baseline.ipynb`: `B-GRU-D`
- `colab/baseline_phase0_flow.ipynb`: baseline notebook index

## Assignment Rule

각 인원에게 위 네 개 중 하나의 노트북을 배정한다. 배정자는 해당 노트북만 실행한다.
빠른 경로 검증이 필요하면 노트북의 학습 셀에서 `SMOKE = True`로 바꾼다.
정식 기준점 산출은 `SMOKE = False`, `EXPORT_TFLITE = True`로 실행한다.

산출물은 다음 위치에 저장된다.

```text
results/baselines_phase0/{experiment_id}/
  metrics.json
  model.keras
  model_fp32.tflite
  model_int8.tflite
  quantization_report.json
  training_curve.png
  threshold_sweep.png
  window_distribution.png
  confusion_matrix.png
  roc_curve.png
  pr_curve.png
  int8_confusion_matrix.png
  int8_roc_curve.png
  int8_pr_curve.png
```

## Recommended Use

1. `TCN raw`, `TCN filtered`, `GRU raw`, `GRU filtered`를 각각 다른 인원에게 배정한다.
2. 각 인원은 자기 노트북에서 smoke를 1회 실행해 Drive 경로와 export 흐름을 확인한다.
3. 같은 노트북에서 `SMOKE = False`로 바꿔 정식 기준점 모델을 학습한다.
4. 모든 기준점 학습이 끝나면 `scripts/collect_baseline_results.py --results-root results/baselines_phase0`로 공통 비교표를 생성한다.
