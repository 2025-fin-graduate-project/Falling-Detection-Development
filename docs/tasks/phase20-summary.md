# Phase 20 Summary: Class-Balanced Dataset Split

작성일: 2026-05-16

## 목적

기존 데이터셋 분할의 불균형 문제를 해결하기 위해, 비디오 단위 분할을 유지하면서 방향(direction)과 비디오 라벨(video_label)을 계층화(stratify)하고 프레임 라벨 비율을 맞춘 새 데이터셋 분할(`splits_v2_class_balanced`)에서의 성능을 확인했다. 또한 임계값(threshold) 선택 기준을 `event-level`로 전환하여 실제 낙상 이벤트 감지 성능을 최적화했다.

## 주요 변경

- **Dataset**: `dataset/splits_v2_class_balanced_filtered` (신규 분할)
  - Train/Val/Test 비디오 중복 없음
  - 방향 및 라벨 계층화 적용
- **Threshold Selection**: `--threshold-eval-level event` (이벤트 기반 임계값 탐색)
- **Model**: GRU(128,64) 기반, Focal Loss (alpha=0.25), neg_stride=2를 기본으로 변형 실험

## 결과

| ID | 변경점 | Test Event MinPR | Test Video MinPR | Threshold / mc | Event CM |
| --- | --- | ---: | ---: | --- | --- |
| `P20-v01` | 기본 (alpha=0.25, neg_stride=2) | 0.9121 | 0.9160 | 0.525 / 2 | TN=218 FP=14 FN=21 TP=600 |
| `P20-v02` | alpha=0.15 | 0.8934 | 0.8971 | 0.450 / 3 | TN=218 FP=14 FN=26 TP=595 |
| `P20-v03` | neg_stride=1 | **0.9142** | **0.9181** | 0.500 / 3 | TN=213 FP=19 FN=20 TP=601 |
| `P20-v04` | GRU(96,48) (Compact) | 0.8966 | 0.8966 | 0.500 / 4 | TN=208 FP=24 FN=24 TP=597 |

## 결론

- **Best Model**: `P20-v03` (neg_stride=1)이 이벤트 기반 MinPR 0.9142로 가장 우수한 성능을 보였다.
- **데이터셋 신뢰도**: 새 분할 방식을 통해 모델 평가의 일반화 성능을 더 정확히 측정할 수 있게 되었다.
- **임계값 기준**: `--threshold-eval-level event`를 사용함으로써, 단순 프레임/비디오 단위가 아닌 실제 낙상 발생 구간을 정확히 잡는 임계값을 선택할 수 있게 되었다.
- **향후 과제**: 학습 중 체크포인트 선택(best epoch)도 `val_loss`가 아닌 `val_event_min_pr` 기준으로 수행하여 성능을 극대화할 필요가 있으며, 이는 Phase 21에서 진행 중이다.
