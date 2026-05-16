# Phase 30 Summary: val_loss 체인 — Architecture × Checkpoint 재설계

작성일: 2026-05-16

## 목적

val_event/video_min_pr 체크포인트가 후처리(threshold×mc) 격자에 과적합하여 모델을 보수적으로 (FN↑, NFallPrecision↓, FallRecall↓) 유도한다는 가설 검증.
val_loss로 전환하여 실질 분리 능력 최적화. GRU/LSTM/TCN 아키텍처 비교.

## 설정 변경

| 항목 | 이전 (P27-vm0) | Phase 30 |
|---|---|---|
| Checkpoint | val_video_min_pr | **val_loss** |
| Epochs | 30 | 100 |
| Patience | 5 | 15 |
| Architecture | GRU(128,64) | GRU / LSTM / TCN / TCN-lg |

## 결과

| ID | 아키텍처 | EventMinP | FN | FP | fall_pr | nfall_pr | fall_rc |
|---|---|---:|---:|---:|---:|---:|---:|
| `P30-lstm` | LSTM(128,64) | **0.9181** | 19 | 19 | 0.9694 | 0.9181 | 0.9694 |
| `P30-gru` | GRU(128,64) | 0.9087 | 22 | 13 | 0.9788 | 0.9087 | 0.9646 |
| `P30-tcn` | TCN [32,32,64,96] | 0.8983 | 24 | 20 | 0.9676 | 0.8983 | 0.9614 |
| `P30-tcn-lg` | TCN [64,64,128,128] k=5 | 0.8740 | 31 | 17 | 0.9720 | 0.8740 | 0.9501 |

## 전체 기준 비교

| 실험 | 아키텍처 | Checkpoint | EventMinP | FN | FP |
|---|---|---|---:|---:|---:|
| P27-vm0 | GRU(128,64) | val_video_min_pr | **0.9241** | 18 | 13 |
| P30-lstm | LSTM(128,64) | val_loss | 0.9181 | 19 | 19 |
| P30-gru | GRU(128,64) | val_loss | 0.9087 | 22 | 13 |

## 분석

### val_loss 효과
- GRU val_loss (0.9087) < GRU val_video_min_pr (0.9241): val_loss 전환이 GRU에서는 오히려 악화
- val_video_min_pr은 "후처리 격자 편향"이 있음에도 GRU에서 더 좋은 가중치 선택
- **가설이 부분적으로 틀렸음**: checkpoint 선택 자체가 아닌 아키텍처가 병목일 수 있음

### LSTM 특이점
- **FP = FN = 19**: 오분류가 완벽히 균형. GRU(FP=13, FN=22)와 대조적
- val_loss 학습에서 LSTM이 더 균형잡힌 결정 경계 학습
- fall_pr = fall_rc = 0.9694: 예측 정밀도와 재현율 일치 (FP=FN이므로 성립)
- LSTM이 fall/non-fall 패턴을 더 대칭적으로 모델링

### TCN 결과
- TCN-lg(더 큰 TCN): FN=31로 최악 — capacity 증가가 오히려 역효과
- TCN은 window-based (stateful streaming 불가), 배포에서도 불리

## 결론

1. **LSTM이 val_loss에서 최우수**: 0.9181, 균형잡힌 FP=FN=19
2. **미탐색 핵심 조합**: LSTM + val_video_min_pr — LSTM 아키텍처 + 타겟 체크포인트
3. **TCN 제외**: val_loss/large 모두 저조, 배포 제약 있음

## 다음 단계 (Phase 31)

LSTM + val_video_min_pr (seed=42, 1, 2) 탐색.
LSTM의 균형잡힌 학습 특성 + val_video_min_pr의 목표 지향 체크포인트 조합.
