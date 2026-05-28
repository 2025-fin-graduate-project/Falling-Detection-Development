# Phase 31 Summary: LSTM × Checkpoint 탐색

작성일: 2026-05-16

## 목적

Phase 30에서 LSTM(val_loss)=0.9181이 GRU(val_loss)=0.9087을 능가. 미탐색 조합인 LSTM + {val_video_min_pr, val_event_min_pr}로 GRU P27-vm0(0.9241) 돌파 가능성 확인.

## 결과

| ID | Checkpoint | Seed | EventMinP | FN | FP |
|---|---|---:|---:|---:|---:|
| `P31-vm42` | val_video_min_pr | 42 | 0.9038 | 23 | 16 |
| `P31-ev1` | val_event_min_pr | 1 | 0.8992 | 24 | 18 |
| `P31-vm1` | val_video_min_pr | 1 | 0.8917 | 26 | 18 |
| `P31-vm2` | val_video_min_pr | 2 | 0.8870 | 27 | 20 |
| `P31-ev42` | val_event_min_pr | 42 | 0.8651 | 34 | 14 |

## Checkpoint별 LSTM 성능 비교 (seed=42)

| Checkpoint | EventMinP | FN |
|---|---:|---:|
| val_loss (P30) | **0.9181** | 19 |
| val_video_min_pr | 0.9038 | 23 |
| val_event_min_pr | 0.8651 | 34 |

## 결론

1. **LSTM 최적 checkpoint = val_loss**: 후처리 기반 checkpoint일수록 LSTM 성능 저하
2. **GRU와 정반대 패턴**: GRU는 val_video_min_pr에서 최고(0.9241), LSTM은 동일 checkpoint에서 최저 수준
3. **LSTM이 GRU를 넘지 못함**: 어떤 checkpoint 조합도 P27-vm0(0.9241) 미달
4. **아키텍처 간 checkpoint 친화성 차이**: GRU ↔ val_video_min_pr / LSTM ↔ val_loss

## 전체 최고 기준

| 실험 | 아키텍처 | Checkpoint | EventMinP |
|---|---|---|---:|
| P27-vm0 | GRU(128,64) | val_video_min_pr | **0.9241** |
| P30-lstm | LSTM(128,64) | val_loss | 0.9181 |
| P30-gru | GRU(128,64) | val_loss | 0.9087 |

## 다음 단계 (Phase 32)

GRU val_loss 시드 탐색 + focal loss 제거 테스트 + GRU(256,128) val_loss.
