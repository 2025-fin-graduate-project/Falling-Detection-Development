# Phase 21 Summary: val_event_min_pr Checkpoint Monitoring

작성일: 2026-05-16

## 목적

Phase 20에서 도입한 `--threshold-eval-level event` 기반 임계값 선택의 효과를 유지하면서, 학습 중 체크포인트 저장 기준도 `val_loss` → `val_event_min_pr`로 전환하여 성능 극대화를 목표로 함.

## 변경점

| 항목 | Phase 20 | Phase 21 |
|---|---|---|
| Checkpoint Monitor | `val_loss` | `val_event_min_pr` |
| Threshold Eval Level | `event` | `event` |
| Dataset | `splits_v2_class_balanced_filtered` | `splits_v2_class_balanced_filtered` |
| Model | GRU(128,64), kp7, 40f | GRU(128,64), kp7, 40f |
| Event Tolerance Windows | 2 | 2 |
| epochs/patience | 100/15 | 30/5 |

## 결과

| ID | 변경점 | Test Event MinPR | Test Video MinPR | Threshold / mc | Event CM |
| --- | --- | ---: | ---: | --- | --- |
| `P21-v01` | 기본 (neg_stride=2) | **0.9227** | **0.9267** | 0.525 / 3 | TN=215 FP=17 FN=18 TP=603 |
| `P21-v02` | neg_stride=1 | 0.8861 | 0.8936 | 0.475 / 3 | — |

## Phase 20 대비 성능 향상

| 실험 | Event MinPR | Video MinPR |
|---|---|---|
| P20-v01 (기준 — val_loss) | 0.9121 | 0.9160 |
| P21-v01 (val_event_min_pr) | **0.9227** | **0.9267** |
| 향상폭 | **+0.0106** | **+0.0107** |

## 중간 결론 (P21-v01 기준)

- `val_event_min_pr` 체크포인트 전환이 동일 아키텍처에서 event MinPR +1.06%p 향상을 가져옴.
- 현재까지 event-level MinPR **0.9227**이 전체 실험 중 최고값.
- **neg_stride=1은 event checkpoint와 궁합 불량**: P21-v02가 0.8861로 급락. patience=5 환경에서 전체 negative 노출 시 val_event_min_pr이 불안정해 조기종료가 과도하게 발생하는 것으로 추정. 이후 모든 실험에서 neg_stride=2 고정.
- **최적 구성 확정**: neg_stride=2 + val_event_min_pr checkpoint = 0.9227.

## 다음 단계

- P21-v02 재실행 (크래시 복구)
- Phase 22: kp7kv (velocity 피처) + val_event_min_pr 체크포인트 조합 확인
- Phase 23 (계획): GRU(256,128) + val_event_min_pr checkpoint + class_balanced_filtered

## 평가 기준

- `test_event_video.min_pr` (primary) — 이벤트 허용 윈도우=2
- `test_video.min_pr` (secondary)
- threshold_selection: val event 2D sweep (threshold × min_consecutive)
