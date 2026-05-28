# Phase 20 Strategy

**핵심 가설**: 기존 데이터셋 분할(`splits_v2`)은 비디오 단위 분할이나 특정 방향 또는 라벨이 Train/Val/Test에 불균형하게 포함되었을 가능성이 있다. 또한, 평가 지표를 `video-level`에만 의존하면 낙상 영상 내의 부정확한 구간 감지도 TP로 처리되는 문제가 있다. 데이터셋을 계층화하여 다시 나누고, 임계값 선택을 `event-level`로 수행하면 더 강건하고 실제적인 성능을 얻을 수 있다.

**이전 Phase 분석**:
- Phase 11~19: `splits_v2_filtered` 사용. alpha, neg_stride, 모델 크기 등을 실험했으나 MinPR 0.9187의 벽을 넘기 어려웠음.
- 지표의 한계: `video-level` MinPR은 낙상 비디오에서 한 프레임이라도 positive가 나오면 성공으로 간주하므로, 실제 낙상 발생 시점과의 일치성을 보장하지 못함.

**주요 변경점**:
- **데이터셋 재분할**: `direction`, `video_label`을 기준으로 계층화 분할 (splits_v2_class_balanced).
- **이벤트 기반 평가**: `--threshold-eval-level event` 옵션을 사용하여, 낙상 이벤트 구간(event window) 내에서 감지가 일어났는지를 기준으로 임계값과 연속 감지 횟수(min_consecutive)를 최적화.
- **neg_stride 실험**: 데이터셋이 변경되었으므로 neg_stride=1 vs 2의 효과를 재검증.

**후보 실험**:
| ID | 변경점 | Test Event MinPR | 결과 |
|----|--------|------------------|------|
| `P20-v01` | GRU(128,64), alpha=0.25, neg_stride=2 | 0.9121 | 기준 모델. 안정적인 성능 (FP=14, FN=21). |
| `P20-v02` | alpha=0.15 | 0.8934 | alpha 감소 시 FN 증가(26)로 성능 저하. |
| `P20-v03` | neg_stride=1 | **0.9142** | FP는 약간 증가(19)하나 FN 감소(20) 및 MinPR 개선. **Best**. |
| `P20-v04` | GRU(96,48) | 0.8966 | 모델 경량화 시 성능 손실 확인. |

**Phase 20 최종 발견**:
- 새 데이터셋 분할에서도 neg_stride=1(`P20-v03`)이 이벤트 레벨 MinPR에서 가장 우수함을 확인.
- `event-level` 임계값 선택은 실제 서비스 환경에 더 가까운 지표를 제공함.
- `val_loss` 기반 체크포인트 선택은 여전히 `event-level` 성능과 괴리가 있을 수 있음.

**다음 단계**:
- Phase 21: 체크포인트 모니터링 지표를 `val_event_min_pr`로 변경하여 학습 과정 자체를 이벤트 성능에 최적화.
