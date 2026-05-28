# Phase 22 Summary: kp7kv (Velocity Features) + val_event_min_pr Checkpoint

작성일: 2026-05-16

## 목적

Phase 17에서 탐색한 velocity 피처(kp7kv — 관절 위치 + 속도)와 Phase 21에서 확립한 `val_event_min_pr` 체크포인트를 조합하여 동적 특성 학습이 event MinPR을 0.93 이상으로 끌어올릴 수 있는지 검증.

## 주요 변경

| 항목 | Phase 21 | Phase 22 |
|---|---|---|
| Feature set | kp7 (27 features) | **kp7kv (37 features = kp7 + velocity)** |
| Dataset | splits_v2_class_balanced_filtered | splits_v2_filtered_kv (KV 피처 포함 전용 분할) |
| Checkpoint Monitor | val_event_min_pr | val_event_min_pr |
| Threshold Eval Level | event | event |
| Model | GRU(128,64), 40f | GRU(128,64), 40f |

## 결과

| ID | 변경점 | Test Event MinPR | Test Video MinPR | Threshold / mc |
| --- | --- | ---: | ---: | --- |
| `P22-v01` | kp7kv, neg_stride=2 | 0.8889 | 0.8926 | 0.550 / 2 |
| `P22-v02` | kp7kv, neg_stride=1 | 0.8974 | 0.9013 | 0.525 / 3 |

## 결론

- **kp7kv 피처는 효과 없음**: P22-v01이 0.8889로 P21-v01(kp7, 0.9227) 대비 **-0.034** 하락.
- kp0_vy/vx 등 per-joint velocity는 이미 kp7 피처셋에 포함된 VHSSC/AHSSC(center-of-mass 속도·가속도)와 중복 신호가 많아 노이즈로 작용.
- 피처 수 증가(27→37)가 GRU 학습을 오히려 방해. **kp7 고정이 최적.**
- 이후 실험에서 kv 피처 계열 전량 제외.

## 평가 기준

- `test_event_video.min_pr` (primary) — 이벤트 허용 윈도우=2
- `test_video.min_pr` (secondary)
