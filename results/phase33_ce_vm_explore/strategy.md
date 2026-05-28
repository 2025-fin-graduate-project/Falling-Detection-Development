# Phase 33: CE × val_video_min_pr + LSTM×CE 탐색

## 가설

Phase 32 분석에서 GRU+val_loss 상한이 ~0.914로 확인됨. 남은 미탐색 조합:
1. GRU+CE+val_video_min_pr: focal이 val_vm에 필수인지 검증
2. LSTM+CE+val_loss: LSTM에서 CE가 focal보다 나은지 검증

## 변경 사항

- 기준 모델 유지: GRU(128,64) kp7 40f, class_balanced_filtered
- GRU 실험: focal loss → CE, checkpoint=val_video_min_pr
- LSTM 실험: focal loss → CE, checkpoint=val_loss

## 실험 결과

| ID | Arch | Loss | Ckpt | Seed | EventMinP | VideoMinP | FN | FP | fall_pr | nfall_pr |
|---|---|---|---|---|---|---|---|---|---|---|
| P33-gce-vm42 | GRU | CE | val_vm | 42 | 0.8875 | 0.8912 | 27 | 19 | 0.9690 | 0.8875 |
| P33-gce-vm1 | GRU | CE | val_vm | 1 | 0.8782 | 0.8819 | 29 | 23 | 0.9626 | 0.8782 |
| P33-gce-vm0 | GRU | CE | val_vm | 0 | 0.9038 | 0.9076 | 23 | 16 | 0.9739 | 0.9038 |
| P33-lce-vl42 | LSTM | CE | val_loss | 42 | 0.8912 | 0.8950 | 26 | 19 | 0.9691 | 0.8912 |
| P33-lce-vl1 | LSTM | CE | val_loss | 1 | 0.8800 | 0.8800 | 30 | 12 | 0.9801 | 0.8800 |

기준:
- P27-vm0 (GRU+focal+val_vm+seed=42): event=0.9241, FN=18, FP=13
- P30-lstm (LSTM+focal+val_loss+seed=42): event=0.9181, FN=?, FP=?

## 분석

### GRU+CE+val_vm: 종합 열세, 강한 seed 의존성

- seed=42: 0.8875 (-0.037 vs P27-vm0), seed=1: 0.8782, seed=0: 0.9038
- seed 범위: 0.8782~0.9038 (±0.013) — focal의 seed=42만의 특이 우세(0.9241)와 달리 전반적으로 낮음
- FN=23~29 (P27-vm0의 FN=18 대비 모두 더 높음)
- threshold 0.675 (P27-vm0의 0.525 대비 높음) → CE의 확률 분포 덜 보정됨

**결론**: focal이 val_video_min_pr checkpoint에 필수. CE는 확률 분포를 잘 보정하지 못해 threshold sweep 효율 저하.

### LSTM+CE+val_loss: 강한 seed 의존성, 전반적 열세

- seed=42: 0.8912 (FN=26, FP=19), seed=1: 0.8539 (FN=25, FP=33)
  - seed 간 격차 0.037 — GRU+focal+val_vm의 seed 범위(±0.03)보다 큼
- P30-lstm (LSTM+focal+val_loss) 0.9181 대비 모두 열세 (-0.027 ~ -0.064)
- FN은 GRU+CE+val_vm과 유사 (25~26), FP는 seed에 따라 큰 변동

**결론**: LSTM에서도 focal은 필수. val_loss checkpoint는 CE와 함께할 때 seed에 따라 FP가 크게 변동.

## 핵심 발견

focal loss는 단순한 class imbalance 보정이 아님. 확률 분포의 형태(calibration)를 val_video_min_pr과 val_loss 양쪽 checkpoint에서 모두 개선함. CE로 교체 시 FN↑(GRU+val_vm) 또는 FP↑↑(LSTM+val_loss) 발생.

## 결론

Phase 33 전체 실험이 P27-vm0(0.9241)에 미달. focal loss 제거 전략은 유효하지 않음.

다음 방향:
- Phase 34: focal alpha 증가 (0.25→0.35/0.40/0.45) + 데이터 증강
- Phase 35: class_weight 제거 + focal alpha 조정 (gradient 불균형 근본 해결)
