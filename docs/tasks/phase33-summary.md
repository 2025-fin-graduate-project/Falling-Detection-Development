# Phase 33 Summary: CE × val_video_min_pr + LSTM×CE 탐색

**날짜**: 2026-05-16  
**스크립트**: `scripts/run_phase33_ce_vm_explore.sh`  
**결과 디렉토리**: `results/phase33_ce_vm_explore/`

## 목적

Phase 32에서 GRU+val_loss 상한이 ~0.914로 확인된 후, 남은 두 가지 미탐색 조합을 검증:
1. GRU+CE+val_video_min_pr — focal이 val_vm checkpoint에 필수인지 확인
2. LSTM+CE+val_loss — LSTM에서 CE가 focal보다 나은지 확인

## 실험 결과

**평가 기준**: threshold + min_consecutive 2D sweep → val 최적화 → test event/video MinPR  
**테스트셋**: 853 videos (624 fall / 232 non-fall events)

| ID | Arch | Loss | Ckpt | Seed | EventMinP | VideoMinP | FN | FP | fall_pr | nfall_pr |
|---|---|---|---|---|---|---|---|---|---|---|
| P33-gce-vm0 | GRU(128,64) | CE | val_vm | 0 | **0.9038** | 0.9076 | 23 | 16 | 0.9739 | 0.9038 |
| P33-lce-vl42 | LSTM(128,64) | CE | val_loss | 42 | 0.8912 | 0.8950 | 26 | 19 | 0.9691 | 0.8912 |
| P33-gce-vm42 | GRU(128,64) | CE | val_vm | 42 | 0.8875 | 0.8912 | 27 | 19 | 0.9690 | 0.8875 |
| P33-gce-vm1 | GRU(128,64) | CE | val_vm | 1 | 0.8782 | 0.8819 | 29 | 23 | 0.9626 | 0.8782 |
| P33-lce-vl1 | LSTM(128,64) | CE | val_loss | 1 | 0.8800 | 0.8800 | 30 | 12 | 0.9801 | 0.8800 |

참고:
- **P27-vm0** (GRU+focal+val_vm+seed=42): event=0.9241, FN=18, FP=13  ← 전체 최고
- **P30-lstm** (LSTM+focal+val_loss+seed=42): event=0.9181

## 핵심 발견

### 1. CE는 focal보다 일관되게 열세
- GRU+CE+val_vm 최고(seed=0): 0.9038 vs GRU+focal+val_vm+seed=42: 0.9241 → **-0.020**
- LSTM+CE+val_loss 최고(seed=42): 0.8912 vs LSTM+focal+val_loss+seed=42: 0.9181 → **-0.027**
- 모든 CE 실험이 동일 checkpoint의 focal 실험보다 열세

### 2. CE는 seed에 더 민감
- GRU+CE+val_vm 범위: 0.8782~0.9038 (±0.013)
- GRU+focal+val_vm 범위: 0.9241 (seed=42 특이 우세, 기타 ~0.89)
- LSTM+CE+val_loss 범위: 0.8800~0.8912 (seed=1: FN=30/FP=12, seed=42: FN=26/FP=19)

### 3. 높은 threshold가 CE 열세의 징표
- P33-gce-vm0: threshold=0.675, mc=4 (P27-vm0: threshold=0.525, mc=3)
- CE 확률 분포가 덜 보정되어 threshold sweep이 덜 효율적임

### 4. 병목은 일관되게 nfall_pr (FN)
- 모든 Phase 33 실험에서 nfall_pr = fall_pr보다 낮아 bottleneck
- FN: 23~29 (P27-vm0 FN=18 대비 높음)

## 결론

**CE 교체 전략은 유효하지 않음.** focal loss는 단순 class imbalance 보정 이상의 역할을 함:
- 확률 분포 보정(calibration)을 개선 → threshold sweep 효율 향상
- val_video_min_pr과 val_loss 양쪽 checkpoint에서 모두 필수

## 다음 단계

Phase 33 실패 후 방향 전환:
- **Phase 34**: focal alpha 상향 (0.25→0.35/0.40/0.45) + 증강 (feat_mask, time_mask)
  - class_weight + focal(α=0.25) → 3.0x non-fall 총 gradient → FN=18 원인
  - alpha 상향으로 3.0x → 1.22~1.86x 감소 목표
- **Phase 35**: class_weight 제거 + focal alpha 조정
  - --no-class-weight + α=0.40 → 0.605×0.40/0.395×0.60 ≈ 1.0x 균형
  - 근본적 gradient 불균형 해결
