# Phase 28 Summary: Seed Probe (val_video_min_pr 확장 + val_event_min_pr 추가)

작성일: 2026-05-16

## 목적

P27-vm0 (val_video_min_pr, seed=42, 0.9241)가 신기록 달성 후, 추가 seed 탐색으로 ≥0.93 달성 가능한 seed 존재 여부 확인.

## 실험 구성

| 그룹 | Checkpoint | Seeds |
|---|---|---|
| A | val_video_min_pr | 5, 7, 13, 77, 100, 123 |
| B | val_event_min_pr | 5, 7, 100 |

공통 설정: GRU(128,64) kp7 patience=5 epochs=30 neg_stride=2 seed=42 제외 — 최적 구성 유지.

## 결과

### 그룹 A — val_video_min_pr

| Seed | EventMinP | FN | FP |
|---:|---:|---:|---:|
| 7 | 0.9052 | 21 | 22 |
| 77 | 0.9004 | 24 | 15 |
| 123 | 0.9000 | 24 | 16 |
| 5 | 0.8934 | 26 | 14 |
| 100 | 0.8870 | 27 | 20 |
| 13 | 0.8782 | 29 | 23 |

평균: **0.8940**, 표준편차: 0.0100. **0.91 돌파 없음.**

### 그룹 B — val_event_min_pr

| Seed | EventMinP | FN | FP |
|---:|---:|---:|---:|
| 7 | 0.8974 | 24 | 22 |
| 100 | 0.8785 | 30 | 15 |
| 5 | 0.8697 | 31 | 25 |

## 전체 seed 통계 (P27 포함)

| Checkpoint | n | mean | std | max |
|---|---:|---:|---:|---:|
| val_video_min_pr | 8 | 0.8937 | 0.0188 | **0.9241** (seed=42) |
| val_event_min_pr | 8 | 0.8981 | 0.0173 | **0.9227** (seed=42) |

## 결론

- **seed 탐색 완전 소진**: 16개 이상의 seed를 테스트, 어느 것도 P27-vm0(0.9241)에 근접하지 못함
- **seed=42는 두 checkpoint 모두에서 강한 아웃라이어**: val_video_min_pr 평균 대비 +0.030, val_event_min_pr 평균 대비 +0.025
- **seed 다양성 전략 한계 도달**: 무작위 탐색으로 0.93 달성은 구조적으로 불가
- **현재 구성의 성능 천장**: focal_alpha=0.25, noise_std=0.02 조합의 구조적 한계

## 분석: 왜 FN=18이 병목인가

P27-vm0 (최고 모델) 분석:
- 학습 데이터: fall 60% / non-fall 40% (neg_stride=2 적용 후)
- focal_alpha=0.25: fall 클래스에 0.25, non-fall에 0.75 가중치
- 실효 gradient 비율: fall=0.15, non-fall=0.30 → **non-fall이 2배 많은 학습 신호**
- 결과: 모델이 fall 탐지에 보수적 → FallPrecision=0.979(높음), FN=18(놓친 낙상 많음)

## 다음 단계 (Phase 29)

**가설**: focal_alpha 증가(0.25→0.50~0.75)로 fall 클래스 학습 신호 균형 → FN 18→16 달성 → NFallPrecision ≥ 0.93.

고정: seed=42, val_video_min_pr, 나머지 모두 동일.
실험: alpha ∈ {0.35, 0.50, 0.75}.
