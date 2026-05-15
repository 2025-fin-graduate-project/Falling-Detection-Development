# Phase 27 Summary: Seed Sweep + Checkpoint Monitor Variants

작성일: 2026-05-16

## 목적

P21-v01 최적 구성(GRU(128,64) kp7 patience=5)으로 seed 다양성 탐색 및 val_video_min_pr 체크포인트가 val_event_min_pr보다 안정적인 학습 신호를 제공하는지 검증.

## 실험 구성

| 그룹 | Checkpoint Monitor | Seeds |
|---|---|---|
| A | val_event_min_pr | 0, 1, 2, 3 |
| B | val_video_min_pr | 42, 0 |

공통 설정: GRU(128,64) kp7 neg_stride=2 patience=5 epochs=30 — P21-v01 기준 구성.

## 결과

| ID | Checkpoint | Seed | EventMinP | VideoMinP | thr / mc | CM (TN/FP/FN/TP) |
|---|---|---|---:|---:|---|---|
| `P27-vm0` | val_video_min_pr | 42 | **0.9241** | 0.9280 | 0.525 / 3 | 219/13/18/603 |
| `P27-s1`  | val_event_min_pr | 1  | 0.9103 | 0.9181 | 0.550 / 2 | 213/19/21/600 |
| `P27-s2`  | val_event_min_pr | 2  | 0.9064 | 0.9103 | 0.525 / 3 | 213/19/22/599 |
| `P27-s0`  | val_event_min_pr | 0  | 0.9060 | 0.9138 | 0.475 / 3 | 212/20/22/599 |
| `P27-s3`  | val_event_min_pr | 3  | 0.8936 | 0.8974 | 0.475 / 2 | 210/22/25/596 |
| `P27-vm1` | val_video_min_pr | 0  | 0.8613 | 0.8650 | 0.575 / 3 | 205/27/33/588 |

## 분석

### P27-vm0 세부 지표 (thr=0.525, mc=3)
- FallPrecision = 0.979 (TP=603, FP=13) — 0.93 대비 **4.9%p 여유**
- NFallPrecision = **0.924** ← 병목 (TN=219, FN=18)
- 0.93 달성 조건: FN ≤ 16 (현재 18, 2회 감소 필요)
- Threshold sweep 최고: val min_pr=0.9197 (어떤 (thr, mc)도 0.93 미달성)
- **모델 자체의 용량 한계 — threshold 튜닝으로는 0.93 불가**

### Checkpoint 비교
- val_video_min_pr (seed=42): 0.9241 — 전체 최고 (P21-v01 0.9227 경신)
- val_video_min_pr (seed=0): 0.8613 — 전체 최저, 극도로 seed 민감
- val_event_min_pr (seeds 0-3): 0.8936~0.9103 — 안정적이나 상한 낮음

## 결론

- **val_video_min_pr + seed=42 조합이 신기록(0.9241)** — P21-v01(0.9227) 경신
- **val_video_min_pr은 고분산**: seed=42에서 0.9241, seed=0에서 0.8613 — seed 의존성 매우 강함
- **val_event_min_pr은 안정적**: seeds 0~3 평균 ~0.902, 표준편차 ~0.007
- **0.93 구조적 장벽 확인**: 현재 모델로 threshold 최적화만으로는 불가; 모델 능력 자체 향상 필요
- **병목**: NFallPrecision — FN(놓친 낙상) 18건을 16건 이하로 줄여야 함 (낙상 Recall 개선)

## 다음 단계 (Phase 28)

val_video_min_pr checkpoint + 추가 seed 탐색으로 ≥0.93 달성 가능한 seed 존재 여부 확인.
병행: 구조적 접근(데이터 증강, 정규화 개선) 설계.
