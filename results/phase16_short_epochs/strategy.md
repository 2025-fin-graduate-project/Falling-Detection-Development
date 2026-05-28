# Phase 16 Strategy

**핵심 발견 (Phase 15 → 16 전환 이유)**:
- Phase 15 전체(100에폭): α 관계없이 FP≥26 — P9O-v01(FP=19) 재현 실패
- P9O-v01 조건: α=0.25, epochs=30, patience=5, seed=42 → FP=19, FN=20, MinPR=0.9187
- 훈련 기간(30에폭)이 핵심 정규화 요인으로 확인

**핵심 가설**:
1. epochs=30+patience=5 복귀로 P9O-v01 재현성 확인
2. α 미세 조정(0.20~0.35)으로 FN 20→19 달성 가능성
3. seed 다양화로 더 좋은 lucky seed 탐색

**P9O-v01 기준점**:
- α=0.25, neg_stride=2, no hard-neg, kp7, epochs=30, patience=5, seed=42
- FP=19 FN=20 MinPR=0.9187 (thr=0.525, mc=3)

**후보 실험**:
| ID | 변경점 | 결과 |
|----|--------|------|
| v01 | α=0.25, seed=42 (P9O-v01 재현) | MinPR=0.8980 (FP=25↑ FN=19✓, thr=0.450/mc=5 — FN 목표 달성! FP 25로 NFallR 병목. thr=0.450 vs P9O-v01 thr=0.525: 같은 조건인데 다른 모델 — checkpoint_monitor/초기화 차이?) |
| v02 | α=0.20, seed=42 | MinPR=0.8980 (FP=25 FN=20, thr=0.425/mc=4 — v01보다 FN 나쁨. α 낮으면 thr도 낮아짐) |
| v03 | α=0.30, seed=42 | - |
| v04 | α=0.35, seed=42 | - |
| v05 | best_α, seed=0 | - |
| v06 | best_α, seed=1 | - |
| v07 | best_α, seed=7 | - |
| v08 | best_α, seed=123 | - |
| v09 | α=0.25, seed=42, mc=[1,3,5,7,9] (P9O-v01 exact grid) | - |
| v10 | best_α, seed=42, mc=[1,3,5,7,9] | - |
| v11 | best_α, seed=0, mc=[1,3,5,7,9] | - |

**P16-v01 핵심 발견**:
- FN=19 달성! (P9O-v01 FN=20보다 개선) — 그러나 FP=25로 NFallR 병목
- thr=0.450/mc=5 (vs P9O-v01 thr=0.525/mc=3) — 다른 val 최적점 선택
- P9O-v01과 차이: min_consecutive_values=[1,3,5,7,9] vs 현재 [1,2,3,4,5,7,9]
  → mc=2/4 추가로 val optimizer가 다른 (더 낮은 thr의) 최적점 선택 가능성
- 추가 탐색 필요: P9O-v01 exact mc grid [1,3,5,7,9] 재현 실험 (v09 예정)

**종료 기준**: test MinPR ≥ 0.92
