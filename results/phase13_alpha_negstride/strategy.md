# Phase 13 Strategy

**핵심 가설**: Phase 11/12 분석으로 focal α=0.10이 핵심 개선임을 확인. 남은 병목은 FP=23(목표≤19)와 FN=20(목표≤19) 동시 해결. α=0.10 + train_negative_stride=1 조합은 미시험 — neg_stride=1으로 non-fall 윈도우 2배 노출시 FP↓ 기대, α=0.10으로 FN 방어. α=0.05(더 강한 non-fall 가중치)도 탐색.

**이전 Phase 실패 원인**:
- Phase 11: hard-neg 단독으로 FP 23 plateau. α=0.10으로 FN 개선(22→20)이 최선.
- Phase 12: LB-3가 FP를 줄이지만 FN을 늘림(20→27) — fallen(2) 포함이 threshold 과보수화.
- 공통: FP≤19 AND FN≤19를 동시에 달성하는 조합 미발견.

**주요 변경점**:
- α=0.10 확정 유지 (P11-v05, P12-v01에서 일관 효과)
- `--train-negative-stride 1`: 전체 non-fall 윈도우 2배 노출 (stride=2→1)
- α=0.05 탐색: non-fall 손실 더 강화 (단, FN 증가 리스크 있음)
- hard-neg 유지

**후보 실험**:
| ID | 변경점 | 결과 |
|----|--------|------|
| v01 | α=0.10 + hard-neg + neg_stride=1 (미시험 핵심 조합) | MinPR=0.8968 (FP=19✓ FN=26↑↑, NFallP bottleneck — sweep에 NFallP≥0.90&NFallR≥0.90 구간 없음) |
| v02 | α=0.05 + hard-neg (더 강한 non-fall 가중치) | MinPR=0.8776 (FP=30↑↑, thr=0.325 — α 낮추면 불안정, 역효과) |
| v03 | α=0.05 + hard-neg + neg_stride=1 | MinPR=0.8776 (FP=30↑ FN=25 — v02와 동일, α=0.05 방향 완전 폐기) |
| v04 | α=0.10 + hard-neg + noise_std=0.05 (강한 augmentation) | MinPR=0.8857 (FP=28, FN=25 — noise 부분 개선, v01 미만) |

**베이스라인**: P11-v05/P12-v01 test MinPR=0.9061 (FP=23, FN=20)
**목표**: test MinPR ≥ 0.92 (FP≤19 AND FN≤19 동시)

**종료 기준**: test MinPR ≥ 0.92
