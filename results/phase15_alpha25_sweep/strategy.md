# Phase 15 Strategy

**핵심 발견 (Phase 14 → 15 전환 이유)**:
- P9O-v01 (α=0.25, neg_stride=2, no hard-neg): FP=19✓ FN=20, MinPR=0.9187 ← 역대 최고
- Phase 11-14 전체 (α=0.10, hard-neg): FP=23 FN=20 at best (MinPR=0.9061) — 0.9187보다 낮음
- hard-neg은 FP를 19→23으로 오히려 악화. α=0.10 도입이 실제로 역효과.
- 결론: Phase 11-14는 잘못된 방향 4 Phase 낭비. α=0.25 + no hard-neg이 유일 최적.

**핵심 가설**: P9O-v01은 30에폭만 훈련. α=0.25 + no hard-neg + neg_stride=2를 100에폭(patience=15)으로 재훈련하면 더 나은 수렴 → FN 20→19 가능. α를 0.30~0.35로 미세 증가해 fall 감도 높이면 FN↓ 가능.

**P9O-v01 설정 (기준)**:
- α=0.25, neg_stride=2, no hard-neg, kp7, epochs=30, patience=5
- FP=19 FN=20 MinPR=0.9187 (thr=0.525, mc=3)

**Phase 11-14 실패 원인 정리**:
- hard-neg: FP 악화 (19→23). Phase 11에서 주요 원인.
- α=0.10: 낮은 threshold → FP ↑ (α=0.25→threshold=0.525, α=0.10→threshold=0.425)
- α=0.05: 불안정 (FP=30)
- neg_stride=1: FP=19 가능하지만 FN=26 (MinPR=0.8968)
- Phase 11-14 전체 MinPR ≤ 0.9061 < P9O-v01 0.9187

**후보 실험**:
| ID | 변경점 | 결과 |
|----|--------|------|
| v01 | α=0.25, neg_stride=2, no hard-neg, kp7, epochs=100 (P9O 재훈련) | MinPR=0.8816 (FP=29↑↑ FN=18, thr=0.425/mc=5 — 가설 반박: 100에폭 재훈련이 P9O-v01 재현 실패, 스코어 전반적 하락) |
| v02 | α=0.30, neg_stride=2, no hard-neg, kp7 | MinPR=0.8939 (FP=26 FN=16, thr=0.525/mc=4 — thr 복구됐으나 FP는 P9O-v01보다 여전히 높음, FN↓ 추세) |
| v03 | α=0.25, neg_stride=2, no hard-neg, kp12 (특징 풍부화) | MinPR=0.8694 (FP=32↑↑ FN=20, thr=0.475/mc=4 — kp12 오히려 최악, 추가 관절이 노이즈) |
| v04 | α=0.35, neg_stride=2, no hard-neg, kp7 | MinPR=0.8939 (FP=26 FN=20, thr=0.525/mc=3 — α↑ FN 개선 없음; v02(α=0.30, FN=16)가 명백히 우수) |

**기준점**:
- P9O-v01 (α=0.25, neg_stride=2): FP=19, FN=20, MinPR=0.9187
- 목표: FP≤19 AND FN≤19 → MinPR ≥ 0.92

**P15-v01 핵심 발견**:
- α=0.25 + 100에폭: thr=0.425/mc=5 선택, FP=29↑↑ — P9O-v01(30에폭, thr=0.525/mc=3) 재현 실패
- val 최적점이 thr=0.425로 이동함 → 모델 스코어 전반적으로 낮음 (100에폭 과적합 또는 다른 수렴)
- P9O-v01은 lucky seed 또는 짧은 훈련(30에폭+patience=5)의 정규화 효과 가능성
- v02(α=0.30)/v03(kp12)/v04(α=0.35) 결과 기다린 후 Phase 16 전략 수정 필요

**Phase 15 최종 결론**:
- 100에폭 훈련은 α 관계없이 FP≥26 — P9O-v01(FP=19) 재현 불가
- α=0.30(v02): FN=16 역대 최소이나 FP=26으로 NFallR 병목
- α=0.35(v04): FN=20 (v02보다 FN↑), α 증가가 FN↓를 보장하지 않음
- kp12(v03): FP=32로 최악 — 추가 관절이 노이즈
- 결론: 훈련 기간(30 vs 100에폭)이 α보다 훨씬 중요한 변인. Phase 16에서 epochs=30/patience=5 복귀 필수.

**종료 기준**: test MinPR ≥ 0.92
