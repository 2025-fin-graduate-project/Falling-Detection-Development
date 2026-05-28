# Phase 14 Strategy

**핵심 가설**: Phase 13에서 α=0.10+neg_stride=1이 FP=19(✓) FN=26(↑)를 달성. 목표는 FP≤19 AND FN≤19 동시 달성. α=0.15를 쓰면 fall class 가중치가 0.10보다 높아져 FN을 줄이면서, neg_stride=1이 FP를 억제한다. α=0.10↔α=0.25 중간값 탐색.

**이전 Phase 실패 원인**:
- Phase 13: neg_stride 조절은 FP↓FN↑ 딜레마만 강화, α=0.05는 역효과.
- 최고 달성: α=0.10+neg_stride=1 → FP=19(✓) FN=26(↑) MinPR=0.8968.
- α=0.10이 FP 방향으로 과도하게 bias됨.

**주요 변경점**:
- α=0.15 (0.10보다 fall 가중치 높음 → FN↓ 기대)
- neg_stride=1 유지 (FP=19 달성 유지 목적)
- α=0.20 추가 탐색 (α=0.25는 FP 증가 확인됨, 안전 마진 고려)

**후보 실험**:
| ID | 변경점 | 결과 |
|----|--------|------|
| v01 | α=0.15 + hard-neg + neg_stride=1 | MinPR=0.8943 (FP=25↑ FN=26, thr=0.425/mc=4 — 가설 반박: α↑은 neg_stride=1에서 FP↑만 유발, FN 불변) |
| v02 | α=0.20 + hard-neg + neg_stride=1 | MinPR=0.8776 (FP=30↑↑ FN=25, thr=0.500/mc=3 — α↑ FP 악화 패턴 확인) |
| v03 | α=0.15 + hard-neg + neg_stride=1 + noise=0.05 | MinPR=0.8735 (FP=31↑↑ FN=26, thr=0.525/mc=1 — noise 증가가 FP 악화, 최악 결과) |
| v04 | α=0.15 + hard-neg + neg_stride=2 (neg_stride 비교) | MinPR=0.9061 (FP=23 FN=18✓, thr=0.425/mc=4 — neg_stride=2에서 α=0.15는 P9O-v01 α=0.25보다 FP 악화: hard-neg 부정적 영향 재확인) |

**Phase 14 최종 발견**:
- α=0.15+neg_stride=1: FP=25 FN=26 (α=0.10+neg_stride=1보다 FP 악화)
- α=0.15+neg_stride=2 (v04): FP=23 FN=18, MinPR=0.9061 — FN=18은 역대 최소이나 FP=23으로 NFallR bottleneck
  - thr=0.425 (vs P9O-v01 thr=0.525): hard-neg이 threshold를 낮춤 → FP 악화
  - P9O-v01 (α=0.25, no hard-neg, neg_stride=2): FP=19, thr=0.525 — hard-neg 없어야 높은 threshold 달성
- 결론: **hard-neg은 neg_stride=2에서도 FP를 악화(19→23)시킴**. α=0.25+no hard-neg이 유일 최적 경로.

**기준점**:
- P12-v01/P11-v05 (α=0.10, neg_stride=2): FP=23, FN=20, MinPR=0.9061
- P13-v01 (α=0.10, neg_stride=1): FP=19, FN=26, MinPR=0.8968
- 목표: FP≤19 AND FN≤19 → MinPR ≥ 0.92

**종료 기준**: test MinPR ≥ 0.92
