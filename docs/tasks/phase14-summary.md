# Phase 14 Summary: Alpha Midpoint Sweep

작성일: 2026-05-16

## 목적

Phase 13에서 alpha=0.10은 non-fall 쪽으로 과도하게 치우쳤다. alpha=0.15/0.20으로 중간 균형점을 찾으려 했다.

## 주요 변경

- alpha=0.15, 0.20 탐색
- neg_stride=1과 2 비교
- hard-negative 유지 여부의 영향을 확인

## 결과

| ID | 변경점 | Test MinPR | CM |
| --- | --- | ---: | --- |
| `P14-v04` | alpha=0.15 + hard-neg + neg_stride=2 | 0.9061 | TN=222 FP=23 FN=18 TP=648 |
| `P14-v01` | alpha=0.15 + hard-neg + neg_stride=1 | 0.8943 | TN=220 FP=25 FN=26 TP=640 |
| `P14-v02` | alpha=0.20 + hard-neg + neg_stride=1 | 0.8776 | TN=215 FP=30 FN=25 TP=641 |
| `P14-v03` | alpha=0.15 + hard-neg + neg_stride=1 + noise=0.05 | 0.8735 | TN=214 FP=31 FN=26 TP=640 |

## 결론

`P14-v04`는 FN=18까지 낮췄지만 FP=23으로 NFall recall이 병목이 됐다. hard-negative가 threshold를 낮추고 FP를 악화시키는 패턴이 확인됐다.
