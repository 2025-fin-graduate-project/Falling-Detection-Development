# Phase 13 Summary: Alpha and Negative Stride

작성일: 2026-05-16

## 목적

alpha=0.10으로 fall recall을 방어하고, `train_negative_stride=1`로 non-fall window 노출을 늘려 FP를 낮출 수 있는지 확인했다.

## 주요 변경

- focal alpha=0.10 유지
- 전체 non-fall window stride를 2에서 1로 변경
- alpha=0.05, stronger noise도 비교

## 결과

| ID | 변경점 | Test MinPR | CM |
| --- | --- | ---: | --- |
| `P13-v01` | alpha=0.10 + hard-neg + neg_stride=1 | 0.8968 | TN=226 FP=19 FN=26 TP=640 |
| `P13-v04` | alpha=0.10 + hard-neg + noise=0.05 | 0.8857 | TN=217 FP=28 FN=25 TP=641 |
| `P13-v02` | alpha=0.05 + hard-neg | 0.8776 | TN=215 FP=30 FN=25 TP=641 |
| `P13-v03` | alpha=0.05 + hard-neg + neg_stride=1 | 0.8776 | TN=215 FP=30 FN=25 TP=641 |

## 결론

`P13-v01`은 FP=19를 달성했지만 FN=26으로 fall recall이 무너졌다. negative exposure를 늘리면 FP는 줄 수 있지만 FN 증가가 너무 커서 목표 MinPR에는 불리하다.
