# Phase 16 Summary: Short Epoch Reproduction

작성일: 2026-05-16

## 목적

P9O-v01의 30epoch/patience=5 조건이 핵심 정규화 요인인지 확인하고, seed/alpha 변화를 통해 FP와 FN 동시 개선 가능성을 탐색했다.

## 주요 변경

- 30epoch, patience=5로 복귀
- alpha=0.20/0.25/0.30/0.35 비교
- seed 탐색

## 결과

| ID | 변경점 | Test MinPR | CM |
| --- | --- | ---: | --- |
| `P16-v01` | alpha=0.25, seed=42 | 0.8980 | TN=220 FP=25 FN=19 TP=647 |
| `P16-v02` | alpha=0.20, seed=42 | 0.8980 | TN=220 FP=25 FN=20 TP=646 |
| `P16-v06` | seed 탐색 | 0.8898 | TN=218 FP=27 FN=20 TP=646 |
| `P16-v04` | alpha/seed 변형 | 0.8816 | TN=216 FP=29 FN=18 TP=648 |

## 결론

FN=19는 달성했지만 FP=25로 실패했다. 같은 계열 설정에서도 P9O-v01이 재현되지 않아 seed, checkpoint, threshold grid, split 분포 민감성이 크다는 신호로 판단한다.
