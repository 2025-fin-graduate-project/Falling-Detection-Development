# Phase 15 Summary: Return to P9O-v01 Direction

작성일: 2026-05-16

## 목적

Phase 11~14의 hard-negative와 낮은 alpha 방향이 실패했으므로, P9O-v01의 alpha=0.25, no hard-neg, neg_stride=2 조건으로 돌아가 100epoch 재훈련과 alpha 미세 조정을 확인했다.

## 주요 변경

- hard-negative 제거
- alpha=0.25/0.30/0.35 비교
- 100epoch, patience=15
- kp12 비교

## 결과

| ID | 변경점 | Test MinPR | CM |
| --- | --- | ---: | --- |
| `P15-v02` | alpha=0.30, 100epoch | 0.8939 | TN=219 FP=26 FN=16 TP=650 |
| `P15-v04` | alpha=0.35, 100epoch | 0.8939 | TN=219 FP=26 FN=20 TP=646 |
| `P15-v01` | alpha=0.25, 100epoch | 0.8816 | TN=216 FP=29 FN=18 TP=648 |
| `P15-v03` | kp12 | 0.8694 | TN=213 FP=32 FN=20 TP=646 |

## 결론

100epoch 재훈련은 FP를 크게 악화시켰다. alpha보다 훈련 길이와 정규화 효과가 더 큰 변인으로 보이며, P9O-v01의 짧은 훈련 조건을 Phase 16에서 다시 확인하기로 했다.
