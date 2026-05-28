# Phase 18 Summary: TCN

작성일: 2026-05-16

## 목적

GRU 대신 TCN이 낙상 시퀀스의 시간 패턴을 더 명시적으로 잡을 수 있는지 확인했다.

## 주요 변경

- TCN residual causal dilated Conv1D
- kp7, 40f 계열
- focal alpha=0.25/0.30 후보

## 결과

| ID | 변경점 | Test MinPR | CM |
| --- | --- | ---: | --- |
| `P18-v04` | TCN 계열 best | 0.8694 | TN=213 FP=32 FN=28 TP=638 |
| `P18-v05` | TCN 계열 | 0.8490 | TN=208 FP=37 FN=22 TP=644 |

## 결론

현재 구성의 TCN은 GRU보다 낮다. FP가 크게 늘고 MinPR이 낮아졌으므로, 이 문제에서는 TCN 전환이 우선순위가 아니다.
