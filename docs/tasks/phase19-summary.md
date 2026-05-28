# Phase 19 Summary: Larger GRU

작성일: 2026-05-16

## 목적

GRU(256,128)로 모델 용량을 키우면 GRU(128,64)의 한계를 넘을 수 있는지 확인했다.

## 주요 변경

- GRU units: 128,64 -> 256,128
- kp7, 40f, conv-pre, focal loss 계열 유지
- 30epoch/100epoch, alpha=0.25/0.30, seed 변형 비교

## 결과

| ID | 변경점 | Threshold / mc | Test Video MinPR | Test Event MinPR | CM |
| --- | --- | --- | ---: | ---: | --- |
| `P19-v02` | GRU(256,128), 100epoch, alpha=0.25 | 0.575 / 3 | 0.9020 | 0.9020 | TN=221 FP=24 FN=23 TP=643 |
| `P19-v03` | GRU(256,128), alpha=0.30 | 0.575 / 2 | 0.8857 | 0.8857 | TN=217 FP=28 FN=26 TP=640 |
| `P19-v04` | GRU(256,128), seed=0 | 0.500 / 4 | 0.8816 | 0.8816 | TN=216 FP=29 FN=20 TP=646 |
| `P19-v01` | GRU(256,128), 30epoch | 0.525 / 3 | 0.8776 | 기록 없음 | TN=215 FP=30 FN=25 TP=641 |

## 결론

모델 크기 증가는 해결책이 아니다. 최고점인 `P19-v02`도 Test MinPR 0.9020으로 P9O-v01 0.9187보다 낮다. P19는 event metric 출력은 일부 포함하지만, threshold 선택은 여전히 `eval_level=video` 기준이다.
