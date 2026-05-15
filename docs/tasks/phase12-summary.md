# Phase 12 Summary: LB-3 Falling/Fallen Split

작성일: 2026-05-16

## 목적

LB-2에서 falling과 fallen이 같은 positive로 묶여 decision boundary가 흐려진다고 보고, `label_3class`로 falling/fallen을 분리해 학습했다.

## 주요 변경

- `--label-column label_3class`
- `--num-classes 3`
- `--positive-labels 1,2`
- Phase 11에서 효과가 있었던 focal alpha=0.10을 유지

## 결과

| ID | 변경점 | Test MinPR | CM |
| --- | --- | ---: | --- |
| `P12-v01` | alpha=0.10 LB-2 pure baseline | 0.9061 | TN=222 FP=23 FN=20 TP=646 |
| `P12-v04` | LB-3 + hard-neg + kp12 | 0.8911 | TN=221 FP=16 FN=27 TP=647 |
| `P12-v03` | LB-3 + hard-neg | 0.8866 | TN=219 FP=18 FN=28 TP=646 |
| `P12-v02` | LB-3 only | 0.8821 | TN=217 FP=20 FN=29 TP=645 |

## 결론

LB-3는 FP를 줄였지만 FN을 크게 늘렸다. 목표인 FP<=19와 FN<=19를 동시에 만족하지 못했고, FP/FN trade-off가 개선되지 않아 폐기한다.
