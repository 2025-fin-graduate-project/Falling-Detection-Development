# Phase 11 Summary: Hard Negative and MinPR Checkpoint

작성일: 2026-05-16

## 목적

P9O-v01의 병목을 비낙상 FP로 보고, val_video_min_pr 체크포인트와 hard-negative 오버샘플링으로 NFall 방어력을 높이는지 확인했다.

## 기준점

| 기준 | Test MinPR | CM |
| --- | ---: | --- |
| `P9O-v01` | 0.9187 | TN=226 FP=19 FN=20 TP=646 |

## 주요 변경

- `--checkpoint-monitor val_video_min_pr`
- val FP 영상의 negative window stride를 1로 줄임
- `min_consecutive` 후보를 `1,2,3,4,5,7,9`로 확장

## 결과

| ID | 변경점 | Test MinPR | CM |
| --- | --- | ---: | --- |
| `P11-v05` | val_loss + hard-neg + focal alpha=0.10 | 0.9061 | TN=222 FP=23 FN=22 TP=644 |
| `P11-v02` | val_video_min_pr + hard-neg | 0.8939 | TN=219 FP=26 FN=22 TP=644 |
| `P11-v04` | val_loss + hard-neg 단독 | 0.8880 | TN=222 FP=23 FN=28 TP=638 |
| `P11-v01` | val_video_min_pr only | 0.8694 | TN=213 FP=32 FN=15 TP=651 |

## 결론

Hard-negative는 FP를 일부 줄였지만 P9O-v01의 FP=19를 넘지 못했다. `val_video_min_pr` 체크포인트는 val 성능은 좋아도 test 일반화가 약했다. Phase 11 best는 `P11-v05`지만 기준점보다 낮아 실패로 판단한다.
