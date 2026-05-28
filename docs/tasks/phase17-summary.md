# Phase 17 Summary: Feature Hypotheses

작성일: 2026-05-16

## 목적

기존 kp7 + engineered feature가 놓치는 관절별 운동 패턴이나 카메라 위치 편차를 보완할 수 있는지 확인했다.

## 17A: Per-Keypoint Velocity

**가설**: HSSC 집계 속도만으로는 어깨/엉덩이 등 개별 관절의 비대칭 낙상 패턴을 놓친다. 주요 keypoint의 `vy/vx`를 추가하면 FN을 줄일 수 있다.

| ID | 변경점 | Test MinPR | CM |
| --- | --- | ---: | --- |
| `P17-v01` | kp7kv + alpha=0.25, 30epoch | 0.9020 | TN=221 FP=24 FN=23 TP=643 |
| `P17-v02` | kp7kv + 100epoch | 0.8939 | TN=219 FP=26 FN=18 TP=648 |

**판단**: velocity feature는 일부 신호가 있지만 P9O-v01보다 낮다.

## 17B: Pose-Relative / New Hypothesis

**가설**: 절대 좌표 대신 몸통 기준 상대 좌표와 자세 기하량을 쓰면 카메라/위치 편차를 줄일 수 있다.

| ID | 변경점 | Test MinPR | CM |
| --- | --- | ---: | --- |
| `P17H-v03` | pose-relative 계열 | 0.6135 | TN=200 FP=126 FN=108 TP=477 |

**판단**: 명확히 실패했다. 기존 filtered kp7 absolute/engineered feature보다 훨씬 낮다.

## 결론

Phase 17의 새 feature 가설은 기준점을 넘지 못했다. 특히 pose-relative 가설은 성능이 크게 무너져 폐기한다.
