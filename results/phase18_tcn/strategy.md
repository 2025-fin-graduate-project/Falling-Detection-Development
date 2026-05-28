# Phase 18 Strategy

**핵심 가설**: GRU는 순환 구조상 장기 의존성 학습에 제약.
TCN(Temporal Convolutional Network) = dilated causal Conv1D residual blocks — 병렬 처리, 
receptive field = 2^D × K × blocks로 명시적으로 통제 가능.
낙상 시퀀스의 시간적 패턴을 GRU보다 잘 포착할 수 있는지 확인.

**TCN 구조** (train_baseline.py 기준):
- residual causal dilated Conv1D blocks
- GlobalAveragePooling + GlobalMaxPooling → concat → Dense head
- default: channels=[32,32,64,96], dilations=[1,2,4,8], kernel=3

**STM32N6 호환성**: TCN도 causal/unidirectional이므로 stateful streaming 가능.
Flash 크기는 채널 수에 비례 — 실측 필요.

**P9O-v01 기준점**: FP=19 FN=20 MinPR=0.9187

**후보 실험**:
| ID | 변경점 | 결과 |
|----|--------|------|
| v01 | TCN default(32,32,64,96) + kp7 + α=0.25, 100에폭 | - |
| v02 | TCN wider(64,64,128,128) + kp7 + α=0.25, 100에폭 | - |
| v03 | TCN(64,128,128,256) + kp7 + α=0.25, 100에폭 | - |
| v04 | TCN default + kp7 + α=0.30, 100에폭 | - |
| v05 | TCN default + kp7 + α=0.25, 100에폭, seed=0 | - |

**종료 기준**: test MinPR ≥ 0.92
