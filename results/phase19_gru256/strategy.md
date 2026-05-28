# Phase 19 Strategy

**핵심 가설**: GRU(256,128)은 Phase 4에서 2-way MinP=0.9513 달성.
현재 4-way MinPR 기준 GRU(128,64) 한계(0.9187)를 용량으로 돌파 가능.
STM32N6 Flash ~1.5 MB — 60 MB 여유로 충분.
P9O-v01의 epochs=30/patience=5 조건을 적용해 동일 정규화 효과 기대.

**GRU(256,128) vs GRU(128,64)**:
- 파라미터 약 4× 증가
- Phase 4 결과: kp12+40f, bidirectional 포함 2-way MinP=0.9513
- 현재 조건(kp7, unidirectional, 4-way metric)에서는 미시험

**P9O-v01 기준점**: FP=19 FN=20 MinPR=0.9187

**후보 실험**:
| ID | 변경점 | 결과 |
|----|--------|------|
| v01 | GRU(256,128) + α=0.25, epochs=30, patience=5, seed=42 | - |
| v02 | GRU(256,128) + α=0.25, epochs=100, patience=15, seed=42 | - |
| v03 | GRU(256,128) + α=0.30, epochs=30, patience=5, seed=42 | - |
| v04 | GRU(256,128) + α=0.25, epochs=30, patience=5, seed=0 | - |

**종료 기준**: test MinPR ≥ 0.92
