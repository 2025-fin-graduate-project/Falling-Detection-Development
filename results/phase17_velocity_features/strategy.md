# Phase 17 Strategy

**핵심 가설**: 현재 피처는 HSSC(머리+어깨 중심)의 집합적 속도만 포함.
낙상 시 개별 관절(코, 어깨, 엉덩이)의 비대칭 운동이 중요한 신호 — 예: 어깨 한쪽이 빠르게 내려가고 엉덩이 한쪽이 올라가는 패턴.
per-keypoint velocity (kp_vy, kp_vx for kp{0,5,6,11,12}) 추가로 FN↓ 기대.

**피처 구성 (kp7kv)**:
- kp7 coords: 7 joints × 3 (y,x,s) = 21
- engineered: HSSC_y, HSSC_x, RWHC, VHSSC, AHSSC, AHSSC_x = 6
- per-joint velocity: kp{0,5,6,11,12} × 2 (vy,vx) = 10
- 합계: 37 features (vs kp7 기존 27)

**데이터셋**: dataset/splits_v2_filtered_kv/ (build_filtered_v2_splits_kv.py)

**P9O-v01 기준점**: FP=19 FN=20 MinPR=0.9187

**후보 실험**:
| ID | 변경점 | 결과 |
|----|--------|------|
| v01 | kp7kv + α=0.25, epochs=30, patience=5, seed=42 | - |
| v02 | kp7kv + α=0.25, epochs=100, patience=15, seed=42 | - |
| v03 | kp7kv + α=0.30, epochs=30, patience=5, seed=42 | - |
| v04 | kp7kv + α=0.25, epochs=30, patience=5, seed=0 | - |

**종료 기준**: test MinPR ≥ 0.92
