# Phase 12 Strategy

**핵심 가설**: Phase 11의 근본 병목은 FP(비낙상 FP=23, baseline=19) plateau. 현재 LB-2(binary)로는 "낙상 중"과 "이미 넘어진" 상태를 구분 못해 fallen 구간이 non-fall 학습을 오염시킨다. LB-3(3-class, positive_labels=[1,2])로 학습 시 falling(1)과 fallen(2)을 분리해 모델이 falling 패턴을 더 명시적으로 학습 → FN 감소. focal α=0.10은 v05에서 FN 28→22 효과 확인됨, 계속 유지.

**이전 Phase 실패 원인**: 
- Phase 11: hard-negative training으로 FP 32→23 감소했지만 plateau. focal α=0.10이 FN을 22까지 줄였지만 FP=23은 해결 못함 (baseline FP=19 대비 4개 초과).
- v05 best: MinPR=0.9061 (NFallR=0.9061 bottle-neck), 목표 0.92 미달.
- LB-2에서 fallen(2) 구간이 학습 시 label=1로 잡혀 decision boundary 혼란.

**주요 변경점**:
- `--label-column label_3class --num-classes 3 --positive-labels 1,2`: fallen/falling 분리 학습
- focal α=0.10 유지 (v05 효과 확인)
- hard-negative 선택적 추가 (v02→v03 단계적 검증)

**후보 실험**:
| ID | 체크포인팅 | 변경점 | 결과 |
|----|-----------|--------|------|
| v01 | val_loss | focal α=0.10만 (hard-neg 없음, LB-2) — α=0.10 pure baseline | MinPR=0.9061 (FP=23, FN=20, thr=0.425/mc=3) |
| v02 | val_loss | focal α=0.10 + LB-3 (hard-neg 없음) | MinPR=0.8821 (FP=20↓ FN=29↑, thr=0.700/mc=7 — 과보수적) |
| v03 | val_loss | focal α=0.10 + LB-3 + hard-neg | MinPR=0.8866 (FP=18↓ FN=28↑, thr=0.600/mc=9 — LB-3 FallR 역효과 확인) |
| v04 | val_loss | focal α=0.10 + LB-3 + hard-neg + kp12 | MinPR=0.8911 (FP=16↓↓ FN=27↑, NFallP bottleneck) |

**Phase 12 결론**:
- LB-3 일관 패턴: FP 23→16(↓), FN 20→27(↑), bottleneck NFallR→NFallP 전환
- 목표 달성: FP≤19 AND FN≤19 동시 필요 (0.92 기준)
- LB-3는 FP/FN trade-off 개선 없음 → Phase 12 폐기

**베이스라인**: P9O-v01 test MinPR=0.9187 / Phase 11/12 best: P11-v05=P12-v01 MinPR=0.9061

**종료 기준**: test MinPR ≥ 0.92
