# Phase 42 — Cross-Variable Validation

## Hypothesis

Phase 41 ablation에서 발견된 패턴을 교차 검증한다:
1. kp17 × w60 조합이 실제로 최고 성능인지 확인 (P41에서 각각 개별 실험만 함)
2. raw preprocessing이 다른 feature set에서도 유효한지
3. 더 큰 모델(h128)이 이점을 주는지

---

## 실험 설계

| 그룹 | 목적 | 실험 |
|---|---|---|
| A: Gap fill | P41에서 누락된 kp13-w40-filt 측정 | P42-kp13-w40 |
| B: Raw × vars | raw prep이 다른 변인에서도 우위인지 | P42-raw-kp7-w60, P42-raw-kp17-w40, P42-raw-kp13-vel |
| C: Best combo | P41 최고 조합 교차 (kp17×w60, vel×w60) | P42-kp17-w60, P42-vel-kp13-w60 |
| D: Larger model | GRU(128,64)의 효과 | P42-kp13-w40-h128, P42-kp13-w60-h128 |

총 8개 실험, 전부 완료.

---

## 실험 결과

### A. Gap Fill

| 모델 | kp | w | prep | ev_minpr |
|---|---|---|---|---|
| P42-kp13-w40 | kp13 | 40 | filtered | 0.9035 |

P41-kp13-w30(0.9079)보다 낮아 w=40이 w=30보다 항상 유리하지 않음을 확인. (kp13에서는 w=30 > w=40 > w=20 패턴)

### B. Raw × Cross Validation

| 모델 | kp | w | prep | vel | ev_minpr |
|---|---|---|---|---|---|
| P42-raw-kp7-w60    | kp7  | 60 | raw | ✗ | 0.9123 |
| P42-raw-kp17-w40   | kp17 | 40 | raw | ✗ | 0.9123 |
| P42-raw-kp13-vel   | kp13 | 40 | raw | ✓ | 0.8728 |

- raw-kp7-w60: P41-raw-kp7-w40(0.9167)보다 낮음. raw에서도 w=60이 항상 유리하지 않음.
- raw-kp17-w40: filtered-kp17-w40(0.9123)과 동일. 전처리 차이 없음.
- raw+velocity: 0.8728로 크게 낮음. raw 데이터에서 velocity는 명확히 해로움.

### C. Best Combination (kp17 × w60)

| 모델 | kp | w | prep | vel | ev_minpr | FallP | FallR | NFallP | NFallR |
|---|---|---|---|---|---|---|---|---|---|
| P42-kp17-w60     | kp17 | 60 | filtered | ✗ | **0.9211** | 0.9716 | 0.9840 | 0.9545 | 0.9211 |
| P42-vel-kp13-w60 | kp13 | 60 | filtered | ✓ | 0.8684   | 0.9538 | 0.9904 | 0.9706 | 0.8684 |

- **P42-kp17-w60: P41+P42 전체 최고 성능** (ev_minpr=0.9211)
- vel-kp13-w60: velocity × w60 조합은 오히려 하락. velocity 추가 시 더 넓은 컨텍스트가 도움 안 됨.

### D. Larger Model: GRU(128,64)

| 모델 | hidden | w | ev_minpr | val_minpr(학습중) | Flash | MACC |
|---|---|---|---|---|---|---|
| P42-kp13-w40-h128 | [128,64] | 40 | 0.8991 | 0.9604 | 590K | 5,844K |
| P42-kp13-w60-h128 | [128,64] | 60 | 0.9035 | — | 590K | 8,764K |
| P42-kp13-w40      | [64,32]  | 40 | 0.9035 | — | — | — |
| P41-kp13-w60      | [64,32]  | 60 | 0.9167 | — | 275K | 4,141K |

**발견**:
- h128 모델이 val에서는 높은 성능(0.9604)이나 event-level test에서 GRU(64,32)보다 오히려 낮음.
- val→event 갭: h128-w40에서 +0.061 (과적합 또는 window↔event 불일치).
- Flash 2.1× 증가(275→590K) 대비 성능 이득 없음. GRU(64,32) 채택 확정.

---

## Flash / MACC (STedgeAI analyze, stm32n6)

| 모델 | Flash (KiB) | MACC | Activation (KiB) | postproc |
|---|---|---|---|---|
| P42-kp17-w60      | 290 | 4,371K | 31.5 | v10k6 |
| P42-kp13-w40-h128 | 590 | 5,844K | 33.0 | v10k6 |
| P42-kp13-w60-h128 | 590 | 8,764K | 48.0 | v10k6 |

---

## P41 + P42 종합 Top 10 (ev_minpr 기준)

| 순위 | 모델 | kp | w | prep | vel | ev_minpr | Flash |
|---|---|---|---|---|---|---|---|
| 1 | P42-kp17-w60    | kp17 | 60 | filt | ✗ | 0.9211 | 290K |
| 2 | P41-kp13-w60    | kp13 | 60 | filt | ✗ | 0.9167 | 275K |
| 2 | P41-raw-kp7-w40 | kp7  | 40 | raw  | ✗ | 0.9167 | 253K |
| 4 | P41-kp17-w40    | kp17 | 40 | filt | ✗ | 0.9123 | 290K |
| 4 | P41-raw-kp13-w60| kp13 | 60 | raw  | ✗ | 0.9123 | — |
| 4 | P41-vel-kp13-w40| kp13 | 40 | filt | ✓ | 0.9123 | 312K |
| 4 | P42-raw-kp7-w60 | kp7  | 60 | raw  | ✗ | 0.9123 | — |
| 4 | P42-raw-kp17-w40| kp17 | 40 | raw  | ✗ | 0.9123 | — |

---

## 결론

1. **최적 모델**: P42-kp17-w60 (GRU(64,32), kp17 57f, w=60, filtered, ev_minpr=0.9211, Flash=290K)
2. **경량 대안**: P41-raw-kp7-w40 (GRU(64,32), kp7 27f, w=40, raw, ev_minpr=0.9167, Flash=253K) — 더 적은 자원으로 동급 성능
3. **h128 탈락**: Flash 2.1× 증가 대비 성능 이득 없음. GRU(64,32)로 최종 확정.
4. **velocity 탈락**: 대부분 조합에서 성능 하락 또는 이득 미미.
5. **전처리**: kp7에서 raw 우세, kp17에서 동등. 데이터셋에 따라 선택.

Phase 43: INT8 양자화 성능 측정 및 시각화 보고서 준비.
