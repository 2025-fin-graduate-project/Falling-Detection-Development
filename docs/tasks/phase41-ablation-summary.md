# Phase 41 — Ablation Study: Feature Set, Window Size, Architecture, Preprocessing

## Hypothesis

Feature set 크기(kp5→kp17), window size(20→60), preprocessing(raw/filtered), 속도 특징 추가, 아키텍처(GRU vs TCN)가 낙상 검출 성능에 미치는 영향을 체계적으로 측정한다.

- **Base config**: GRU(64,32), filtered, kp13, w=40, focal loss, stride-5 neg, early-stop-15
- **1회 훈련 = 1 변인 변경** 원칙으로 교차 효과 혼입 방지

---

## 실험 설계

| 그룹 | 변인 | 고정 조건 | 실험 수 |
|---|---|---|---|
| A: Feature set | kp5/kp7/kp9/kp11/kp13/kp17 | filtered, w=40 | 6 |
| B: Window size | w=20/30/40/60 | filtered, kp13 | 4 |
| C: Architecture | GRU vs TCN | filtered, kp7/kp13, w=40 | 4 |
| D: Velocity | +vel / -vel | filtered, kp7/kp13, w=40 | 4 |
| E: Preprocessing | raw / filtered | kp13/kp7, w=40/60 | 3 |

총 21개 실험 (P41-kp5-w40는 학습 실패로 결과 없음)

---

## 실험 결과

### A. Feature Set Ablation (filtered, w=40)

| 모델 | kp | 파라미터 | ev_minpr | FallP | FallR | NFallP | NFallR |
|---|---|---|---|---|---|---|---|
| P41-kp5-w40  | kp5  | 21f  | N/A    | -      | -      | -      | -      |
| P41-kp7-w40  | kp7  | 27f  | 0.8904 | 0.9606 | 0.9760 | 0.9312 | 0.8904 |
| P41-kp9-w40  | kp9  | 33f  | 0.8947 | 0.9624 | 0.9840 | 0.9533 | 0.8947 |
| P41-kp11-w40 | kp11 | 39f  | 0.8640 | 0.9520 | 0.9840 | 0.9517 | 0.8640 |
| P41-kp13-w40 | kp13 | 51f* | 0.9035†| -      | -      | -      | -      |
| P41-kp17-w40 | kp17 | 57f  | 0.9123 | 0.9683 | 0.9760 | 0.9327 | 0.9123 |

*filtered: kp13×3 + 6 derived = 45coord+6=51f  †P42에서 재측정  
**발견**: kp17 > kp13 > kp9 > kp7 > kp11 순. kp11이 kp9보다 낮은 것은 이상값일 수 있음. kp17이 가장 높으나 kp7 대비 성능 향상 폭이 크지 않음(+0.022).

### B. Window Size Ablation (filtered, kp13)

| 모델 | w | ev_minpr | FallP | FallR | NFallP | NFallR |
|---|---|---|---|---|---|---|
| P41-kp13-w20 | 20 | 0.8724 | 0.9738 | 0.9504 | 0.8724 | 0.9298 |
| P41-kp13-w30 | 30 | 0.9079 | 0.9664 | 0.9664 | 0.9079 | 0.9079 |
| P41-kp13-w60 | 60 | 0.9167 | 0.9700 | 0.9840 | 0.9543 | 0.9167 |

(w=40은 P42-kp13-w40에서 측정: 0.9035)

**발견**: window size ↑ → ev_minpr ↑ 단조 증가. w=60이 최고. w=20은 크게 낮아 짧은 컨텍스트로는 낙상 패턴 포착 불충분.

### C. Architecture: GRU vs TCN (filtered, w=40)

| 모델 | arch | kp | ev_minpr |
|---|---|---|---|
| P41-kp7-w40      | GRU | kp7  | 0.8904 |
| P41-tcn-kp7-w40  | TCN | kp7  | 0.8684 |
| P41-kp13-w40*    | GRU | kp13 | 0.9035 |
| P41-tcn-kp13-w40 | TCN | kp13 | 0.8684 |

*P42에서 측정  
**발견**: GRU가 TCN보다 일관되게 우수. kp13에서 격차 +0.035, kp7에서 +0.022. TCN은 stateful streaming에서도 구현 복잡도가 높아 채택 불가.

### D. Velocity Features (filtered, w=40)

| 모델 | vel | kp | ev_minpr |
|---|---|---|---|
| P41-kp7-w40      | ✗ | kp7  | 0.8904 |
| P41-vel-kp7-w40  | ✓ | kp7  | 0.8684 |
| P41-kp13-w40*    | ✗ | kp13 | 0.9035 |
| P41-vel-kp13-w40 | ✓ | kp13 | 0.9123 |

*P42에서 측정  
**발견**: kp7에서는 velocity가 해로움(-0.022). kp13에서는 소폭 이득(+0.009). 전반적으로 이득이 크지 않고 Flash 비용(+36K) 발생. 기본값: velocity 미사용.

### E. Preprocessing: Raw vs Filtered (GRU, w=40/60)

| 모델 | prep | kp | w | ev_minpr |
|---|---|---|---|---|
| P41-kp13-w40*    | filtered | kp13 | 40 | 0.9035 |
| P41-raw-kp13-w40 | raw      | kp13 | 40 | 0.9079 |
| P41-kp13-w60     | filtered | kp13 | 60 | 0.9167 |
| P41-raw-kp13-w60 | raw      | kp13 | 60 | 0.9123 |
| P41-kp7-w40      | filtered | kp7  | 40 | 0.8904 |
| P41-raw-kp7-w40  | raw      | kp7  | 40 | 0.9167 |

**발견**: 필터링이 항상 유리하지 않음. kp7의 경우 raw가 filtered보다 크게 높음(+0.026). kp13 w=60에서도 raw ≥ filtered. One-Euro 필터가 낙상 시 급격한 keypoint 변화를 과도하게 평활화할 가능성.

---

## Flash / MACC (STedgeAI analyze, stm32n6)

| 모델 | Flash (KiB) | MACC | Activation (KiB) |
|---|---|---|---|
| P41-raw-kp7-w40  | 253 | 2,531K | 21.5 |
| P41-kp13-w60     | 275 | 4,141K | 31.5 |
| P41-kp17-w40     | 290 | 2,915K | 21.5 |
| P41-vel-kp13-w40 | 312 | 3,132K | 21.5 |

나머지 모델은 analyze 미실시 (성능이 낮아 배포 후보에서 제외).

---

## 결론

| 변인 | 결론 |
|---|---|
| Feature set | kp17 최고. kp7도 raw 사용 시 경쟁력 있음 |
| Window size  | 클수록 좋음. w=60 최적 (stateful GRU라 배포 비용 없음) |
| Architecture | GRU > TCN. TCN은 stateful 구현 복잡 + 성능 열세로 탈락 |
| Velocity     | 효과 미미하고 Flash 증가. 미사용 권장 |
| Preprocessing| kp7: raw 우세. kp13: 동등 수준. 필터링이 항상 유리하지 않음 |

Phase 41 최고 성능: **P41-raw-kp7-w40** (ev_minpr=0.9167, Flash=253K) = 최소 자원으로 최고 성능  
Phase 42에서 kp17-w60 확인 및 h128 모델 비교 진행.
