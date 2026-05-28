# Phase 41 Summary — Ablation Study

**기간**: 2026-05  
**상태**: 완료  
**스크립트**: `scripts/train_window_phase37.py`  
**결과 디렉토리**: `results/phase41_ablation/`

---

## 목적

단일 변인 변경 원칙으로 feature set, window size, architecture, velocity feature, preprocessing이 낙상 검출 성능(ev_minpr)에 미치는 영향을 독립적으로 측정.

**기본 config**: GRU(64,32), filtered, kp13, w=40, focal loss(α=0.65, γ=2.0), fall-stride=1, nfall-stride=5, early-stop=15, epochs=80

---

## 평가 지표

- **ev_minpr**: min(FallPrecision, FallRecall, NFallPrecision, NFallRecall) at event level
- **Event-level**: 비디오 단위, K-of-N 투표 post-processing (val에서 최적 선택)
- **데이터셋**: class-balanced splits (train ~1.27M rows → neg-stride=5 서브샘플)

---

## 주요 결과

### Feature Set (filtered, w=40)

| kp | 특징 수 | ev_minpr |
|---|---|---|
| kp5  | 21 | N/A (학습 실패) |
| kp7  | 27 | 0.8904 |
| kp9  | 33 | 0.8947 |
| kp11 | 39 | 0.8640 |
| kp17 | 57 | 0.9123 |

(kp13=51f는 P42에서 측정: 0.9035)  
→ **kp17 최고**, 그러나 kp7도 raw preprocessing 시 경쟁력 있음.

### Window Size (filtered, kp13)

| w | ev_minpr |
|---|---|
| 20 | 0.8724 |
| 30 | 0.9079 |
| 40 | 0.9035† |
| 60 | 0.9167 |

†P42에서 측정  
→ **w=60 최고**. Stateful GRU 배포에서 window_size는 훈련 하이퍼파라미터이므로 추가 비용 없음.

### Architecture (filtered, w=40)

| arch | kp7 | kp13 |
|---|---|---|
| GRU | 0.8904 | 0.9035† |
| TCN | 0.8684 | 0.8684 |

†P42에서 측정  
→ **GRU > TCN**. TCN은 stateful streaming 구현 복잡도 및 성능 열세로 탈락.

### Velocity Features (filtered, w=40)

| vel | kp7 | kp13 |
|---|---|---|
| ✗ | 0.8904 | 0.9035† |
| ✓ | 0.8684 | 0.9123 |

†P42에서 측정  
→ kp7에서 해롭고, kp13에서 소폭 이득. Flash +36K 비용 대비 효과 미미.

### Preprocessing (GRU, w=40)

| prep | kp7 | kp13 |
|---|---|---|
| filtered | 0.8904 | 0.9035† |
| raw      | 0.9167 | 0.9079 |

†P42에서 측정  
→ **kp7에서 raw 명확히 우세(+0.026)**. kp13에서는 동등 수준.

---

## 배포 분석 대상 (STedgeAI analyze, stm32n6)

| 모델 | Flash (KiB) | MACC | Activation (KiB) |
|---|---|---|---|
| P41-raw-kp7-w40  | 253 | 2,531K | 21.5 |
| P41-kp13-w60     | 275 | 4,141K | 31.5 |
| P41-kp17-w40     | 290 | 2,915K | 21.5 |
| P41-vel-kp13-w40 | 312 | 3,132K | 21.5 |

모두 STM32N6 Flash 여유(~60MB)에 충분히 여유 있음.

---

## Phase 41 Top 모델

| 모델 | ev_minpr | Flash | 비고 |
|---|---|---|---|
| P41-raw-kp7-w40  | 0.9167 | 253K | 최소 자원 최고 성능 |
| P41-kp13-w60     | 0.9167 | 275K | 동률, 더 많은 특징 |
| P41-kp17-w40     | 0.9123 | 290K | kp17의 w=40 기준점 |
| P41-vel-kp13-w40 | 0.9123 | 312K | velocity 이득 확인 |

---

## 후속 Phase

Phase 42에서 kp17×w60 교차 조합 및 GRU(128,64) 모델 크기 효과 검증.
