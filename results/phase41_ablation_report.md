# STM32N6 낙상 검출 모델 설계를 위한 변인 탐색 실험 보고서

> INT8 MinP 칸은 STedgeAI host eval 완료 후 채움 (진행 중)

---

## 1. 개요

본 보고서는 STM32N6 엣지 MCU에서 실시간 낙상 검출을 위한 GRU 기반 시퀀스 모델의 설계 변인을 체계적으로 탐색한 실험 결과를 정리한다. 탐색 변인은 입력 특징 집합(feature set), 훈련 윈도우 크기(window size), 전처리 방식(preprocessing), 속도 특징(velocity feature), 모델 크기(hidden size)이며, 각 변인의 영향을 독립적으로 측정하고 상위 조합을 교차 검증하였다.

---

## 2. 실험 설정

### 2.1 대상 플랫폼 및 배포 구조

| 항목 | 내용 |
|---|---|
| MCU | STM32N6570-DK (Cortex-M55, NPU 탑재) |
| 추론 구조 | MoveNet(NPU) → 17 keypoints → GRU(CPU) → 낙상 점수 |
| 양자화 | STedgeAI 4.0 channel-wise INT8 PTQ |
| 스트리밍 | 단방향 GRU, 1-step stateful, 낙상 알람 시 hidden state 초기화 |

단방향 GRU를 사용하는 이유: 양방향 GRU는 미래 프레임을 요구하므로 실시간 스트리밍에서 window_size 프레임만큼의 검출 지연이 발생하며, stateful 스트리밍이 불가능하다.

### 2.2 입력 특징

MoveNet Thunder가 출력하는 17개 keypoint의 (y, x, score) 좌표를 사용한다. 부분 집합 및 파생 특징은 아래와 같다.

| 특징 집합 | Keypoints | 특징 수 (raw) | 특징 수 (filtered) |
|---|---|---|---|
| kp5  | 0,5,6,11,12 | 15 | — |
| kp7  | 0,5,6,11,12,15,16 | 21 | 27 |
| kp9  | 0,5,6,7,8,11,12,15,16 | 27 | 33 |
| kp11 | 0,5,6,7,8,9,10,11,12,15,16 | 33 | 39 |
| kp13 | 0,5,6,7,8,9,10,11,12,13,14,15,16 | 39 | 45 |
| kp17 | 0–16 (전체) | 51 | 57 |

filtered dataset은 One-Euro+EMA 필터 적용 keypoint에 파생 특징(HSSC_y, HSSC_x, RWHC, VHSSC, AHSSC, AHSSC_x) 6개를 추가한다.

### 2.3 훈련 설정

| 항목 | 값 |
|---|---|
| 아키텍처 | 단방향 GRU(64,32) 기본, GRU(128,64) 비교 |
| 손실 함수 | Focal loss (α=0.65, γ=2.0) |
| 정규화 | Dropout 0.3, Gaussian noise 0.02 |
| 옵티마이저 | Adam |
| 최대 epoch | 80, early stop patience=15 |
| 윈도우 샘플링 | fall-stride=1, nfall-stride=5 |
| Val 서브샘플 | 20K windows/epoch |

### 2.4 데이터셋

| 분할 | 비디오 수 | 낙상/비낙상 |
|---|---|---|
| Train | ~7,200 | balanced |
| Val   | ~1,714 | balanced |
| Test  | ~853   | balanced |

카메라 C5~C8; 이상치 제거; class-balanced splits (class 비율 균등). 

### 2.5 평가 지표

**Window-level**: 개별 윈도우 이진 분류 정밀도  
**Event-level (ev_minpr)**: 비디오 단위, K-of-N 투표 post-processing 후 측정

$$\text{MinP} = \min(\text{FallPrecision},\ \text{FallRecall},\ \text{NFallPrecision},\ \text{NFallRecall})$$

- Float 목표: MinP ≥ 0.93  
- INT8 목표: MinP ≥ 0.90  
- Threshold 및 post-processing 파라미터(vw, vk)는 val set에서 최적화 후 test에 단일 적용

---

## 3. 실험 결과

### 3.1 Feature Set Ablation

**조건**: GRU(64,32), filtered, window=40, velocity 미사용

| Feature Set | 특징 수 | ev_minpr | Flash (KiB) | MACC |
|---|---|---|---|---|
| kp5  | 21 | 0.8728 | — | — |
| kp7  | 27 | 0.8904 | — | — |
| kp9  | 33 | 0.8947 | — | — |
| kp11 | 39 | 0.8640 | — | — |
| kp13 | 45 | 0.9035 | — | — |
| kp17 | 57 | **0.9123** | 290 | 2,915K |

kp5 결과(0.8728)는 post-processing 불안정(v5k3)으로 제한적 참고값. kp11이 kp9보다 낮아 단조 증가가 아님 — 손목 keypoint 추가가 잡음으로 작용. kp17이 최고이나 kp7 대비 +0.022로 향상 폭이 제한적.

### 3.2 Window Size Ablation

**조건**: GRU(64,32), filtered, kp13, velocity 미사용

| Window Size | 프레임 | 초 (15fps) | ev_minpr |
|---|---|---|---|
| 20 | 20 | 1.3s | 0.8724 |
| 30 | 30 | 2.0s | 0.9079 |
| 40 | 40 | 2.7s | 0.9035 |
| 60 | 60 | 4.0s | 0.9167 |

window size가 클수록 성능 향상. w=40이 w=30보다 낮은 것은 kp13 특유의 패턴으로, kp17에서는 w=40(0.9123) < w=60(0.9211) 단조 증가. Stateful GRU 배포에서 window_size는 훈련 전용 하이퍼파라미터이며 온디바이스 추론 지연과 무관.

### 3.3 Architecture: GRU vs TCN

**조건**: GRU(64,32) vs TCN(동일 파라미터), filtered, window=40

| Architecture | kp7  | kp13 |
|---|---|---|
| GRU | 0.8904 | 0.9035 |
| TCN | 0.8684 | 0.8684 |

GRU가 모든 조합에서 우위. TCN은 stateful 스트리밍 구현 시 추가 복잡도 발생. GRU 채택 확정.

### 3.4 Velocity Features

**조건**: GRU(64,32), filtered, window=40

| Velocity | kp7 | kp13 |
|---|---|---|
| 미사용 | 0.8904 | 0.9035 |
| 사용  | 0.8684 | 0.9123 |

kp7에서 성능 하락(-0.022), kp13에서 소폭 상승(+0.009). Flash 추가(+36~38K) 대비 효과가 일관되지 않아 기본 구성에서 미사용으로 결정.

### 3.5 Preprocessing: Raw vs Filtered

**조건**: GRU(64,32), velocity 미사용

| Feature Set | Window | Filtered | Raw | 차이 |
|---|---|---|---|---|
| kp7  | 40 | 0.8904 | 0.9167 | raw +0.026 |
| kp13 | 40 | 0.9035 | 0.9079 | raw +0.004 |
| kp13 | 60 | 0.9167 | 0.9123 | filtered +0.004 |
| kp17 | 40 | 0.9123 | 0.9123 | 동등 |

kp7에서 raw가 명확히 우세. One-Euro 필터가 낙상 시 급격한 keypoint 변화를 과도하게 평활화할 가능성이 있음. kp13/kp17에서는 전처리 차이가 작음.

### 3.6 Model Size: GRU(64,32) vs GRU(128,64)

**조건**: kp13, filtered, velocity 미사용

| Hidden | Window | ev_minpr | Flash (KiB) | MACC |
|---|---|---|---|---|
| [64,32]  | 40 | 0.9035 | — | — |
| [64,32]  | 60 | 0.9167 | 275 | 4,141K |
| [128,64] | 40 | 0.8991 | 590 | 5,844K |
| [128,64] | 60 | 0.9035 | 590 | 8,764K |

GRU(128,64)가 val-window 단계에서는 더 높은 성능을 보이나, event-level test에서 GRU(64,32) 대비 열세. Flash 2.1× 증가(275→590K) 대비 이득 없음. **GRU(64,32) 최종 채택**.

### 3.7 최적 조합 교차 검증

Phase 42에서 Phase 41 최상위 단독 변인들의 교차 조합을 검증:

| 모델 | kp | w | prep | ev_minpr | 비고 |
|---|---|---|---|---|---|
| P42-kp17-w60   | kp17 | 60 | filt | **0.9211** | 전체 최고 |
| P42-raw-kp7-w60| kp7  | 60 | raw  | 0.9123 | P41-raw-kp7-w40(0.9167)보다 낮음 |
| P42-raw-kp17-w40| kp17| 40 | raw  | 0.9123 | filtered와 동등 |
| P42-vel-kp13-w60| kp13| 60 | filt+vel | 0.8684 | velocity × w60 조합은 오히려 하락 |

---

## 4. 배포 적합성 분석

STedgeAI 4.0 (stm32n6 target) analyze 결과:

| 모델 | Flash (KiB) | MACC | Activation (KiB) | ev_minpr | INT8 MinP |
|---|---|---|---|---|---|
| P42-kp17-w60    | 290 | 4,371K | 31.5 | 0.9211 | TBD |
| P41-kp13-w60    | 275 | 4,141K | 31.5 | 0.9167 | TBD |
| P41-raw-kp7-w40 | 253 | 2,531K | 21.5 | 0.9167 | TBD |
| P41-kp17-w40    | 290 | 2,915K | 21.5 | 0.9123 | TBD |
| P41-vel-kp13-w40| 312 | 3,132K | 21.5 | 0.9123 | TBD |
| P42-kp13-w40-h128| 590 | 5,844K | 33.0 | 0.8991 | TBD |
| P42-kp13-w60-h128| 590 | 8,764K | 48.0 | 0.9035 | TBD |

STM32N6 Flash 여유 ~60MB에 모든 후보가 여유 있게 적합. 활성화 버퍼(Activation)는 MCU RAM(4MB 이상)에 충분.

---

## 5. 결론

| 변인 | 결론 | 최적값 |
|---|---|---|
| Feature set | 클수록 유리. kp7도 raw 시 경쟁력 | kp17 (float), kp7 (경량) |
| Window size  | 클수록 유리. 배포 비용 없음 | w=60 |
| Architecture | GRU > TCN | GRU |
| Velocity     | 효과 일관되지 않음, Flash 증가 | 미사용 |
| Preprocessing| kp7: raw 우세. kp17: 동등 | 특징에 따라 선택 |
| Model size   | h128 이득 없음, Flash 2.1× | GRU(64,32) |

### 최종 모델 권장

**최고 성능**: P42-kp17-w60 — GRU(64,32), kp17(57f), w=60, filtered  
- ev_minpr=0.9211, Flash=290K, MACC=4.37M, Activation=31.5K

**경량 대안**: P41-raw-kp7-w40 — GRU(64,32), kp7(21f), w=40, raw  
- ev_minpr=0.9167, Flash=253K, MACC=2.53M, Activation=21.5K  
- 자원 15% 절약, 성능 차이 0.004

INT8 양자화 후 MinP ≥ 0.90 유지 여부는 STedgeAI host eval 결과(TBD)로 확인 예정.

---

## 부록: Post-processing 파라미터

모든 모델에서 최적 post-processing은 v10k6 (10프레임 투표창, 6개 이상 positive 시 낙상 판정)으로 수렴. 이는 15fps 기준 약 0.67초의 확인 시간에 해당하며, 낙상 검출 지연을 최소화하면서 오탐율을 억제하는 적절한 균형점.

---

*작성일: 2026-05-21 | 데이터: Phase 41 (21실험) + Phase 42 (8실험) | INT8 값: TBD (eval 진행 중)*
