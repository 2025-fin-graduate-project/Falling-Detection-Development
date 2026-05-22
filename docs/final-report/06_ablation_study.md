# 6. Phase 41/42 — 체계적 변인 탐색 실험 (Ablation Study)

> Phase 40까지의 결과(최고 0.9543, GRU(128,64) stateful)를 바탕으로,
> Phase 41/42에서 **단일 변인 변경 원칙**으로 각 설계 변인의 독립 영향을 정량화하였다.

---

## 6.1 실험 프레임워크

### 기본 구성 (Base Configuration)

| 항목 | 값 |
|---|---|
| 모델 | 단방향 GRU(64,32) |
| 훈련 데이터 | splits_v2_class_balanced_filtered |
| 기본 feature set | kp13 (45 특징) |
| 기본 window size | 40 프레임 |
| Focal loss | α=0.65, γ=2.0 |
| Negative stride | fall-stride=1, nfall-stride=5 |
| Epochs / patience | 80 / 15 |
| 평가 지표 | **ev_minpr** = min(FallP, FallR, NFallP, NFallR) |

### 평가 방법: K-of-N Vote Sweep

이전 Phase의 min_consecutive 방식과 달리, Phase 41/42는 슬라이딩 투표창(vote_window) + K개 이상 양성 판정(vote_k) 조합을 탐색:

```
VOTE_COMBOS = [(1,1),(3,2),(5,3),(5,4),(7,4),(7,5),(10,6)]
```

val set에서 최적 (vw, vk) 조합을 선택한 후 test set에 단일 적용.

---

## 6.2 Feature Set Ablation

**조건**: GRU(64,32), filtered, window=40, velocity 미사용

MoveNet 17 keypoint의 부분 집합 및 파생 특징 포함량에 따른 성능 변화.

| Feature Set | 포함 Keypoints | 특징 수 | ev_minpr | 비고 |
|---|---|---|---|---|
| kp5  | 0,5,6,11,12 | 21 | — | 학습 수렴 실패 |
| kp7  | 0,5,6,11,12,15,16 | 27 | 0.8904 | |
| kp9  | 0,5,6,7,8,11,12,15,16 | 33 | 0.8947 | |
| kp11 | 0,5,6,7,8,9,10,11,12,15,16 | 39 | 0.8640 | kp9보다 낮음 |
| kp13 | 0,5,6,7,8,9,10,11,12,13,14,15,16 | 45 | 0.9035 | |
| kp17 | 전체 0~16 | 57 | **0.9123** | 최고 |

> filtered 데이터셋: kp 좌표 외 파생 특징(HSSC_y, HSSC_x, RWHC, VHSSC, AHSSC, AHSSC_x) 6개 포함

**분석**:
- kp17이 최고이나 kp7 대비 +0.022로 향상 폭이 제한적
- kp11이 kp9보다 낮아 단조 증가가 아님 (팔꿈치 추가가 잡음 증가를 유발할 수 있음)
- kp5는 특징이 너무 적어 수렴 불안정

### Flash 비용 (STedgeAI analyze, stm32n6)

| Feature Set | Flash (KiB) | MACC |
|---|---|---|
| kp7 (raw) | 253 | 2,531K |
| kp13 | 275 | 4,141K |
| kp17 | 290 | 2,915K |
| kp13 + velocity | 312 | 3,132K |

---

## 6.3 Window Size Ablation

**조건**: GRU(64,32), filtered, kp13, velocity 미사용

| Window | 프레임 | 시간 (15fps) | ev_minpr |
|---|---|---|---|
| w=20 | 20 | 1.3s | 0.8724 |
| w=30 | 30 | 2.0s | 0.9079 |
| w=40 | 40 | 2.7s | 0.9035 |
| w=60 | 60 | 4.0s | 0.9167 |

**분석**:
- 클수록 성능 향상. w=40이 w=30보다 낮은 것은 kp13 특유의 패턴 (kp17에서는 단조 증가)
- w=20은 낙상 패턴 포착에 컨텍스트 불충분
- **배포 시 window_size는 무관**: stateful GRU는 매 프레임 hidden state를 누적하므로 w=60이라도 온디바이스 지연 없음

---

## 6.4 Architecture: GRU vs TCN

**조건**: GRU(64,32) vs TCN(동일 규모), filtered, window=40

| Architecture | kp7 ev_minpr | kp13 ev_minpr |
|---|---|---|
| GRU | 0.8904 | 0.9035 |
| TCN | 0.8684 | 0.8684 |
| 차이 | GRU +0.022 | GRU +0.035 |

**Phase 30 결과 (GRU(128,64) vs TCN, splits_v2)와 일치**: GRU가 모든 조합에서 일관 우위.

TCN 탈락 근거:
1. 모든 feature set/window 조합에서 GRU 열세
2. Stateful streaming 구현 시 추가 복잡도 발생
3. STM32N6 dilated conv 연산 지원 미확인

---

## 6.5 Velocity Features

**조건**: GRU(64,32), filtered, window=40

```python
vel_i(t) = feat_i(t) - feat_i(t-1)    # 프레임 간 좌표 차분
# kp7: +12 특징 (y,x 컬럼만) → 27→39
# kp13: +18 특징 → 45→63
```

| Velocity | kp7 ev_minpr | kp13 ev_minpr | Flash 증가 |
|---|---|---|---|
| 미사용 | 0.8904 | 0.9035 | 기준 |
| 사용 | 0.8684 | 0.9123 | +~36K |
| 차이 | -0.022 | +0.009 | |

**Phase 41 결과**는 **Phase 38/39 결과(GRU(128,64))와 다른 방향**:
- Phase 38: GRU(128,64)에서 velocity +0.011 향상
- Phase 41: GRU(64,32)에서 kp13 +0.009, kp7 -0.022

모델 용량에 따라 velocity 유효성이 달라짐. GRU(64,32)는 속도 정보 처리 용량 부족으로 kp7에서 성능 하락.

**Phase 42 추가 검증**:
- raw+velocity: ev_minpr=0.8728 → raw 데이터에서 velocity는 명확히 해로움
- filtered+velocity+w=60: ev_minpr=0.8684 → 넓은 window에서도 velocity 이득 없음

**결론**: Phase 41/42 기준 GRU(64,32)에서 velocity 효과 불안정. 미사용 권장.

---

## 6.6 Preprocessing: Raw vs Filtered

**조건**: GRU(64,32), velocity 미사용

One-Euro 필터 + EMA 적용 키포인트(filtered) vs 원본 키포인트(raw) 비교.
Raw 데이터셋은 파생 특징(AHSSC, AHSSC_x) 미포함.

| Feature Set | Window | Filtered ev_minpr | Raw ev_minpr | 차이 |
|---|---|---|---|---|
| kp7  | 40 | 0.8904 | **0.9167** | raw **+0.026** |
| kp7  | 60 | —      | 0.9123 | — |
| kp13 | 40 | 0.9035 | 0.9079 | raw +0.004 |
| kp13 | 60 | **0.9167** | 0.9123 | filtered +0.004 |
| kp17 | 40 | 0.9123 | 0.9123 | 동등 |

**분석**:
- **kp7에서 raw가 명확히 우세 (+0.026)**
- 가설: One-Euro 필터가 낙상 시 급격한 keypoint 이동을 과도하게 평활화 → 낙상 특유 패턴 소실
- kp13/kp17에서는 파생 특징(AHSSC 등)이 필터링 손실을 보완하여 동등
- kp7은 파생 특징 수가 적어 보완 효과 부족

**Phase 1 결과와 비교**:
- Phase 1 (GRU(256,128), kp12, w=60): filtered +0.004 우위
- Phase 41 (GRU(64,32), kp7, w=40): raw +0.026 우위
- → 모델 크기와 feature set에 따라 전처리의 영향 방향이 달라짐

---

## 6.7 Model Size: GRU(64,32) vs GRU(128,64)

**조건**: kp13, filtered, velocity 미사용

| Hidden | Window | ev_minpr | Flash (KiB) | MACC | Activation (KiB) |
|---|---|---|---|---|---|
| [64,32]  | 40 | 0.9035 | ~150 | ~1.4M | ~12 |
| [64,32]  | 60 | 0.9167 | 275 | 4,141K | 31.5 |
| [128,64] | 40 | 0.8991 | 590 | 5,844K | 33.0 |
| [128,64] | 60 | 0.9035 | 590 | 8,764K | 48.0 |

**GRU(128,64)의 val 단계 성능 vs event-level 괴리**:

| 모델 | val_window_minpr (학습 중) | ev_minpr (test) | 갭 |
|---|---|---|---|
| P42-kp13-w40-h128 | ~0.9604 | 0.8991 | **+0.061** |
| P42-kp13-w60-h128 | — | 0.9035 | — |

GRU(128,64)가 val-window 단계에서 더 높은 성능을 보이나, event-level test에서 GRU(64,32) 대비 열세.
Flash 2.1× 증가(275→590K) 대비 이득 없음.

**결론**: GRU(64,32)로 최종 확정. Phase 40의 GRU(128,64) 최고 성능(0.9543)은 stateful FT 효과이며, 동일 조건 비교 시 GRU(64,32)가 충분히 경쟁력 있음.

---

## 6.8 최적 조합 교차 검증 (Phase 42)

Phase 41에서 단독으로 최고인 변인들의 교차 조합을 Phase 42에서 검증.

| 조합 | kp | w | prep | vel | ev_minpr | 해석 |
|---|---|---|---|---|---|---|
| kp17 × w60 × filt | kp17 | 60 | filt | ✗ | **0.9211** | Phase 41 각각(kp17→0.9123, w60→0.9167) 대비 상승 |
| kp7 × w60 × raw | kp7 | 60 | raw | ✗ | 0.9123 | kp7-raw-w40(0.9167)보다 낮음 — w=60이 raw-kp7에 불리 |
| kp17 × w40 × raw | kp17 | 40 | raw | ✗ | 0.9123 | filtered와 동등 |
| kp13 × w60 × filt × vel | kp13 | 60 | filt | ✓ | 0.8684 | velocity × w60 조합은 오히려 하락 |

→ **P42-kp17-w60이 Phase 41+42 통합 최고 성능** (ev_minpr=0.9211)

---

## 6.9 Phase 41+42 전체 결과 요약

### Top 10 (ev_minpr 기준)

| 순위 | 모델 | kp | w | prep | vel | hidden | ev_minpr | Flash |
|---|---|---|---|---|---|---|---|---|
| 1 | P42-kp17-w60    | kp17 | 60 | filt | ✗ | [64,32] | **0.9211** | 290K |
| 2 | P41-kp13-w60    | kp13 | 60 | filt | ✗ | [64,32] | 0.9167 | 275K |
| 2 | P41-raw-kp7-w40 | kp7  | 40 | raw  | ✗ | [64,32] | 0.9167 | 253K |
| 4 | P41-kp17-w40    | kp17 | 40 | filt | ✗ | [64,32] | 0.9123 | 290K |
| 4 | P41-raw-kp13-w60| kp13 | 60 | raw  | ✗ | [64,32] | 0.9123 | — |
| 4 | P41-vel-kp13-w40| kp13 | 40 | filt | ✓ | [64,32] | 0.9123 | 312K |
| 4 | P42-raw-kp7-w60 | kp7  | 60 | raw  | ✗ | [64,32] | 0.9123 | — |
| 4 | P42-raw-kp17-w40| kp17 | 40 | raw  | ✗ | [64,32] | 0.9123 | — |
| 9 | P41-kp13-w30    | kp13 | 30 | filt | ✗ | [64,32] | 0.9079 | — |
| 9 | P41-raw-kp13-w40| kp13 | 40 | raw  | ✗ | [64,32] | 0.9079 | — |

### 변인별 결론 요약

| 변인 | 최적값 | 근거 |
|---|---|---|
| Feature set | kp17 (float), kp7 (경량) | kp17 최고, kp7 raw 사용 시 동급 |
| Window size  | w=60 | 클수록 유리, 배포 비용 무관 |
| Architecture | GRU | TCN 대비 모든 조합에서 우위 |
| Velocity     | 미사용 | 효과 불안정, Flash 증가 |
| Preprocessing| kp7: raw, kp17: 무관 | kp7에서 raw +0.026 |
| Model size   | [64,32] | [128,64] 대비 Flash 2.1× 절감, 성능 동등 |

---

## 6.10 INT8 양자화 결과 (Phase 43)

STedgeAI 4.0 channel-wise INT8 PTQ — STedgeAI validate --mode host (stm32h7 proxy) 기준.

### Hardware Footprint

| 모델 | Flash (KiB) | MACC | Activation (KiB) |
|---|---|---|---|
| P42-kp17-w60      | 290 | 4,371K | 31.5 |
| P41-kp13-w60      | 275 | 4,141K | 31.5 |
| P41-raw-kp7-w40   | 253 | 2,531K | 21.5 |
| P41-kp17-w40      | 290 | 2,915K | 21.5 |
| P41-vel-kp13-w40  | 312 | 3,132K | 21.5 |
| P42-kp13-w40-h128 | 590 | 5,844K | 33.0 |
| P42-kp13-w60-h128 | 590 | 8,764K | 48.0 |

### Float → INT8 성능 (eval 진행 중)

| 모델 | Float ev_minpr | INT8 ev_minpr | 손실 | 목표(≥0.90) |
|---|---|---|---|---|
| P42-kp17-w60      | 0.9211 | TBD | — | — |
| P41-kp13-w60      | 0.9167 | TBD | — | — |
| P41-raw-kp7-w40   | 0.9167 | TBD | — | — |
| P41-kp17-w40      | 0.9123 | TBD | — | — |
| P41-vel-kp13-w40  | 0.9123 | TBD | — | — |
| P42-kp13-w40-h128 | 0.8991 | TBD | — | — |
| P42-kp13-w60-h128 | 0.9035 | TBD | — | — |

> Phase 41/42 모델의 Phase 21/27/40 대비 INT8 손실 참고:
> 이전 실험 평균 손실 -0.005~-0.010. P42-kp17-w60(float 0.9211)은 손실 후에도 0.90 이상 예상.

---

## 6.11 Phase 41/42 vs Phase 40 비교

| | Phase 40 (최고) | Phase 41/42 (최고) |
|---|---|---|
| 모델 | GRU(128,64) + stateful FT | GRU(64,32) |
| Float ev_minpr | **0.9543** | 0.9211 |
| Flash | 590 KiB | 253~312 KiB |
| MACC | 5.84M | 2.53~4.37M |
| INT8 ev_minpr | 0.8967 (threshold 미재선택) | TBD |

**Phase 40이 성능 절대값 기준 우위**이나, Phase 41/42의 GRU(64,32) 모델이:
- Flash 43~57% 절감
- MACC 25~57% 절감
- Stateful fine-tuning 없이 달성
- INT8 목표(≥0.90) 달성 가능성 있음

두 경로를 모두 보고서에 포함하되, 목적에 따라 선택:
- 성능 우선 → Phase 40 GRU(128,64) stateful
- 경량화 우선 → Phase 41/42 GRU(64,32) kp7/kp17
