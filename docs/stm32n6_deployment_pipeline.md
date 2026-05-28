# STM32N6 낙상 감지 배포 파이프라인

**모델**: P27-vm0 기준 (GRU(128,64) kp7 40f, event MinPR=0.9241)  
**대상**: STM32N6570-DK / NUCLEO-N657X0-Q (Cortex-M55 + NPU)

---

## 전체 추론 흐름

```
카메라 (15 fps)
    ↓
MoveNet Thunder (256×256, NPU)
    ↓  17 keypoints × (y, x, score)
키포인트 선택 (kp7: 7개)
    ↓
One-Euro 필터 (x, y 좌표 스무딩)
EMA 필터 (confidence 스무딩)
    ↓
엔지니어드 피처 계산 (HSSC, RWHC, VHSSC, AHSSC, AHSSC_x)
    ↓
MinMax 정규화 (27-dim 벡터)
    ↓
GRU 모델 1-step 추론 (CPU, stateful)
    ↓  fall score (0~1)
후처리: threshold + consecutive 카운터
    ↓
낙상 알람 / 상태 초기화
```

---

## 1. 키포인트 선택 (kp7)

MoveNet 17개 keypoint 중 7개 선택:

| MoveNet 인덱스 | 부위 | 선택 이유 |
|---|---|---|
| kp0 | nose (코) | 머리 위치 기준 |
| kp5 | left shoulder (왼 어깨) | 상체 자세 |
| kp6 | right shoulder (오른 어깨) | 상체 자세 |
| kp7 | left elbow (왼 팔꿈치) | 팔 움직임 |
| kp8 | right elbow (오른 팔꿈치) | 팔 움직임 |
| kp11 | left hip (왼 골반) | 낙상 핵심 부위 |
| kp12 | right hip (오른 골반) | 낙상 핵심 부위 |

> **제외 이유**: 발목/무릎(kp13~16)은 카메라 앵글에 따라 미감지 빈번. 손목(kp9,10)은 노이즈 대비 정보량 낮음.

각 keypoint에서 추출: `kp_y`, `kp_x`, `kp_score` → 7개 × 3 = **21개 원시 피처**

---

## 2. 전처리 필터

### 2-1. One-Euro 필터 (좌표 스무딩)

각 keypoint의 **x, y 좌표**에 독립적으로 적용 (14개 스트림).

```
파라미터:
  min_cutoff = 0.5   # 최소 차단 주파수 (Hz)
  beta       = 0.3   # 속도 기반 차단주파수 증가 계수
  d_cutoff   = 1.0   # 미분값 차단 주파수 (Hz)
```

**알고리즘**:
```
매 프레임 t에서:
  1. 속도 추정: dx_hat = EMA(x[t] - x[t-1], alpha=alpha(t_e, d_cutoff))
  2. 동적 차단주파수: cutoff = min_cutoff + beta × |dx_hat|
  3. 스무딩 계수: alpha = 2π×cutoff×dt / (1 + 2π×cutoff×dt)
  4. 출력: x_hat = alpha × x[t] + (1 - alpha) × x_hat[t-1]
```

빠른 움직임(낙상) → beta가 cutoff 증가 → alpha 증가 → 즉각 반응  
느린 움직임(호흡 등) → cutoff 낮음 → 강한 스무딩

### 2-2. EMA 필터 (confidence 스무딩)

각 keypoint의 **score** 에 적용 (7개 스트림).

```
파라미터: alpha = 0.5

score_hat[t] = 0.5 × score[t] + 0.5 × score_hat[t-1]
```

### 2-3. 온디바이스 구현 포인트

- **초기화**: 인물 감지 시작 시 필터 상태 초기화 (이전 인물 오염 방지)
- **인물 미감지**: 카메라 frame skip 시 time step(dt)을 실제 경과 시간으로 설정
- **프레임 순서**: 반드시 시간 순서대로 1-step씩 처리

---

## 3. 엔지니어드 피처 계산

필터링된 좌표로 **6개 파생 피처** 계산:

### HSSC_y, HSSC_x (Hip-Shoulder-Spine Center)
```
HSSC_y = mean(kp0_y, kp5_y, kp6_y, kp7_y, kp8_y, kp11_y, kp12_y)  ← 7 keypoints 평균
HSSC_x = mean(kp0_x, kp5_x, kp6_x, kp7_x, kp8_x, kp11_x, kp12_x)
```
신체 중심의 y/x 위치. 낙상 시 HSSC_y가 급변.

### RWHC (Ratio of Width to Height of bounding box Center)
```
bbox_width  = max(kp_x) - min(kp_x)   ← 전체 keypoint 기준
bbox_height = max(kp_y) - min(kp_y)
RWHC = bbox_width / max(bbox_height, 1e-4)
```
서 있을 때 ≈ 0.3~0.5, 누울 때 > 1.0. 자세 변환 감지.

### VHSSC (Velocity of HSSC_y)
```
VHSSC[t] = EMA((HSSC_y[t] - HSSC_y[t-1]) / dt, alpha=0.4)
```
신체 중심의 수직 속도. 낙상 시 급증.

### AHSSC (Acceleration of HSSC_y)
```
AHSSC[t] = (VHSSC[t] - VHSSC[t-1]) / dt
```
수직 가속도. 낙하 충격 감지.

### AHSSC_x (Lateral Acceleration)
```
VHSSC_x[t] = (HSSC_x[t] - HSSC_x[t-1]) / dt
AHSSC_x[t] = (VHSSC_x[t] - VHSSC_x[t-1]) / dt
```
수평 방향 가속도.

> **주의**: VHSSC/AHSSC/AHSSC_x는 최소 2~3 프레임이 쌓여야 의미있는 값이 나옴.  
> 초기 프레임(t=0, t=1)에서는 0으로 초기화하거나 단순 차분 사용.

---

## 4. 최종 피처 벡터 구성 (27-dim)

| 인덱스 | 피처 | 설명 |
|---|---|---|
| 0 | kp0_y | 코 y (필터링) |
| 1 | kp0_x | 코 x (필터링) |
| 2 | kp0_s | 코 confidence (EMA) |
| 3 | kp5_y | 왼 어깨 y |
| 4 | kp5_x | 왼 어깨 x |
| 5 | kp5_s | 왼 어깨 confidence |
| 6 | kp6_y | 오른 어깨 y |
| 7 | kp6_x | 오른 어깨 x |
| 8 | kp6_s | 오른 어깨 confidence |
| 9 | kp7_y | 왼 팔꿈치 y |
| 10 | kp7_x | 왼 팔꿈치 x |
| 11 | kp7_s | 왼 팔꿈치 confidence |
| 12 | kp8_y | 오른 팔꿈치 y |
| 13 | kp8_x | 오른 팔꿈치 x |
| 14 | kp8_s | 오른 팔꿈치 confidence |
| 15 | kp11_y | 왼 골반 y |
| 16 | kp11_x | 왼 골반 x |
| 17 | kp11_s | 왼 골반 confidence |
| 18 | kp12_y | 오른 골반 y |
| 19 | kp12_x | 오른 골반 x |
| 20 | kp12_s | 오른 골반 confidence |
| 21 | HSSC_y | 신체 중심 y |
| 22 | HSSC_x | 신체 중심 x |
| 23 | RWHC | 바운딩박스 가로/세로 비율 |
| 24 | VHSSC | 수직 속도 (EMA) |
| 25 | AHSSC | 수직 가속도 |
| 26 | AHSSC_x | 수평 가속도 |

---

## 5. MinMax 정규화

훈련 데이터 기준으로 피처별 min/scale 저장 → 온디바이스에서 상수로 적용.

```c
// 의사 코드
for (int i = 0; i < 27; i++) {
    feature[i] = (raw_feature[i] - NORM_MIN[i]) / NORM_SCALE[i];
    feature[i] = clip(feature[i], 0.0f, 1.0f);  // 안전장치
}
```

P27-vm0 정규화 상수는 `results/phase27_seed_sweep/P27-vm0/normalization.json` 참조.  
핵심: VHSSC/AHSSC/AHSSC_x는 음수 가능 → clip(-∞, +∞) 또는 별도 처리 필요.

---

## 6. GRU 모델 구조

```
Input: (1, 27)  ← 1 timestep, 27 features (stateful 스트리밍)

Conv1D(64, kernel=5, padding='causal')  → (1, 64)
Conv1D(64, kernel=5, padding='causal')  → (1, 64)
GRU(128, return_sequences=True)         → (1, 128)   ← hidden state 유지
GRU(64, return_sequences=False)         → (64,)      ← hidden state 유지
Dropout(0.3)  [훈련 시만]
Dense(2, softmax)                       → [p_nonfall, p_fall]
```

**Stateful 주의사항**:
- 학습: 40-frame window 단위, 매 배치 state reset
- 온디바이스: 매 프레임 1-step, hidden state **유지** (reset 안 함)
- 낙상 알람 발생 후: `network_reset()` 호출로 hidden state 초기화

---

## 7. 후처리 (Post-processing)

### 알람 트리거 조건

```
threshold       = 0.525   ← val_video_min_pr 기준으로 선택됨
min_consecutive = 3       ← 연속 3개 윈도우가 threshold 초과 시 알람

if p_fall[t] >= 0.525:
    consecutive_count += 1
else:
    consecutive_count = 0

if consecutive_count >= 3:
    → FALL ALARM
    → GRU hidden state reset
    → consecutive_count = 0
```

### 15fps 기준 타이밍

- 윈도우 판정: 매 프레임마다 (40f 누적 후부터 유효)
- 3 연속 = 최소 **3/15 = 0.2초** 내에 3개 → 실질적으로 연속 3프레임
- 알람 지연: GRU가 40프레임 채운 시점부터 판정 시작 (≈ 2.67초)

### 인물 미감지 처리

```
if no_person_detected_frames >= 45:   ← 3초 = 45프레임 (v26 기준)
    → GRU hidden state reset
    → consecutive_count = 0
    → filter states reset
```

---

## 8. 하드웨어 리소스 (STM32N6 기준)

| 항목 | 값 | 비고 |
|---|---|---|
| Flash (GRU weights) | **~537 KiB** | 0x70680000, 여유 ~60MB |
| Activation RAM | **~40 KiB** | 런타임 버퍼 |
| MACC/frame | **~5.3M** | Cortex-M55 (1프레임) |
| 입력 크기 | (1, 27) float32 | 108 bytes |
| MoveNet Flash | 2.64 MB | 0x70380000 (NPU) |

---

## 9. 포팅 순서

```
1. model.keras 준비
   results/phase27_seed_sweep/P27-vm0/model.keras

2. STedgeAI analyze
   scripts/util/export_stedgeai.py --exp-dir ... --target stm32n6
   → metrics.json에 weights_kib, activations_kib, analyze_ok 기록

3. STedgeAI generate (C코드 생성)
   Model/generate-gru-model_STM32N6570-DK.sh 업데이트 후 실행

4. 온디바이스 C 구현
   - One-Euro + EMA 필터: 프레임마다 호출
   - 엔지니어드 피처: HSSC, RWHC, VHSSC, AHSSC 계산
   - MinMax 정규화: NORM_MIN[], NORM_SCALE[] 상수 배열
   - GRU 1-step: network_run() 호출
   - 후처리: consecutive counter, threshold

5. STedgeAI host eval (INT8 MinPR 검증)
   scripts/util/eval_stedgeai_host.py
   목표: INT8 MinPR ≥ 0.90
```

---

## 10. 구현 시 주의사항

| 항목 | 주의 |
|---|---|
| 좌표 스케일 | MoveNet 출력은 0~1 정규화 (이미지 크기로 나누지 않음) |
| y/x 순서 | MoveNet은 `[y, x, score]` 순서 (x/y 혼용 주의) |
| VHSSC dt | 15fps 고정 시 dt=1/15, 프레임 드랍 시 실제 dt 사용 |
| AHSSC 초기값 | t=0,1에서 0 초기화 (2 프레임 전 데이터 없음) |
| 정규화 음수값 | VHSSC/AHSSC/AHSSC_x는 음수 가능 → clip 없이 그대로 전달 |
| Causal conv | Conv1D padding='causal' → 온디바이스는 과거 kernel-1개 프레임 버퍼 필요 (kernel=5 → 4개 버퍼) |
| GRU state 타입 | float32, 학습과 동일한 정밀도 유지 |
