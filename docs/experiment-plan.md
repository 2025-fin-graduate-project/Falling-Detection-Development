# 낙상 감지 모델 실험 계획서

**버전**: v1.1
**작성일**: 2026-05-13
**목표**: 데이터 분할을 고정한 뒤, **전처리 파이프라인·키포인트·모델·프레임 설정**을 독립 변인으로 제어하여 재현 가능한 성능 비교 수행

---

## 1. 프로젝트 목표

| 항목 | 내용 |
|------|------|
| **성능 목표** | 모든 핵심 지표 **0.93 ~ 0.95** 수렴 |
| **평가 기준** | Accuracy, Precision, Recall, F1, AUC-ROC (낙상 클래스 기준) |
| **배포 대상** | STM32N6 (TFLite INT8 양자화) |
| **추가 요구사항** | 양자화 전·후 지표 모두 기록 |

---

## 2. 변인 / 상수 분류

### 2.1 변인 (실험마다 바뀌는 것)

| 변인 | 선택지 | 담당 Phase |
|------|--------|-----------|
| **전처리 파이프라인** | PP-raw / PP-D | Phase 0 |
| **모델 구조** | TCN / GRU | 전 Phase |
| **라벨 구성** | LB-2 / LB-3 | Phase 1 |
| **키포인트 구성** | KP-12 / KP-8 | Phase 2 |
| **학습 데이터 범위** | 전방향 / BY제외 | Phase 3 |
| **프레임 설정** | 15fps×4s / 30fps×2s | Phase 4 |

### 2.2 상수 (모든 실험 고정)

| 항목 | 값 |
|------|----|
| **데이터 분할** | `dataset/splits/` (seed=42, stratified) |
| **배치 크기** | 64 |
| **옵티마이저** | Adam (lr=1e-3, ReduceLROnPlateau) |
| **손실함수** | Binary Cross-Entropy + class weights |
| **조기 종료** | patience=10 (val F1 기준) |
| **최대 에폭** | 100 |
| **정규화** | MinMax (train fit → val/test transform) |
| **난수 시드** | 42 |
| **이상치 제거** | 비디오 단위, z>2.5 & conf<0.28 (83.5% 유지) |

> **test.csv는 최종 보고 시 1회만 사용.** 하이퍼파라미터 탐색 및 변인 선정은 val 기준.

---

## 3. 고정 사양

### 3.1 원본 데이터

```
dataset/final_dataset.csv
  rows   : 4,093,620
  videos : 20,380
  label  : 0=비낙상, 1=낙상
  방향   : BY(후면) / FY(전면) / SY(측면) / N(비낙상)
```

### 3.2 데이터 분할 (`dataset/splits/`, 고정)

| 파일 | 비디오 수 | rows | 비율 |
|------|-----------|------|------|
| `train.csv` | 11,910 | 2,391,321 | 70% |
| `val.csv`   |  3,401 |   683,451 | 20% |
| `test.csv`  |  1,707 |   342,744 | 10% |

분할 기준: 비디오 단위 stratified (방향별 비율 유지)

---

## 4. 전처리 파이프라인 정의 (변인)

> 각 파이프라인은 **동일한 분할**(`splits/`) 위에서 적용.
> 피처 컬럼 구성이 달라지므로 파이프라인별 별도 CSV 생성.

### PP-raw: 원본 (클리핑만)

```
좌표 클리핑:  kp*_y/x → [0, 1]
파생 피처:    HSSC_y/x, RWHC, VHSSC  (원본 좌표 기반 계산)
필터링:       없음
```

- 원본 노이즈를 그대로 학습에 노출
- **기준선(baseline) 역할**: 필터링 효과를 측정하는 대조군
- 사용 파일: `dataset/final_dataset.csv`

### PP-D: Pipeline D (현재 구현)

```
1. 좌표 클리핑         kp*_y/x → [0, 1]
2. 저신뢰도 마스킹     score < 0.15 → NaN → 선형보간
3. One-Euro Filter     min_cutoff=0.5, beta=0.3  (좌표 스무딩)
4. EMA Filter          alpha=0.5  (신뢰도 스무딩)
5. VHSSC EMA 스무딩    alpha=0.4  (미분 잡음 -71.9%)
6. 파생 피처 재계산    HSSC_y/x, RWHC, VHSSC, AHSSC, AHSSC_x
```

- VHSSC Cohen's d 0.59 → 0.96 (+62%), AHSSC 잡음 std 3.35 → 0.94 (-72%)
- 스크립트: `scripts/build_filtered_dataset.py`
- 사용 파일: `dataset/final_dataset_filtered.csv`

| 설정 | 필터링 | 파생 피처 수 | 특징 |
|------|--------|------------|------|
| PP-raw | 없음 | 4개 (HSSC_y/x, RWHC, VHSSC) | 대조군 |
| PP-D   | One-Euro + EMA | 6개 (+ AHSSC, AHSSC_x) | 현재 최적 |

---

## 5. 라벨 구성 정의 (변인)

> 낙상 직후 쓰러진 상태(fallen)를 별도 클래스로 분리하면
> 모델이 "낙상 중"과 "쓰러진 후 정지" 를 구별해 학습할 수 있음.
> `label_3class` 컬럼은 `build_filtered_dataset.py`에서 이미 생성됨.

### LB-2: 이진 라벨 (기본)

```
0 = 비낙상 (normal + fallen 이후 포함)
1 = 낙상 중 (falling)

출력층: Dense(1, sigmoid)
손실함수: Binary Cross-Entropy
```

- 구현 단순, 기존 파이프라인과 호환
- 낙상 후 정지 상태가 0으로 학습되어 혼동 가능성 존재

### LB-3: 3-클래스 라벨

```
0 = 비낙상 (normal)
1 = 낙상 중 (falling)   ← 비디오에서 label=1인 구간
2 = 낙상 후 (fallen)    ← 마지막 label=1 프레임 이후 구간

출력층: Dense(3, softmax)
손실함수: Categorical Cross-Entropy
평가 시: 클래스 1+2를 합쳐 "낙상 이벤트"로 집계 가능
```

- 낙상 직후 정지 상태를 학습 신호로 활용
- 모델이 낙상 진행 단계를 명시적으로 구분
- 추론 시 클래스 1 또는 2 → 낙상 경보 발생으로 처리 가능

| 설정 | 클래스 수 | 출력층 | 비고 |
|------|---------|--------|------|
| LB-2 | 2 | sigmoid | 대조군 |
| LB-3 | 3 | softmax | 단계 구분 학습 |

---

## 6. 키포인트 구성 정의 (변인)

> MoveNet은 항상 17개 출력 → **학습 시 CSV에서 사용할 컬럼만 선택**
> 파생 피처는 어느 설정에서든 공통 포함

### KP-12: 상체+골반 세트

```
kp0  코,  kp1 왼눈,  kp2 오른눈,  kp3 왼귀,  kp4 오른귀
kp5  왼어깨,  kp6 오른어깨
kp7  왼팔꿈치,  kp8 오른팔꿈치
kp9  왼손목,  kp10 오른손목
kp11 왼골반,  kp12 오른골반
```

제거: kp13~16 (무릎·발목) — VHSSC 상관 < 0.05

### KP-8: 핵심 관절 세트

```
kp0  코
kp5  왼어깨,  kp6 오른어깨
kp7  왼팔꿈치,  kp8 오른팔꿈치
kp11 왼골반,  kp12 오른골반
```

추가 제거: kp1~4 (눈·귀, 코와 중복), kp9~10 (손목, 신뢰도 최하 ~0.33)

| 설정 | kp 수 | PP-raw feature | PP-D feature |
|------|-------|----------------|-------------|
| KP-12 | 12개 | 12×3 + 4 = **40** | 12×3 + 6 = **42** |
| KP-8  |  8개 |  8×3 + 4 = **28** |  8×3 + 6 = **30** |

---

## 6. 모델 아키텍처 사양

### GRU

```
Input [batch, T, features]
→ GRU(64) → GRU(32) → Dropout(0.3)
→ Dense(32, relu) → Dropout(0.2)
→ Dense(1, sigmoid)
```

### TCN

```
Input [batch, T, features]
→ TCN Block × 4  (filter=64, kernel=3, dilation=[1,2,4,8])
   각 블록: Conv1D → BatchNorm → ReLU → Dropout(0.2) + Residual
→ GlobalAveragePooling → Dense(32, relu) → Dense(1, sigmoid)
```

T = 60 (15fps×4s) 또는 60 (30fps×2s)

---

## 7. 실험 매트릭스

### Phase 0 — 전처리 파이프라인 비교 (베이스라인 확립)

> 첫 번째로 결정할 변인. 이후 모든 Phase의 전처리 고정.
> 라벨은 LB-2, 키포인트 KP-12, 데이터 BY제외, 15fps×4s 고정.

| ID | 모델 | 전처리 | 라벨 | 키포인트 | 데이터 | fps×window |
|----|------|--------|------|---------|--------|------------|
| **B-TCN-raw** | TCN | PP-raw | LB-2 | KP-12 | BY 제외 | 15fps × 4s |
| **B-TCN-D**   | TCN | PP-D   | LB-2 | KP-12 | BY 제외 | 15fps × 4s |
| **B-GRU-raw** | GRU | PP-raw | LB-2 | KP-12 | BY 제외 | 15fps × 4s |
| **B-GRU-D**   | GRU | PP-D   | LB-2 | KP-12 | BY 제외 | 15fps × 4s |

비교 포인트: 필터링이 실제 모델 성능에 기여하는지 정량 검증
**결과**: val F1 기준 최적 전처리 선정 → 이후 Phase 고정

---

### Phase 1 — 라벨 구성 변인 실험

> Phase 0 최적 전처리 고정. 이진(LB-2) vs 3-클래스(LB-3) 비교.
> 키포인트 KP-12, 데이터 BY제외, 15fps×4s 고정.

| ID | 모델 | 전처리 | 라벨 | 키포인트 | 데이터 | fps×window |
|----|------|--------|------|---------|--------|------------|
| **L-TCN-2** | TCN | Phase0 최적 | LB-2 | KP-12 | BY 제외 | 15fps × 4s |
| **L-TCN-3** | TCN | Phase0 최적 | LB-3 | KP-12 | BY 제외 | 15fps × 4s |
| **L-GRU-2** | GRU | Phase0 최적 | LB-2 | KP-12 | BY 제외 | 15fps × 4s |
| **L-GRU-3** | GRU | Phase0 최적 | LB-3 | KP-12 | BY 제외 | 15fps × 4s |

비교 포인트: fallen 상태 분리가 낙상 탐지 성능(Recall, F1)에 미치는 영향
LB-3 평가 시 class 1+2 합산 → 낙상 이벤트 단위로 재집계
**결과**: 최적 라벨 구성 선정 → 이후 Phase 고정

---

### Phase 2 — 키포인트 변인 실험

> Phase 0·1 최적 설정 고정. 키포인트 구성만 변경.

| ID | 모델 | 전처리 | 라벨 | 키포인트 | 데이터 | fps×window |
|----|------|--------|------|---------|--------|------------|
| **K-TCN-12** | TCN | Phase0 최적 | Phase1 최적 | KP-12 | 전방향 | 15fps × 4s |
| **K-TCN-8**  | TCN | Phase0 최적 | Phase1 최적 | KP-8  | 전방향 | 15fps × 4s |
| **K-GRU-12** | GRU | Phase0 최적 | Phase1 최적 | KP-12 | 전방향 | 15fps × 4s |
| **K-GRU-8**  | GRU | Phase0 최적 | Phase1 최적 | KP-8  | 전방향 | 15fps × 4s |

비교 포인트: 피처 축소가 성능·모델 크기에 미치는 영향

---

### Phase 3 — 학습 데이터 범위 변인 실험

> Phase 0~2 최적 설정 고정. 후면(BY) 포함 여부만 변경.

| ID | 모델 | 전처리 | 라벨 | 키포인트 | 데이터 | fps×window |
|----|------|--------|------|---------|--------|------------|
| **D-TCN-full** | TCN | Phase0 최적 | Phase1 최적 | Phase2 최적 | 전방향 | 15fps × 4s |
| **D-TCN-noby** | TCN | Phase0 최적 | Phase1 최적 | Phase2 최적 | BY 제외 | 15fps × 4s |
| **D-GRU-full** | GRU | Phase0 최적 | Phase1 최적 | Phase2 최적 | 전방향 | 15fps × 4s |
| **D-GRU-noby** | GRU | Phase0 최적 | Phase1 최적 | Phase2 최적 | BY 제외 | 15fps × 4s |

비교 포인트: 후면 포함 시 전체 F1 vs 후면 방향 Recall

---

### Phase 4 — 프레임 설정 변인 실험 (추후)

> Phase 0~3 최적 설정 고정. 윈도우·fps만 변경.

| ID | 모델 | 전처리 | 라벨 | 키포인트 | 데이터 | fps×window |
|----|------|--------|------|---------|--------|------------|
| **F-TCN-15** | TCN | Phase0 최적 | Phase1 최적 | Phase2 최적 | Phase3 최적 | **15fps × 4s** |
| **F-TCN-30** | TCN | 동일 | 동일 | 동일 | 동일 | **30fps × 2s** |
| **F-GRU-15** | GRU | 동일 | 동일 | 동일 | 동일 | 15fps × 4s |
| **F-GRU-30** | GRU | 동일 | 동일 | 동일 | 동일 | 30fps × 2s |

비교 포인트: 동일 프레임 수(60f)에서 시간 해상도·커버 구간 차이

---

### 전체 실험 요약

| Phase | 변인 | 실험 수 | 고정 조건 |
|-------|------|---------|----------|
| 0 전처리 | PP-raw vs PP-D | 4 | LB-2, KP-12, BY제외, 15fps×4s |
| 1 라벨 | LB-2 vs LB-3 | 4 | Phase0 최적 PP, KP-12, BY제외, 15fps×4s |
| 2 키포인트 | KP-12 vs KP-8 | 4 | Phase0~1 최적, 전방향, 15fps×4s |
| 3 데이터 | 전방향 vs BY제외 | 4 | Phase0~2 최적, 15fps×4s |
| 4 프레임 (추후) | 15fps×4s vs 30fps×2s | 4 | Phase0~3 최적 |
| **합계** | | **20** | |

---

## 8. 평가 지표 사양

### 8.1 기본 지표

| 지표 | 설명 | 기준 |
|------|------|------|
| **Accuracy** | 전체 정확도 | 프레임 단위 |
| **Precision** | 낙상 예측의 정확률 | label=1 |
| **Recall** | 실제 낙상 탐지율 | label=1 |
| **F1** | Precision·Recall 조화평균 | label=1 |
| **AUC-ROC** | ROC 곡선 면적 | — |
| **Latency** | 추론 시간 (ms/window) | CPU |

### 8.2 양자화 지표 (TFLite INT8)

| 지표 | 설명 |
|------|------|
| **Q-F1** | 양자화 후 F1 |
| **ΔF1** | float → int8 F1 저하량 |
| **Model size (KB)** | .tflite 파일 크기 |
| **Q-Latency** | STM32N6 추론 시간 추정 |

**목표 기준**:
```
F1 ≥ 0.93,  Recall ≥ 0.93  (낙상 클래스)
ΔF1 < 0.02  (양자화 열화 허용치)
```

### 8.3 방향별 세분 지표 (Phase 2 필수)

BY / FY / SY별 Recall 개별 기록. 특정 방향 취약 여부 진단.

---

## 9. 실험 결과 출력물 사양

### 9.1 각 실험당 생성 파일

```
results/{experiment_id}/
  ├── metrics.json             # 모든 수치 지표
  ├── confusion_matrix.png     # 혼동 행렬
  ├── roc_curve.png            # ROC 커브
  ├── pr_curve.png             # Precision-Recall 커브
  ├── training_curve.png       # loss / F1 학습 곡선
  ├── model.keras              # 원본 모델
  ├── model_q.tflite           # INT8 양자화 모델
  └── quantization_report.json # 양자화 전후 비교
```

### 9.2 `metrics.json` 구조

```json
{
  "experiment_id": "K-GRU-8",
  "preprocessing": "PP-D",
  "label_config": "LB-3",
  "model": "GRU",
  "keypoints": "KP-8",
  "data": "all_directions",
  "window": "15fps_4s",
  "val": {
    "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
    "f1": 0.0, "auc_roc": 0.0, "latency_ms": 0.0
  },
  "test": {
    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
    "by_direction": {"BY": 0.0, "FY": 0.0, "SY": 0.0}
  },
  "quantized": {
    "q_f1": 0.0, "delta_f1": 0.0,
    "model_size_kb": 0.0, "q_latency_ms": 0.0
  }
}
```

### 9.3 Phase 종료 시 비교 시각화

| 차트 | 내용 |
|------|------|
| `compare_f1_bar.png` | Phase 내 F1 비교 |
| `compare_radar.png` | 5축 radar (Acc·Pre·Rec·F1·AUC) |
| `compare_quantization.png` | float vs int8 F1 |
| `compare_by_direction.png` | 방향별 Recall 히트맵 |

---

## 10. 실험 진행 절차

```
Phase 0: 전처리 파이프라인 결정
  ├─ PP-raw × TCN·GRU  학습·평가·양자화
  ├─ PP-D   × TCN·GRU  학습·평가·양자화
  └─ val F1 기준 최적 PP 선정 → 이후 고정
        ↓
Phase 1: 라벨 구성 결정
  ├─ LB-2 × TCN·GRU  학습·평가·양자화
  ├─ LB-3 × TCN·GRU  학습·평가·양자화
  │   ※ LB-3 평가: class 1+2 합산 → 낙상 이벤트 F1 재계산
  └─ val F1(낙상 이벤트) 기준 최적 라벨 선정 → 이후 고정
        ↓
Phase 2: 키포인트 결정
  ├─ KP-12 / KP-8 각 2모델 × 학습·평가·양자화
  └─ val F1 + 모델 크기 고려 최적 kp 선정 → 이후 고정
        ↓
Phase 3: 데이터 범위 결정
  ├─ 전방향 / BY제외 각 2모델 × 학습·평가·양자화
  └─ 방향별 Recall 포함 종합 비교 → 최적 data 선정
        ↓
Phase 4: 프레임 설정 결정 (추후)
  ├─ 15fps×4s / 30fps×2s 각 2모델
  └─ 최종 최적 설정 확정
        ↓
최종 보고
  ├─ 전 실험 비교표 (metrics.json 집계)
  ├─ 최적 모델 test.csv 최종 평가 (1회)
  └─ STM32N6 배포 패키지
```

---

## 11. 파일 구조

```
Falling-Model-Development/
├── dataset/
│   ├── final_dataset.csv               # 원본
│   ├── final_dataset_filtered.csv      # PP-D 적용본
│   └── splits/                         # 고정 분할
│       ├── train.csv   (11,910 videos)
│       ├── val.csv     ( 3,401 videos)
│       └── test.csv    ( 1,707 videos)
├── scripts/
│   ├── build_filtered_dataset.py       # PP-D 전처리
│   ├── analyze_and_split_dataset.py    # 분석·분할
│   ├── train_baseline.py               # Phase 0 학습
│   └── evaluate_and_quantize.py        # 평가 + 양자화
├── results/
│   ├── B-TCN-raw/  B-TCN-D/
│   ├── B-GRU-raw/  B-GRU-D/
│   ├── K-TCN-12/   K-TCN-8/
│   ├── K-GRU-12/   K-GRU-8/
│   └── ...
└── docs/
    ├── experiment-plan.md              # 이 문서
    ├── analysis/
    │   └── dataset_analysis_report.md
    └── experiment-results.md           # 실험 완료 후 작성
```

---

## 12. 미결 사항 (TODO)

| 항목 | 상태 |
|------|------|
| `train_baseline.py` 구현 (TCN·GRU 공통) | 미시작 |
| `evaluate_and_quantize.py` 구현 | 미시작 |
| BY 제외 분할 (`splits_no_by/`) 생성 | 미시작 |
| KP-8 컬럼 목록 코드 상수화 | 미시작 |
| 30fps×2s 전처리 파이프라인 | 추후 |
