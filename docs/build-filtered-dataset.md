# final_dataset_filtered.csv 생성 가이드

`scripts/build_filtered_dataset.py` 는 `final_dataset.csv` (또는 `final_dataset_3class.csv`) 에  
One-Euro Filter, EMA Filter, 파생 피처 재계산, **가속도 피처 추가**를 적용하여  
`final_dataset_filtered.csv` 를 생성하는 전처리 스크립트이다.

---

## 출력 스키마

| 컬럼 | 수 | 설명 |
|---|---|---|
| `video_id`, `frame`, `time_sec` | 3 | 메타데이터 |
| `kp0_y` … `kp16_s` | 51 | One-Euro 필터 적용된 키포인트 (y, x, confidence) |
| `HSSC_y`, `HSSC_x` | 2 | 상체 중심 좌표 (kp0–kp6 평균, 필터 후 재계산) |
| `RWHC` | 1 | 신체 바운딩박스 종횡비 (width / height) |
| `VHSSC` | 1 | 상체 중심 수직 속도 d(HSSC_y)/dt |
| **`AHSSC`** | **1** | **상체 중심 수직 가속도 d²(HSSC_y)/dt²** ← 신규 |
| **`AHSSC_x`** | **1** | **상체 중심 수평 가속도 d²(HSSC_x)/dt²** ← 신규 |
| `label` (+ `label_3class`) | 1–2 | 입력 파일에 있는 라벨 컬럼 |
| **합계** | **61** (2class) / **62** (3class) | |

> **가속도 피처 의미**
> - `AHSSC`  : 낙상 시 신체 중심이 빠르게 아래로 가속되는 순간을 포착.  
>   낙상 직전 급격한 양수값 → 착지 후 감속으로 음수값으로 전환되는 패턴.
> - `AHSSC_x`: 수평 방향 가속도. 미끄러짐 등 수평 이동이 수반되는 낙상 구분에 유용.

---

## 적용 처리 파이프라인

```
final_dataset.csv
        │
        ▼
[1] One-Euro Filter  (kp*_y, kp*_x)
        min_cutoff = 1.0 Hz  ← STM32 POSE_EURO_MIN_CUTOFF
        beta       = 0.5     ← STM32 POSE_EURO_BETA
        d_cutoff   = 1.0 Hz  ← STM32 POSE_EURO_D_CUTOFF
        │
        ▼
[2] EMA Filter  (kp*_s, 신뢰도)
        alpha = 0.5           ← STM32 POSE_CONF_EMA_ALPHA
        │
        ▼
[3] 파생 피처 재계산
        HSSC_y/x  ← kp0–kp6 평균 (필터 후)
        RWHC      ← max_x - min_x  /  max_y - min_y
        VHSSC     ← d(HSSC_y) / dt
        │
        ▼
[4] 가속도 추가  (신규)
        AHSSC     ← d(VHSSC) / dt  = d²(HSSC_y) / dt²
        AHSSC_x   ← d²(HSSC_x) / dt²
        │
        ▼
final_dataset_filtered.csv
```

> **파라미터 정합성**  
> 기본값은 STM32 C 코드의 `app_postprocess.c` 에 정의된 값과 동일하게 설정되어  
> 학습-배포 간 전처리 일관성을 보장한다.

---

## 실행 방법

### 기본 실행 (권장)

```bash
# 프로젝트 루트에서 실행
uv run python scripts/build_filtered_dataset.py
```

입력: `dataset/final_dataset.csv`  
출력: `dataset/final_dataset_filtered.csv`

---

### 3-class 버전 처리

```bash
uv run python scripts/build_filtered_dataset.py \
    --input  dataset/final_dataset_3class.csv \
    --output dataset/final_dataset_3class_filtered.csv
```

---

### 파라미터 조정 (실험용)

```bash
uv run python scripts/build_filtered_dataset.py \
    --input      dataset/final_dataset.csv \
    --output     dataset/final_dataset_filtered.csv \
    --min-cutoff 1.0 \
    --beta       0.5 \
    --d-cutoff   1.0 \
    --conf-alpha 0.5
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `--min-cutoff` | `1.0` | One-Euro 최소 컷오프 주파수 [Hz] |
| `--beta` | `0.5` | One-Euro 속도 민감도 (클수록 빠른 동작 추종) |
| `--d-cutoff` | `1.0` | One-Euro 미분 저역통과 컷오프 [Hz] |
| `--conf-alpha` | `0.5` | EMA 신뢰도 스무딩 계수 (0→강한 스무딩, 1→원본) |

---

## 예상 소요 시간 및 메모리

| 항목 | 값 |
|---|---|
| 입력 크기 | ~4.3 GB (`final_dataset_3class.csv`) |
| 총 row 수 | 4,093,620 |
| 총 video 수 | 20,380 |
| 예상 RAM 사용량 | ~8–10 GB |
| 예상 처리 시간 | 약 15–25 분 (CPU 단일 스레드) |
| 출력 크기 | ~4.7 GB |

> 메모리가 부족하면 video_id 목록을 분할해 여러 번 실행한 뒤 결과를 `pd.concat`으로 합칠 수 있다.

---

## 출력 검증

실행 완료 후 터미널에 다음 통계가 출력된다.

```
─── 완료 ──────────────────────────────────────────────────
  출력 rows : 4,093,620
  출력 cols : 61  (입력 59 + 신규 2개: ['AHSSC', 'AHSSC_x'])
  label 분포: {0: 3778880, 1: 314740}
  AHSSC        min=...  max=...  mean=...  std=...
  AHSSC_x      min=...  max=...  mean=...  std=...
─────────────────────────────────────────────────────────
```

확인 사항:
- `출력 cols`가 61 (2class) 또는 62 (3class) 인지 확인
- `label 분포`가 원본과 동일한지 확인
- `AHSSC` / `AHSSC_x` 의 범위가 비정상적으로 크지 않은지 확인  
  (정상 범위 기준: |AHSSC| < 500 정도)

---

## 학습 파이프라인과의 연결

생성된 `final_dataset_filtered.csv` 는 기존 학습 스크립트에 직접 투입할 수 있다.

```bash
# 3-class 학습 예시
uv run python scripts/unified_preprocessing.py \
    --input dataset/final_dataset_filtered.csv \
    --output dataset/final_filtered_3class.sqlite

uv run python scripts/train_comparison.py \
    --db dataset/final_filtered_3class.sqlite \
    --mode 3class
```

> `AHSSC` / `AHSSC_x` 컬럼이 추가되었으므로 학습 스크립트의  
> **feature 컬럼 목록 또는 `input_dim` 설정을 57로 업데이트**해야 한다.  
> (기존 55 → 필터링 후 57)

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `scripts/build_filtered_dataset.py` | 이 가이드의 대상 스크립트 |
| `scripts/apply_euro_filter_and_recompute_features.py` | 구버전 (AHSSC 없음, SQLite 출력 지원) |
| `docs/GRU-Implementation-Architecture.md` (STM32 레포) | 필터 파라미터 원본 정의 |
| `dataset/final_dataset.csv` | 입력 원본 (59 cols) |
| `dataset/final_dataset_filtered.csv` | 이 스크립트의 출력 (61 cols) |

---

*작성일: 2026-05-13*
