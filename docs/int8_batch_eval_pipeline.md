# INT8 배치 평가 파이프라인

**목적**: STM32N6 포팅 가능한 모델들의 INT8 성능을 일괄 측정하여 배포 후보 선정

---

## 대상 모델 기준

| 조건 | 값 |
|---|---|
| Float32 event MinPR | ≥ 0.90 |
| 아키텍처 | GRU 또는 LSTM (unidirectional) |
| 제외 | bidirectional, temporal_attention, TCN |
| 총 대상 | 22개 모델 |

---

## 파이프라인 구성

```
model.keras
    │
    ▼ (1단계)
STedgeAI analyze          ← export_stedgeai.py
    │  - Flash(KiB), RAM(KiB), MACC 확인
    │  - analyze_ok 검증
    │  - compat .keras 생성 (quantization_config 제거)
    ▼
INT8 compat .keras
    │
    ▼ (2단계)
STedgeAI validate         ← eval_stedgeai_host.py
  --mode host               (stm32h7 proxy, stm32n6 미지원)
  --target stm32h7
    │  - val set로 INT8 threshold 재선택 (--reselect-threshold)
    │  - test set INT8 추론 → video-level MinPR 계산
    ▼
metrics.json["stedgeai_host_eval"]
    {
      "eval_ok": true,
      "min_precision": 0.XXXX,
      "fall_precision": 0.XXXX,
      "nfall_precision": 0.XXXX,
      "confusion_matrix": [[TN,FP],[FN,TP]],
      ...
    }
```

---

## 실행 방법

```bash
# 기본 실행 (stride=10, ~20분/모델)
bash scripts/run_int8_batch_eval.sh

# 빠른 근사 (stride=50, ~5분/모델)
EVAL_STRIDE=50 bash scripts/run_int8_batch_eval.sh

# 고정밀 (stride=5, ~48분/모델) — 최종 검증용
EVAL_STRIDE=5 bash scripts/run_int8_batch_eval.sh
```

결과: `results/quantization/int8_eval_summary.tsv`

---

## 평가 지표

| 지표 | 설명 | 목표 |
|---|---|---|
| **INT8 event MinPR** | min(fall_pr, nfall_pr) @ test event level | **≥ 0.90** |
| Float→INT8 손실 | float_minpr - int8_minpr | ≤ 0.03 허용 |
| Flash | weights 크기 (KiB) | ≤ 60,000 KiB (여유 충분) |
| MACC/frame | 연산량 | Cortex-M55 기준 충분 |

---

## 대상 모델 목록 (float MinPR 기준 정렬)

| # | 모델 ID | Float MinPR | FN | FP | 아키텍처 | 비고 |
|---|---|---|---|---|---|---|
| 1 | P27-vm0 | **0.9241** | 18 | 13 | GRU(128,64) kp7 40f | 전체 최고, analyze 완료 |
| 2 | P21-v01 | 0.9227 | 18 | 17 | GRU(128,64) kp7 40f | |
| 3 | P30-lstm | 0.9181 | 19 | 19 | LSTM(128,64) kp7 40f | LSTM 최고 |
| 4 | P20-v03 | 0.9142 | 20 | 19 | GRU(128,64) kp7 40f | |
| 5 | P32-ce-42 | 0.9138 | 17 | 20 | GRU(128,64) kp7 40f | CE 손실, FN=17 |
| 6 | P20-v01 | 0.9121 | 21 | 14 | GRU(128,64) kp7 40f | |
| 7 | P32-fl-1 | 0.9110 | 21 | 17 | GRU(128,64) kp7 40f | |
| 8 | P27-s1 | 0.9103 | 21 | 19 | GRU(128,64) kp7 40f | seed=1 |
| 9 | P26-v01 | 0.9103 | 21 | 19 | GRU(128,64) kp7 40f | |
| 10 | P30-gru | 0.9087 | 22 | 13 | GRU(128,64) kp7 40f | |
| 11 | P35-ncw | 0.9068 | 22 | 18 | GRU(128,64) kp7 40f | no-class-weight |
| 12 | P34-a35-fm10 | 0.9064 | 22 | 19 | GRU(128,64) kp7 40f | α=0.35+mask |
| 13 | P27-s2 | 0.9064 | 22 | 19 | GRU(128,64) kp7 40f | seed=2 |
| 14 | P23-v01 | 0.9064 | 22 | 19 | GRU(256,128) kp7 40f | 대형 모델 |
| 15 | P27-s0 | 0.9060 | 22 | 20 | GRU(128,64) kp7 40f | seed=0 |
| 16 | P28-vm7 | 0.9052 | 21 | 22 | GRU(128,64) kp7 40f | |
| 17 | P33-gce-vm0 | 0.9038 | 23 | 16 | GRU(128,64) kp7 40f | CE 손실 |
| 18 | P31-vm42 | 0.9038 | 23 | 16 | LSTM(128,64) kp7 40f | LSTM |
| 19 | P19-v02 | 0.9020 | 23 | 24 | GRU(256,128) kp7 40f | 대형 모델 |
| 20 | P28-vm77 | 0.9004 | 24 | 15 | GRU(128,64) kp7 40f | |
| 21 | P29-a35 | 0.9000 | 24 | 16 | GRU(128,64) kp7 40f | α=0.35 |
| 22 | P28-vm123 | 0.9000 | 24 | 16 | GRU(128,64) kp7 40f | |

---

## 주요 파일

| 파일 | 역할 |
|---|---|
| `scripts/run_int8_batch_eval.sh` | 배치 실행 스크립트 |
| `scripts/util/export_stedgeai.py` | STedgeAI analyze + compat .keras 생성 |
| `scripts/util/eval_stedgeai_host.py` | INT8 host eval, metrics.json 업데이트 |
| `results/quantization/int8_eval_summary.tsv` | 전체 결과 TSV (배치 완료 후) |

---

## 알려진 제한사항

| 항목 | 내용 |
|---|---|
| stm32n6 host 미지원 | STedgeAI 4.0이 stm32n6 --mode host 미지원. stm32h7 proxy 사용 |
| 채널 순서 차이 | stm32h7(Cortex-M7) vs stm32n6(Cortex-M55) 정확도 수치 동일, 실행속도만 다름 |
| INT8 threshold 재선택 | val set INT8 점수로 threshold 재탐색 → float threshold와 다를 수 있음 |
| eval_stride 영향 | stride가 클수록 빠르지만 video coverage 낮아짐 (stride=5 권장, stride=10 실용적) |

---

## eval_stedgeai_host.py 버그 수정 이력

| 날짜 | 버그 | 수정 |
|---|---|---|
| 2026-05-17 | CSV glob 패턴 불일치 (`_val_output*` → `_val_c_outputs_*`) | STedgeAI 4.0 출력 파일명 변경 반영 |
| 2026-05-17 | CSV 파싱 실패 (`# dtype=float32` 주석 라인) | `#` 시작 라인 및 비숫자 라인 스킵 처리 |
