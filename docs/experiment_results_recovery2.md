# 실험 결과 복구 기록
> 2026-05-22 세션 기록 기반 재구성. results/ 디렉토리 소실로 인해 작성.

---

## Phase 41 Ablation — Float 성능

| 모델 | Feature | Window | ev_minpr (float) | threshold | postproc |
|---|---|---|---|---|---|
| P41-kp13-w60 | kp13 filtered | 60 | 0.9167 | 0.75 | v10k6 |
| P41-vel-kp13-w40 | kp13+vel filtered | 40 | 0.9123 | 0.725 | v10k6 |
| P41-kp17-w40 | kp17 filtered | 40 | 0.9123 | 0.70 | v10k6 |
| P41-raw-kp7-w40 | kp7 raw | 40 | 0.9167 | 0.70 | v10k6 |
| P41-raw-kp13-w60 | kp13 raw | 60 | 0.9123 | - | - |
| P41-raw-kp13-w40 | kp13 raw | 40 | 0.9079 | - | - |
| P41-kp13-w30 | kp13 filtered | 30 | 0.9079 | - | - |
| P41-kp9-w40 | kp9 filtered | 40 | 0.8947 | - | - |
| P41-kp7-w40 | kp7 filtered | 40 | 0.8904 | - | - |
| P41-kp11-w40 | kp11 filtered | 40 | 0.8640 | - | - |
| P41-kp13-w20 | kp13 filtered | 20 | 0.8724 | - | - |
| P41-vel-kp7-w40 | kp7+vel filtered | 40 | 0.8684 | - | - |
| P41-tcn-kp13-w40 | TCN kp13 | 40 | 0.8684 | - | - |
| P41-tcn-kp7-w40 | TCN kp7 | 40 | 0.8684 | - | - |

## Phase 42 Cross — Float 성능

| 모델 | 조건 | ev_minpr (float) |
|---|---|---|
| P42-kp17-w60 | kp17, w=60, filtered | 0.9211 |
| P42-kp13-w40-h128 | kp13, w=40, GRU h=128, filtered | 0.8991 |
| P42-kp13-w60-h128 | kp13, w=60, GRU h=128, filtered | 0.9035 |
| P42-kp13-w40 | kp13, w=40, filtered | 0.9035 |
| P42-raw-kp7-w60 | kp7, w=60, raw | 0.9123 |
| P42-raw-kp17-w40 | kp17, w=40, raw | 0.9123 |
| P42-raw-kp13-vel | kp13+vel, raw | 0.8728 |
| P42-vel-kp13-w60 | kp13+vel, w=60, filtered | 0.8684 |

---

## INT8 평가 결과 (STedgeAI validate --mode host, stride=20, v1k1 최적)

| 모델 | INT8 MinP | Flash (KiB) | MACC | Activation (KiB) |
|---|---|---|---|---|
| P41-kp13-w60 | **0.9298** | 275.38 | 4,140,832 | 31.5 |
| P41-vel-kp13-w40 | **0.9211** | 311.63 | 3,132,192 | - |
| P41-kp17-w40 | **0.9167** | 290.38 | 2,914,592 | - |
| P42-kp17-w60 | 0.9123 | 290.38 | 4,371,xxx | - |
| P42-kp13-w40-h128 | 0.9123 | 590.01 | 5,843,xxx | 48.0 |
| P41-raw-kp7-w40 | 0.9004 | 252.88 | 2,530,592 | - |
| P42-kp13-w60-h128 | 0.8860 | 590.01 | 8,763,552 | 48.0 |

> stride=20+v10k6 조합은 MinPR=0 버그 → reeval_int8_from_cache.py --scan-all --stride 20으로 수정 (v1k1 최적)

---

## expextra 결과 (추가 실험)

| 모델 | 조건 | ev_minpr |
|---|---|---|
| kp7-w60-raw | kp7, w=60, raw | 0.9035 |
| kp9-w40-raw | kp9, w=40, raw | 0.8991 |
| kp7-h128-w40-raw | kp7, w=40, GRU h=128, raw | 0.8904 |

---

## Exp4 씨드 안정성 (kp7-w40-raw, 6 seeds)

| Seed | ev_minpr | threshold | postproc |
|---|---|---|---|
| 0 | 0.9123 | 0.700 | v10k6 |
| 7 | 0.9035 | 0.650 | v7k5 |
| 13 | 0.8816 | 0.700 | v10k6 |
| 21 | 0.9254 | 0.725 | v7k5 |
| 37 | 0.9254 | 0.650 | v10k6 |
| 99 | 0.9123 | 0.725 | v10k6 |

- Mean = 0.9101 ± 0.0149
- Range = [0.8816, 0.9254]
- seed=13이 0.88로 INT8 목표(0.90) 미달 → seed 의존성 경고

---

## Conv-GRU 분리 PoC (P41-kp13-w60 기준)

### 수치 검증
- GRU 격리 최대 오차: 0.00e+00 ✓ (완전 일치)
- Conv receptive field: 9프레임

### STedgeAI generate 결과

| 서브모델 | generate | Flash (KiB) | Activation (KiB) | States | MACC |
|---|---|---|---|---|---|
| 원본 전체 (w=60) | ✅ | 275.38 | 31.5 | - | 4,140,832/window |
| Conv (9f→64) | ✅ | 136.75 | 5.75 | - | 315,200 |
| GRU (non-stateful) | ✅ | 138.63 | 2.0 | - | 35,168 |
| GRU (stateful, Keras3.13) | ❌ | - | - | - | Keras 버전 충돌 |

### 스트리밍 MACC 비교

| 방식 | MACC/frame | 레이턴시 |
|---|---|---|
| 원본 stride=1 (전체 매 프레임) | 4,140,832 | 없음 |
| **분리 스트리밍** | **350,368** | **없음** |
| 원본 stride=10 | 414,083 | 10f (0.67s) |

→ 분리 스트리밍이 원본 대비 약 12배 효율적

---

## Phase 44 Keras 3.7 재학습 (진행 중 중단)

목적: STedgeAI stateful GRU generate 호환성 확보

| 모델 | 상태 | val_minpr (최고) | early stop |
|---|---|---|---|
| P44-kp13-w60 | 학습 완료 (metrics.json 미생성 시점 중단) | 0.9613 (ep14) | ep29 |
| P44-vel-kp13-w40 | 미시작 | - | - |
| P44-kp17-w40 | 미시작 | - | - |

> 재실행 명령: `bash scripts/run_phase44_keras37.sh`

---

## 포팅 후보 최종 정리

| 순위 | 모델 | INT8 MinP | Flash | MACC/frame(split) | 비고 |
|---|---|---|---|---|---|
| 1 | P41-kp13-w60 | **0.9298** | 275 KiB | 350K | 최고 INT8 성능 |
| 2 | P41-vel-kp13-w40 | 0.9211 | 312 KiB | - | 속도 우선 |
| 3 | P41-kp17-w40 | 0.9167 | 290 KiB | - | 최소 latency |

---

## 추가 실험 완료 현황

| 실험 | 상태 |
|---|---|
| Exp1 카메라 방향 | ✅ 완료 |
| Exp2 필터 단계 | ✅ 완료 (§8.2) |
| Exp3 | ✅ 완료 |
| Exp4 씨드 안정성 | ✅ 완료 (§8.5) |
| Exp6 | ✅ 완료 |
| Exp7 지연시간 | ✅ 완료 (§8.7) |
| Exp9 Stateful FT | ⛔ 미시행 처리 |
| expextra (3개) | ✅ 완료 |
| INT8 eval (7모델) | ✅ 완료 |
