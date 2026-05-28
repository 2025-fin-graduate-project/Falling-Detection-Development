# Phase 43 Summary — Report Preparation

**기간**: 2026-05-21  
**상태**: 진행 중 (INT8 eval 실행 중)  
**결과 디렉토리**: `results/phase43_report/`

---

## 목적

Phase 41+42 ablation 실험 결과를 논문 형식으로 정리하기 위한 보고서 준비 단계. 추가 훈련 없음.

1. **STedgeAI analyze** — 상위 모델들의 Flash/MACC/Activation 측정
2. **INT8 host eval** — STedgeAI channel-wise INT8 PTQ 후 ev_minpr 측정
3. **시각화** — 논문용 차트 생성
4. **보고서 초안** — 변인별 결과 표 및 결론 작성

---

## 완료 작업

### STedgeAI Analyze (완료)

7개 대상 모델 전부 완료 (`export_stedgeai.py`, stm32n6 target):

| 모델 | Flash (KiB) | MACC | Activation (KiB) |
|---|---|---|---|
| P42-kp17-w60      | 290 | 4,371K | 31.5 |
| P41-kp13-w60      | 275 | 4,141K | 31.5 |
| P41-kp17-w40      | 290 | 2,915K | 21.5 |
| P41-vel-kp13-w40  | 312 | 3,132K | 21.5 |
| P41-raw-kp7-w40   | 253 | 2,531K | 21.5 |
| P42-kp13-w40-h128 | 590 | 5,844K | 33.0 |
| P42-kp13-w60-h128 | 590 | 8,764K | 48.0 |

### 시각화 (완료)

`scripts/plot_phase43_results.py` 실행 완료. `results/phase43_report/`에 저장:

| 파일 | 내용 |
|---|---|
| fig1_feature_set.png | Feature set(kp5~kp17) vs ev_minpr |
| fig2_window_size.png | Window size(20~60) vs ev_minpr |
| fig3_raw_vs_filt.png | Raw vs filtered preprocessing 비교 |
| fig4_velocity.png    | Velocity feature 유무 비교 |
| fig5_hidden_size.png | GRU(64,32) vs GRU(128,64) 비교 |
| fig6_tcn_vs_gru.png  | TCN vs GRU 아키텍처 비교 |
| results_table.csv    | 전체 결과 테이블 |
| top10_summary.txt    | P41+P42 합산 Top 10 |

### 문서화 (완료)

| 파일 | 내용 |
|---|---|
| `results/phase41_ablation/strategy.md` | Phase 41 변인별 분석 |
| `results/phase42_cross/strategy.md`   | Phase 42 교차 검증 분석 |
| `docs/tasks/phase41-summary.md`       | Phase 41 요약 |
| `docs/tasks/phase42-summary.md`       | Phase 42 요약 |
| `results/phase43_report/experiment_report.md` | 논문용 보고서 초안 |

---

## 진행 중

### INT8 Host Eval (진행 중)

`scripts/util/eval_stedgeai_host_p37.py` — tmux `p43int8` 세션에서 실행 중.

- 스크립트: `eval_stedgeai_host_p37.py` (train_window_phase37.py 포맷 호환)
- 대상: 7개 모델, eval-stride=5
- 예상 소요: 모델당 ~20분, 총 ~2.3시간
- 결과: `metrics.json["stedgeai_host_eval"]`에 저장

완료 후 `experiment_report.md`의 INT8 MinP 칸(TBD) 업데이트 예정.

---

## 주요 스크립트

| 스크립트 | 용도 |
|---|---|
| `scripts/util/export_stedgeai.py` | STedgeAI analyze + compat keras 생성 |
| `scripts/util/eval_stedgeai_host_p37.py` | INT8 host eval (phase37 포맷) |
| `scripts/plot_phase43_results.py` | 시각화 + 결과 테이블 생성 |
