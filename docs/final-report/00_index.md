# 졸업 프로젝트 보고서 자료 인덱스

> **마감**: 2026-05-30  
> **보고서 요구사항**: 모델별·필터별·키포인트별·프레임별 변인 분석,
> 데이터셋·전처리·학습·후처리 단계별 유의미한 데이터 변화 기술

---

## 문서 목록

| 파일 | 내용 | 상태 |
|------|------|------|
| [01_dataset_preprocessing.md](01_dataset_preprocessing.md) | 데이터셋 통계 + 전처리 변인 (raw/filtered, kp 집합, 윈도우 크기, 레이블링) | ✅ 완료 |
| [02_model_architecture.md](02_model_architecture.md) | 아키텍처 비교 (GRU/LSTM/TCN, hidden size, bidir, velocity) | ✅ 완료 |
| [03_training_config.md](03_training_config.md) | 학습 설정 변인 (loss, checkpoint, class weight, negative stride, 패러다임 전환) | ✅ 완료 |
| [04_postprocessing_quantization.md](04_postprocessing_quantization.md) | 후처리·양자화·경량화·stateful 추론·INT8 결과 | ✅ 완료 (INT8 TBD) |
| [05_report_proposals.md](05_report_proposals.md) | 보고서 작성 제안 + 그래프 목록 + 참고 문헌 | ✅ 완료 |
| [06_ablation_study.md](06_ablation_study.md) | Phase 41/42 체계적 변인 탐색 (kp 집합·window·arch·vel·prep·model size 전체) | ✅ 완료 (INT8 TBD) |

---

## 핵심 성능 흐름 (ev_minpr 기준)

```
Phase 1~5 (기초 탐색)      Phase 20~35 (class-balanced)  Phase 36~40 (Window-level 패러다임)
   0.9187 (video MinP)         0.9241 (P27-vm0)                0.9543 (P40, stateful FT)
   GRU(256,128), kp12          GRU(128,64), kp7, seed=42       GRU(128,64), kp13, no-vel

                     Phase 41/42 (Ablation, GRU(64,32))
                         0.9211 (P42-kp17-w60)
                         Flash 290K, MACC 4.37M
```

> Phase 40과 Phase 41/42는 직접 비교 불가 (모델 크기, 훈련 방식 다름).
> Phase 40: GRU(128,64) + stateful FT = 성능 최고
> Phase 41/42: GRU(64,32) ablation = 경량 경로 검증

---

## 최종 모델 후보

| 우선순위 | 모델 | Float ev_minpr | Flash | MACC | 특징 |
|---|---|---|---|---|---|
| 성능 우선 | P40-nv-a65-stateful | **0.9543** | 590K | 5.84M | GRU(128,64), stateful FT |
| 경량 최고 | P42-kp17-w60 | 0.9211 | 290K | 4.37M | GRU(64,32), kp17, w=60 |
| 최경량 | P41-raw-kp7-w40 | 0.9167 | **253K** | **2.53M** | GRU(64,32), kp7, raw |

---

## 변인별 최종 결론 요약

| 변인 | 결론 | 근거 Phase |
|---|---|---|
| Feature set | kp17 최고 (0.9123↑), kp7 raw 사용 시 동급 | Phase 41 |
| Window size | 클수록 유리, w=60 최고 | Phase 41/42 |
| Architecture | GRU > TCN 일관 | Phase 30, 41 |
| Velocity | 효과 불안정 (GRU(64,32)에서 kp7 해로움) | Phase 38~42 |
| Preprocessing | kp7: raw +0.026, kp13/kp17: 동등 | Phase 41/42 |
| Model size | GRU(64,32) = GRU(128,64) (ablation 기준) | Phase 42 |
| Label | LB-2 > LB-3 (FN 증가) | Phase 12 |
| Window label | Pure (margin=5) >> Mixed (+0.156) | Phase 36~37 |
| Loss | Focal > CE, α 선택 중요 | Phase 33~35 |
| Checkpoint | GRU: val_vm 최적 | Phase 27~30 |

---

## 실험 타임라인

```
Phase 1~5   : 초기 그리드 탐색 (60f→40f, raw→filtered, GRU 크기)
Phase 6~9   : PostProcess 최적화, MinPR 기준 확립
Phase 10~19 : Hard-negative, LB-3, 속도 피처, TCN, LSTM 탐색
Phase 20~22 : Class-balanced split 도입, event-level 평가 전환
Phase 23~29 : Seed sweep, alpha 탐색 → 구조적 한계(0.93 미달) 확인
Phase 30~33 : val_loss 체인, CE vs Focal, 아키텍처 재탐색
Phase 34~35 : alpha 상향, no-class-weight 탐색
Phase 36~37 : [패러다임 전환] Window-level pure label
Phase 38~39 : alpha/velocity/window 조합 탐색
Phase 40    : Stateful fine-tuning → 0.9543 달성 ✅
Phase 41    : [Ablation] Feature set / window / arch / velocity / prep 단독 변인 탐색
Phase 42    : [Cross Validation] 최적 조합 교차 검증, h128 비교
Phase 43    : [Report Prep] STedgeAI analyze + INT8 eval + 시각화 ← 진행 중
```

---

## 현재 진행 중 (2026-05-21)

| 작업 | 상태 | 예상 완료 |
|------|------|---------|
| INT8 host eval (7개 모델) | tmux p43int8 실행 중 | ~2시간 |
| experiment_report.md INT8 칸 | eval 완료 후 업데이트 예정 | — |
| 06_ablation_study.md INT8 섹션 | eval 완료 후 업데이트 예정 | — |
