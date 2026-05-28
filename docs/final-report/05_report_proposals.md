# 5. 보고서 작성 제안 및 참고 자료

## 5.1 보고서 섹션 구성 제안

```
1. 서론
   - 낙상 감지 중요성 (노인 인구 증가, 의료 비용)
   - 기존 접근법 한계 (카메라 기반 vs 웨어러블, 딥러닝 vs 규칙 기반)
   - 본 연구 기여: STM32N6 엣지 배포 + MoveNet 키포인트 기반 GRU

2. 관련 연구 (아래 5.4 참조)

3. 데이터셋 및 전처리
   3.1 데이터셋 구성 (01_dataset_preprocessing.md §1.1)
   3.2 키포인트 전처리: Raw vs Filtered (§1.2)
   3.3 특징 집합 선택: kp7 / kp12 / kp13 (§1.3)
   3.4 윈도우 크기: 60f / 40f / 30f (§1.4)
   3.5 윈도우 레이블링: mixed vs pure (§1.5)

4. 모델 아키텍처
   4.1 전체 구조 (Conv1D + GRU + Dense)
   4.2 시퀀스 모델 비교: GRU / LSTM / TCN (§2.3)
   4.3 Hidden size 탐색: (64,32) / (128,64) / (256,128) (§2.2)
   4.4 배포 제약: Unidirectional + Stateful (§2.4)

5. 학습 설정
   5.1 손실 함수: Focal vs CE, α 탐색 (§3.1)
   5.2 Checkpoint 전략 (§3.2)
   5.3 Class weight, Hard negative, Negative stride (§3.3~3.6)
   5.4 학습 패러다임 전환: Video→Window-level (§3.7)

6. 후처리 및 평가
   6.1 Threshold × Min-consecutive sweep (§4.1)
   6.2 Video/Event-level 평가 기준 (§4.1)
   6.3 Stateful Fine-tuning (§4.3)

7. 경량화 및 양자화
   7.1 STedgeAI INT8 양자화 (§4.4)
   7.2 모델 크기별 성능-효율 trade-off (§4.5)
   7.3 STM32N6 배포 사양 (§2.6, §4.6)

8. 결론 및 향후 연구
```

---

## 5.2 핵심 그래프/표 제안

### 반드시 포함할 것

1. **변인별 성능 비교 막대 그래프** (MinP 기준)
   - 전처리: Raw vs Filtered
   - 키포인트: kp7 / kp13 / kp13+vel
   - 윈도우 크기: 30f / 40f / 60f
   - 레이블링: mixed vs pure
   - 아키텍처: GRU(64,32) / (128,64) / (256,128)
   - 손실함수: CE / Focal α=0.25 / α=0.65 / α=0.75

2. **성능 향상 흐름 라인 그래프**
   ```
   Phase 1(0.9187) → Phase 21(0.9227) → Phase 27(0.9241)
     → Phase 37(0.9577 event, unified) → Phase 38-vel(0.9656 event, unified)
     → Phase 40-stateful(0.9543 event_vote)
   ```
   > Phase 37~40은 unified event MinP(min_consec=3) 기준, 동일 val-sweep threshold 적용

3. **Float vs INT8 MinP 비교표** (실험별)

4. **혼동 행렬 (Confusion Matrix)** — 최종 모델 (P40)
   - TP=615, TN=209, FP=19, FN=10
   - 카메라 방향별 recall (BY/FY/SY)

5. **모델 크기 vs 성능 산점도** (Flash KiB vs event MinP)

6. **Threshold sweep 곡선** — FallPrecision / NFallPrecision vs threshold

### 있으면 좋을 것

7. **Window MinP vs Event MinP 분리 그래프** — 두 지표의 상관관계 비교
8. **학습 곡선 (loss/val_minp)** — 대표 실험 (P37-pure-kp13-w40-gru)
9. **Stateful FT 에포크별 val_ev 변화** — P40 파인튜닝 효과 시각화
10. **시스템 다이어그램**: `카메라 → MoveNet(NPU) → GRU(CPU) → Event Vote → 낙상 알람`

---

## 5.3 추가 실험 제안 (우선순위 순)

| 우선순위 | 실험 | 소요 시간 | 보고서 가치 |
|---------|------|---------|-----------|
| ★★★ | **P40 INT8 threshold 재선택** (실행 중, ~완료) | — | INT8 MinP ≥ 0.90 달성 여부 확정 |
| ★★★ | **P40-h64 stateful FT** (실행 중, ep10/15) | ~1.5h | 경량화 섹션 완성 |
| ★★★ | **velocity 모델 ablation 정리** ✅ 완료 | — | 03_training_config §3.8 작성 완료 |
| ★★☆ | **P40-h64 INT8 eval** (h64 FT 완료 후) | 30분 | 경량화 INT8 성능 비교 |
| ★★☆ | **P38-vel-a65 event_vote eval** | 30분 CPU | velocity 최고 모델 공정 비교 (unified vs event_vote) |
| ★★☆ | **Raw vs Filtered (Phase37 기준)** | 30분 GPU | 전처리 변인 신 프레임워크 기준 정량화 |
| ★☆☆ | 방향별(BY/FY/SY) FN/FP 분포 분석 | 코드 1시간 | 데이터 편향 분석 강화 |

---

## 5.4 참고 문헌 (Reference Models & Papers)

### 키포인트 추출 모델

| 모델 | 출처 | 특징 |
|------|------|------|
| **MoveNet Lightning** | Google (2021) | 256×256 입력, 17 keypoints, 경량 NPU 호환 |
| MoveNet Thunder | Google (2021) | 더 정확하나 무거움 |
| PoseNet | Google (2018) | MoveNet 이전 버전 |
| BlazePose | Google/MediaPipe (2020) | 33 keypoints, 실시간 |
| HRNet | Wang et al. (2020) | 고정밀 2D pose estimation |
| OpenPose | Cao et al. (2017) | 다인원 pose |

### 낙상 감지 관련 논문

| 논문 | 내용 | 관련성 |
|------|------|--------|
| Núñez-Marcos et al. (2017) - "Vision-based fall detection with convolutional neural networks" | CNN 기반 낙상 감지 서베이 | 기존 방법 비교 |
| Nogas et al. (2020) - "DeepFall: Unsupervised learning for automatic fall detection" | 비지도 학습 기반 | 데이터 부족 문제 |
| Adhikari et al. (2017) - "Activity recognition for indoor fall detection using GRU-RNN" | GRU 기반 낙상 감지 | **직접 관련** |
| Xu et al. (2021) - "A real-time fall detection system using GRU" | GRU + 스마트폰 센서 | 비교 대상 |
| Thangal & Vaithiyanathan (2022) - "Skeleton-based fall detection" | 키포인트 기반 | **직접 관련** |
| Ramirez et al. (2021) - "Pose-based fall detection using LSTM" | LSTM + pose keypoints | LSTM 비교 근거 |

### 엣지 AI / 임베디드 배포

| 논문/자료 | 내용 |
|----------|------|
| Liberis et al. (2019) - "μNAS: Constrained Neural Architecture Search for Microcontrollers" | MCU 대상 NAS |
| STMicroelectronics STEdgeAI 4.0 docs | STM32N6 배포 툴체인 |
| Banbury et al. (2021) - "MLPerf Tiny Benchmark" | 임베디드 ML 벤치마크 |

### Focal Loss

| 자료 | 내용 |
|------|------|
| Lin et al. (2017) - "Focal Loss for Dense Object Detection" (RetinaNet) | Focal loss 원 논문 |

---

## 5.5 보고서 작성 시 주의사항

### 지표 표기 통일

```
MinP (Primary)  = min(FallPrecision, NFallPrecision)
MinPR (Strict)  = min(FallPrecision, NFallPrecision, FallRecall, NFallRecall)
```
→ Phase 9까지 MinPR 사용, Phase 10 이후 MinP로 전환. 보고서에서 명확히 구분.

### 평가 프레임워크 구분

| 버전 | 데이터셋 | 집계 단위 | 사용 Phase |
|------|---------|---------|-----------|
| 구 (v1) | splits_v2 | Video MinP | Phase 1~19 |
| 신 (v2) | class_balanced | Event MinP | Phase 20~35 |
| 최신 (v3) | class_balanced | Window MinP + Event vote | Phase 36~ |

→ **각 Phase의 수치를 직접 비교하지 말 것.** 같은 평가 기준 내에서만 비교.

### 성능 목표 (CLAUDE.md 기준)

| 단계 | 목표 |
|------|------|
| Float MinP | ≥ 0.93 |
| INT8 MinP | ≥ 0.90 |
| 현재 달성 | Float 0.9543 ✅ / INT8 측정 중 |

---

## 5.6 실험 진행 타임라인

```
Phase 1~5   : 초기 그리드 탐색 (60f→40f, raw→filtered, GRU 크기)
Phase 6~9   : PostProcess 최적화, MinPR 기준 확립
Phase 10~19 : Hard-negative, LB-3, 속도 피처, TCN, LSTM 탐색
Phase 20~22 : Class-balanced split, event-level 평가 전환
Phase 23~29 : Seed sweep, alpha 탐색 → 구조적 한계 확인 (0.93 미달)
Phase 30~33 : val_loss 체인, CE vs Focal, 아키텍처 재탐색
Phase 34~35 : alpha 상향, no-class-weight 탐색
Phase 36~37 : [패러다임 전환] Window-level pure label, 변인 ablation
Phase 38~39 : alpha/velocity/window 조합 탐색
Phase 40    : Stateful fine-tuning → 0.9543 달성 ✅
```
