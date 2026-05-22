# 3. 학습 설정 변인 분석

## 3.1 손실 함수 변인

### Focal Loss vs Cross-Entropy

**Focal Loss**: `FL(p_t) = α_t × (1-p_t)^γ × BCE`
- α: 클래스 불균형 가중치 (fall class weight)
- γ=2: 쉬운 샘플 다운웨이팅 (hard example mining 효과)

**Phase 33 비교 (val_video_min_pr, GRU(128,64), seed 여러 개):**

| 손실 | checkpoint | 대표 seed | event MinP | 비고 |
|------|-----------|---------|-----------|------|
| **Focal (α=0.25)** | val_vm | **42** | **0.9241** | P27-vm0 전체 최고 |
| CE | val_vm | 0 | 0.9038 | P33-gce-vm0 |
| **Focal (α=0.25)** | val_loss | **42** | **0.9181** | P30-lstm (LSTM) |
| CE | val_loss | 42 | 0.8912 | P33-lce-vl42 |

**Focal α 영향 분석 (gradient 비율)**:

| α 값 | fall gradient 비중 | nfall gradient 비중 | 효과 |
|-----|-----------------|-------------------|------|
| 0.10 | 매우 낮음 | 높음 | FP↓ but FN↑↑ (낙상 미검출 급증) |
| 0.25 | 25% | 75% | 비낙상 편향 → FN=18 병목 |
| 0.35~0.45 | 중간 | 중간 | Phase 35 탐색, 0.9068~0.9103 |
| 0.65 | 65% | 35% | window MinP **0.9515** (최고) but event FP 악화 |
| **0.75** | **75%** | **25%** | **event FP 최소, 채택** |

> α=0.65 역설: window 최고이나 최적 threshold 낮아져 event FP 오히려 증가
> → window MinP ≠ event MinP 최적 threshold 불일치 현상

**결론**: Focal loss > CE (일관). 배포 모델(P38-nv-a65): Focal α=0.65, γ=2.0 채택.

---

## 3.2 Checkpoint Monitor 변인

학습 중 어떤 지표로 최적 가중치를 저장할지 선택.

| Monitor | 설명 | GRU 성능 | LSTM 성능 |
|---------|------|---------|---------|
| `val_loss` | 검증 손실 최소화 | 0.9087 | **0.9181** |
| `val_video_min_pr` | 비디오 단위 MinP 최대화 | **0.9241** | 0.9038 |
| `val_event_min_pr` | 이벤트 단위 MinP 최대화 | 0.9227 | 0.8651 |

**아키텍처별 친화 checkpoint:**
- GRU ↔ `val_video_min_pr`: 목표 지표 직접 최적화로 가중치 선택 정확
- LSTM ↔ `val_loss`: 후처리 기반 지표의 노이즈에 LSTM이 과민반응

**Seed 의존성 (val_video_min_pr 기준, Phase 27/28):**

| Seed | event MinP | 비고 |
|------|-----------|------|
| **42** | **0.9241** | 전체 최고 (이상치 수준의 우세) |
| 1 | 0.9103 | |
| 2 | 0.9064 | |
| 0 | 0.9060 | |
| 5~123 (8개) | 0.878~0.905 | Phase 28 탐색 |

→ **16개 seed 탐색 후 seed=42만 0.91+ 달성** → 구조적 성능 한계(0.93 미달) 확인의 계기

---

## 3.3 Early Stopping (Patience) 변인

**Phase 26 비교 (GRU(128,64), kp7, 40f):**

| Patience | Epochs | val_event_min_pr | event MinP |
|---------|--------|-----------------|-----------|
| **5** | 30 | 노이즈에 강건 | **0.9227~0.9241** |
| 15 | 100 | spurious peak 과추적 | 0.9103 |

**분석**: val_event_min_pr은 에포크마다 노이즈가 크므로 patience=15는 spurious peak을 좋은 가중치로 오인.
작은 patience=5가 실질적으로 더 좋은 정규화 효과.

**결론**: GRU(128,64) + val_vm + patience=5 + epochs=30이 최적 학습 조건.
(Phase 36/37 이후 신 프레임워크: epochs=80, patience=10으로 변경)

---

## 3.4 Class Weight 변인

**class_weight**: 학습 시 클래스별 손실 가중치 자동 계산
- 비낙상 데이터가 낙상보다 많으면 non-fall에 높은 weight 부여

**Phase 35 비교 (class_weight ON vs OFF, val_vm, seed=42):**

| class_weight | Focal α | event MinP | FP | FN |
|-------------|---------|-----------|----|----|
| **ON (기본)** | **0.25** | **0.9241** | **13** | **18** |
| OFF | 0.40 | 0.9068 | 18 | 22 |
| OFF | 0.35 | 0.9103 | 19 | 21 |
| OFF | 0.45 | 0.9103 | 19 | 21 |

**결론**: class_weight ON이 일관되게 유리. α 조정으로 보상 불가.
→ class_weight + Focal α=0.25 조합이 이전 프레임워크에서 최적.

---

## 3.5 Hard Negative Mining 변인

**Hard negative**: val set에서 FP가 발생한 비낙상 영상의 window stride를 2→1로 축소,
오분류 영상을 더 많이 학습.

**Phase 11/14 결과:**

| 실험 | 변경 | event MinP | FP | FN |
|------|------|-----------|----|----|
| P9O-v01 (기준) | baseline | 0.9187 | 19 | 20 |
| P11-v04 | hard-neg 단독 | 0.8880 | 23 | 28 |
| P11-v05 | hard-neg + α=0.10 | 0.9061 | 23 | 22 |
| P14-v04 | hard-neg + α=0.15 | 0.9061 | 23 | 18 |

**결론**: Hard negative가 FP를 줄이지만 FN이 더 크게 증가 → MinP 하락.
threshold를 낮춰 FP를 악화시키는 부작용. **Phase 15부터 폐기**.

---

## 3.6 Negative Stride 변인

**train_negative_stride**: 비낙상 window 샘플링 간격 (1=전체, 2=절반, 5=20%)

| Stride | 비낙상 비중 | 효과 |
|--------|-----------|------|
| 1 | ~65% | FP↓ but FN↑↑ (낙상 gradient 부족) |
| **2** | **~40%** | **최적 균형** |
| 5 | ~25% | fall 편향 → FP↑ |

**Phase 13 결과 (α=0.10, hard-neg, stride=1 vs 2):**
- stride=1: FP=19 달성 but FN=26 (낙상 미검출 급증)
- stride=2: FP=23 but FN=19~22 (균형)

**결론**: neg_stride=2가 표준. 신 프레임워크(Phase 36+)에서도 nfall_stride=5 (더 공격적 필터링으로 학습 안정성 향상).

---

## 3.7 패러다임 전환: Event/Video-level → Window-level 학습

### 기존 방식 (Phase 1~35)

- 학습 목표: window별 예측 정확도
- **평가**: threshold × min_consecutive sweep → **비디오 단위 집계** → MinP
- Checkpoint: `val_video_min_pr` 또는 `val_event_min_pr`
- 문제: window 레이블이 낙상 이벤트 경계에서 노이즈 → 학습-평가 불일치

### 새 방식 (Phase 36~)

- **레이블**: Pure-window (margin=5) — 경계 윈도우 완전 배제
- **평가**: window 단위 MinP 직접 최적화
- Checkpoint: val_window_minpr
- 후처리: event_vote (sliding window 5개, 다수결 3/5)

| | 기존 방식 | 새 방식 |
|--|---------|---------|
| 최고 MinP (event/video) | 0.9241 (P27-vm0) | **0.9543** (P40, stateful) |
| 이슈 | 레이블 노이즈, checkpoint 과적합 | 없음 |
| 채택 여부 | 1~35 Phase | **36+ Phase, 최종** |

**결론**: 패러다임 전환이 가장 큰 성능 도약 요인 (0.9241 → 0.9543, +0.030).
Window-level pure label 학습 + stateful fine-tuning의 조합이 핵심.

---

## 3.8 Velocity Feature 변인 (Phase 38 Ablation)

### 속도 피처 구성

프레임 간 좌표 차분으로 속도 계산:
```
vel_i(t) = feat_i(t) - feat_i(t-1)    for i in {_y, _x 컬럼}
```
- kp13 기준: 29개 속도 피처 추가 → 입력 차원 45 → **74**
- 정규화: raw features 연결 후 전체 74차원 일괄 표준화

### Phase 38 Velocity Ablation 결과 (unified event MinP, test set)

| 모델 | 속도 피처 | window | event MinP | FP | FN |
|------|---------|--------|-----------|----|----|
| P38-nv-a65-gru | ✗ | 40f | 0.9545 | 18 | 10 |
| **P38-vel-a65-gru** | **✓** | **40f** | **0.9656** | **22** | **7** |
| P38-vel-drp4-gru (dropout=0.4) | ✓ | 40f | 0.9597 | 26 | 6 |
| P38-vel-h256-gru (GRU 256,128) | ✓ | 40f | 0.9643 | 23 | 3 |
| P38-vel-w35-gru | ✓ | 35f | 0.9567 | 28 | 6 |
| P38-vel-w30-a60-gru (α=0.60) | ✓ | 30f | 0.9612 | 25 | 6 |
| P38-vel-w30-drp35-gru | ✓ | 30f | 0.9540 | 30 | 3 |
| P38-vel-w30-h256-gru | ✓ | 30f | 0.9626 | 24 | 7 |
| P38-vel-w30-gru | ✓ | 30f | 0.9447 | 23 | 12 |

> unified eval 기준: val set threshold sweep (THRESHOLDS=0.05~0.95) → test set 평가

**주요 관찰:**
- velocity 추가로 event MinP **+0.011** 향상 (0.9545 → 0.9656)
- FN이 10→7 감소: 낙상 미검출 방지에 속도 정보가 직접 기여
- 40f window가 30f보다 일관되게 유리 (속도 신호 안정화에 긴 문맥 필요)
- dropout=0.4: 약간 과다 정규화로 FP 증가

### Phase 39 No-velocity 비교

| 모델 | GRU 크기 | event MinP | FP | FN |
|------|---------|-----------|----|----|
| P39-nv-a60 (α=0.60) | 128,64 | 0.9493 | 22 | 11 |
| P39-nv-a65-w30 (w30) | 128,64 | 0.9434 | 28 | 12 |
| **P39-nv-a65-h256 (h256)** | **256,128** | **0.9672** | **21** | **6** |

- h256 no-velocity (0.9672) > h128 with-velocity (0.9656): 모델 용량으로 속도 피처 효과 일부 대체 가능
- 그러나 GRU(256,128)은 Flash ~2,200 KiB, MACC 16.7M → 배포 불가 수준

**결론**: velocity 피처는 GRU(128,64) 기준 +0.011 성능 향상을 제공하나,
배포 모델(P40-nv-a65-stateful)은 MCU에서의 velocity 실시간 계산 추가 구현 부담으로
인해 no-velocity 경로를 유지함. 보고서에서 velocity ablation으로 명시 권장.
