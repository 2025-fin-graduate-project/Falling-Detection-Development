# 4. 후처리·경량화·양자화 분석

## 4.1 후처리: Threshold × Min-Consecutive Sweep

### 메커니즘

```
window scores [0.3, 0.8, 0.9, 0.7, 0.4, 0.85, 0.92, ...]
      ↓ threshold=0.75
binary       [0,   1,   1,   0,   0,   1,    1,  ...]
      ↓ min_consecutive=3
filtered     [0,   0,   1*,  0,   0,   0,    1*, ...]  (* 연속 3개 충족 시점)
      ↓ 비디오 내 1개라도 1 → 낙상 감지
```

### 스윕 범위 및 선택 기준

| 파라미터 | 탐색 범위 | 선택 기준 |
|---------|---------|---------|
| threshold | 0.05~0.95 (37개) | val set 기준 MinPR ≥ 0.90 중 MaxMinPR |
| min_consecutive | {1, 3, 5} | 동일 |

### Threshold 선택 기준: Video-level vs Event-level (Phase 20/21)

| 선택 기준 | 대표 실험 | event MinP | video MinP | 차이 |
|---------|---------|-----------|-----------|------|
| Video-level | P20-v01 | 0.9121 | 0.9160 | — |
| **Event-level** | **P21-v01** | **0.9227** | **0.9267** | **+0.011** |

**이벤트 허용 윈도우(tolerance=2)**: 낙상 감지가 실제 낙상 이벤트 전후 2윈도우 이내면 정답으로 인정.
→ 더 실제적인 평가 기준, 성능 +0.011 향상.

---

## 4.2 후처리: Event Vote (Phase 36+ 표준)

Window-level 학습 전환 후 새로운 후처리 방식.

```
window scores (연속): [0.8, 0.9, 0.7, 0.85, 0.6]
vote_window=5, vote_k=3, threshold=0.75

→ 각 점수 ≥ 0.75: [1, 1, 0, 1, 0]
→ 합계=3 ≥ vote_k=3 → 낙상 감지!
```

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| vote_window | 5 | 직전 5개 윈도우 관찰 |
| vote_k | 3 | 3개 이상 fall 판정 시 알람 |
| threshold | 0.65~0.75 | 모델 출력 이진화 경계 |

**최적 threshold (Phase 40, float)**: 0.75 (val set sweep 기준)

---

## 4.3 Stateful Fine-tuning (Phase 40)

### 배경

Window batch 학습 vs 온디바이스 추론의 불일치:
- 학습: 각 윈도우를 독립적으로 처리 (GRU hidden state 초기화)
- 온디바이스: 매 윈도우마다 GRU state 누적 (연속 영상 컨텍스트 활용)

### 방법

1. 기학습 GRU(128,64) 모델 (P38-nv-a65-gru)을 베이스로 로드
2. GRU를 `stateful=True`로 재구성 (batch_shape=(1, 40, 45) 고정)
3. Conv1D 동결 (freeze), GRU + Dense만 파인튜닝
4. 학습: 비디오별 순서 보장, 비디오 경계에서 `reset_states()`
5. 평가: proxy(stateless) 모델으로 event_vote 기준

### 결과

| 단계 | event_vote MinP | FP | FN | threshold |
|------|----------------|----|----|----------|
| Phase 37 기준 (window batch) | 0.8816 | 28 | 7 | — |
| Phase 38-nv-a65 base | 0.9079 | 21 | 12 | — |
| **P40-nv-a65-stateful (fine-tuned)** | **0.9543** | **19** | **10** | **0.75** |
| 향상 | **+0.073** | **-9** | **-2** | |

> val event_vote: 0.9598 (최고 val MinP 기록)

### 경량 모델 Stateful Fine-tuning

| 모델 | 베이스 event_vote | FT 후 event_vote | FP | FN | Flash | MACC |
|------|----------------|----------------|----|----|-------|------|
| GRU(128,64) P40-nv-a65 | 0.9079 | **0.9543** | 19 | 10 | 590 KiB | 5.84M |
| GRU(64,32) P40-h64 | 0.9452 (unified) | **0.9495** | 21 | 11 | ~138 KiB | ~1.4M |

> GRU(64,32) FT 효과: +0.004 (미미) — 베이스 모델이 이미 최적에 근접, 최고 가중치 = 베이스라인 epoch
> GRU(128,64) FT 효과: +0.046 — threshold 재선택 효과 포함

---

## 4.4 INT8 양자화 (STedgeAI Channel-wise PTQ)

### 양자화 파이프라인

```
model.keras  →  STedgeAI generate  →  C code + INT8 weights  →  STM32N6 Flash
```

- **STedgeAI channel-wise INT8**: TFLite per-tensor보다 정밀도 손실 적음
- **평가 방법**: STedgeAI validate --mode host (stm32h7 proxy, stm32n6 미지원)

### Float → INT8 성능 비교

| 실험 | Float MinP | INT8 MinP | 손실 | Flash(KiB) | MACC |
|------|-----------|---------|------|-----------|------|
| P21-v01 | 0.9227 | **0.9181** | -0.005 | 537 | 5.32M |
| P27-vm0 | 0.9241 | 0.9138 | -0.010 | 537 | 5.32M |
| P30-lstm | 0.9181 | 0.9145 | -0.004 | 681 | 6.82M |
| P35-ncw | 0.9068 | **0.9336** | +0.027† | 537 | 5.32M |
| P20-v01 | 0.9121 | 0.9224 | +0.010† | 537 | 5.32M |
| **P40-nv-a65-stateful** | **0.9543** | **0.8967** | **-0.058** | 590 | 5.84M |

> †: INT8이 float보다 높은 이상치 — threshold 재선택 효과로 추정
> ‡: P40 INT8 손실이 큰 이유 — stateful FT 후 float threshold(0.75)를 그대로 적용 시 FN 급증(10→25).
>    threshold 재선택(INT8 val 기준) 시 개선 예상. [INT8 FP=11, FN=25, fall_pr=0.982, nfall_pr=0.897]

**평균 Float→INT8 손실**: 약 -0.005~-0.010 MinP (P40은 stateful FT 특성상 -0.058로 예외적으로 큼)
**목표**: INT8 MinP ≥ 0.90 → P40 현재 0.8967 (threshold 재선택 미적용 기준, 목표 미달)

### STedgeAI Analyze 결과 (GRU(128,64) kp13)

| 항목 | GRU(128,64) kp7 | GRU(128,64) kp13 |
|------|----------------|----------------|
| Flash (KiB) | 537 | **590** |
| 활성화 RAM (KiB) | ~35 | **33** |
| MACC/window | 5,322,472 | **5,843,872** |
| analyze_ok | ✅ | ✅ |

---

## 4.5 경량화 비교 (모델 크기별)

### 파라미터 및 배포 비용

| 모델 | 총 파라미터 | Flash(KiB) | RAM(KiB) | MACC | event MinP (float) |
|------|-----------|-----------|---------|------|-------------------|
| GRU(64,32) | 70,498 | ~138 | ~12 | ~1.4M | **0.9495** (P40-h64-stateful, FP=21, FN=11) — base 0.9452 대비 +0.004 |
| **GRU(128,64)** | **151,042** | **590** | **33** | **5.84M** | **0.9543** ✅ |
| GRU(256,128) | ~603,000 | ~2,200 | ~120 | 16.7M | 0.9672 (P39-nv-a65-h256, no FT) — 배포 불가 수준 |

### 추론 속도 추정 (Cortex-M55 @ 400MHz)

| 모델 | MACC | 추정 추론시간/window | 40f=2.7s 내 가능 여부 |
|------|------|-------------------|---------------------|
| GRU(64,32) | ~1.4M | ~0.9ms | ✅ 매우 여유 |
| **GRU(128,64)** | **5.84M** | **~3.7ms** | **✅ 여유** |
| GRU(256,128) | 16.7M | ~10.4ms | ⚠️ 실측 필요 |

> 온디바이스 실측치 없음 — STedgeAI analyze MACC 기준 추정
> GRU(256,128)은 15fps에서 window마다 10ms → 지연 없으나 실측 권고

---

## 4.6 v26 PoC 배포 모델 (참고용)

기존 STM32N6 PoC 배포 모델 (성능 최적화 대상 아님, 포팅 가능성 확인용):

| 항목 | 값 |
|------|-----|
| 모델 | gru_v26_int8.tflite (kp7 기반, GRU(64,32)) |
| Flash | ~150 KiB |
| 활성화 RAM | 2,816 B |
| MACC/frame | 37,378 |
| threshold | ≥0.65, 인물 미감지 45f → reset |
| 온디바이스 성능 | 미평가 (포팅 가능성만 확인) |

**현재 배포 목표 모델**: P40-nv-a65-stateful (GRU(128,64), kp13, event_vote MinP=0.9543)
→ `.keras` → STedgeAI 직접 경로로 포팅 예정

---

## 4.7 Phase 41/42 — K-of-N Vote Sweep (새 후처리 방식)

Phase 41/42에서 post-processing을 슬라이딩 투표창 방식으로 일반화.

```
VOTE_COMBOS = [(1,1),(3,2),(5,3),(5,4),(7,4),(7,5),(10,6)]
               (vote_window, vote_k) 조합
```

- `vote_window`: 관찰할 직전 window 수
- `vote_k`: 그 중 몇 개 이상 fall 판정 시 알람

val set에서 7개 조합을 모두 평가하여 ev_minpr이 최고인 (vw, vk)를 선택 후 test 적용.

**Phase 41/42 대부분의 최적 조합**: **(v10, k6)** — 10프레임 창에서 6개 이상 양성.  
15fps 기준 10 window = 0.67초 확인 시간.

이전 방식(`min_consecutive=3/5`)과 비교:
- `min_consecutive=3`은 3개 연속(연속 조건), `v5k3`은 5개 중 3개(비연속 허용)
- K-of-N이 더 유연하여 일시적 점수 하락에 강건

## 4.8 Phase 43 — INT8 양자화 결과 (GRU(64,32), class-balanced split)

STedgeAI 4.0 channel-wise INT8 PTQ. 평가: STedgeAI validate --mode host (stm32h7 proxy).

### Hardware Footprint (stm32n6 analyze)

| 모델 | Flash (KiB) | MACC | Activation (KiB) |
|---|---|---|---|
| P42-kp17-w60      | 290 | 4,371K | 31.5 |
| P41-kp13-w60      | 275 | 4,141K | 31.5 |
| P41-raw-kp7-w40   | 253 | 2,531K | 21.5 |
| P41-kp17-w40      | 290 | 2,915K | 21.5 |
| P41-vel-kp13-w40  | 312 | 3,132K | 21.5 |
| P42-kp13-w40-h128 | 590 | 5,844K | 33.0 |
| P42-kp13-w60-h128 | 590 | 8,764K | 48.0 |

### Float → INT8 성능 비교

| 모델 | Float ev_minpr | INT8 ev_minpr | 손실 | INT8 목표(≥0.90) |
|---|---|---|---|---|
| P42-kp17-w60      | 0.9211 | TBD | — | — |
| P41-kp13-w60      | 0.9167 | TBD | — | — |
| P41-raw-kp7-w40   | 0.9167 | TBD | — | — |
| P41-kp17-w40      | 0.9123 | TBD | — | — |
| P41-vel-kp13-w40  | 0.9123 | TBD | — | — |
| P42-kp13-w40-h128 | 0.8991 | TBD | — | — |
| P42-kp13-w60-h128 | 0.9035 | TBD | — | — |

> 평가 진행 중 (tmux p43int8, eval-stride=5). 완료 후 업데이트 예정.
> 이전 실험(Phase 21~35) 기준 평균 손실 -0.005~-0.010MinP.
> P42-kp17-w60(float 0.9211)은 손실 후에도 INT8 목표(0.90) 달성 가능성 높음.

### 전체 Phase INT8 비교표 (기존 + Phase 41/42)

| 실험 | Float MinP | INT8 MinP | 손실 | Flash(KiB) |
|---|---|---|---|---|
| P21-v01 | 0.9227 | 0.9181 | -0.005 | 537 |
| P27-vm0 | 0.9241 | 0.9138 | -0.010 | 537 |
| P40-nv-a65-stateful | 0.9543 | 0.8967† | -0.058† | 590 |
| P42-kp17-w60 (Phase 41/42 최고) | 0.9211 | TBD | — | 290 |
| P41-kp13-w60 | 0.9167 | TBD | — | 275 |
| P41-raw-kp7-w40 | 0.9167 | TBD | — | 253 |

†P40 INT8 손실이 큰 이유: stateful FT 후 float threshold를 그대로 적용. Threshold 재선택 시 개선 예상.
