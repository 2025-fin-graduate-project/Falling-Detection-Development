# 2. 모델 아키텍처 변인 분석

## 2.1 전체 아키텍처 구조

```
Input (N, T, F)
    ↓
Conv1D × 2 [causal, filters=64, kernel=5]   ← 시계열 지역 패턴 추출
    ↓
GRU layer 1 [units, return_sequences=True]  ← 시간적 의존성 모델링
    ↓
GRU layer 2 [units, return_sequences=False]
    ↓
Dense(units, relu)
    ↓
Dense(2, softmax)                           ← fall / non-fall
```

- **Causal padding**: 미래 프레임 참조 없이 인과적 추론 (배포 시 1-step 추론 가능)
- **Unidirectional**: STM32N6 on-device stateful streaming 필수 조건

---

## 2.2 아키텍처 변인: GRU Hidden Size

### 비교 결과 (Phase 37, kp13, w40, pure-window)

| 구성 | 파라미터 | Flash(KiB) | MACC | test window MinP | event_vote MinP | 실험 |
|------|---------|-----------|------|-----------------|----------------|------|
| GRU(64,32) | 70,498 | ~138 | ~1.4M | 0.9327 | 0.8947 | P37-pure-h64-kp13-w40-gru |
| **GRU(128,64)** | **151,042** | **590** | **5.84M** | **0.9404** | **0.8816→0.9543** | P37/P40-nv-a65-stateful |
| GRU(256,128) | ~603K | ~2,200 | 16.7M | 0.9428 | 0.8904 | P37-pure-h256-kp13-w40-gru |

> P40 stateful fine-tuning 후 GRU(128,64) event_vote: 0.8816 → **0.9543**
> GRU(64,32) stateful fine-tuning: 실행 중 (2026-05-20)

**Phase 32 추가 비교 (GRU(256,128) vs (128,64), val_loss, seed=42):**

| 구성 | event MinP | FN | FP |
|------|-----------|----|----|
| GRU(256,128) + val_loss | 0.8966 | 19 | 24 |
| GRU(128,64) + val_loss | 0.9087 | 22 | 13 |

→ 더 큰 모델이 오히려 FP 증가, 과적합 경향

**결론**:
- **GRU(128,64) 최적**: window MinP 및 event MinP 균형, Flash 590KiB (STM32N6 여유)
- GRU(256,128): window MinP는 소폭 높으나 event MinP 낮고 MACC 3배 (추론 속도 문제)
- GRU(64,32): 파라미터 절반 이하, Flash ~138KiB → **경량 배포 후보** (stateful FT 결과 대기)

---

## 2.3 아키텍처 변인: 시퀀스 모델 종류 (GRU vs LSTM vs TCN)

### Phase 30 비교 (val_loss, GRU(128,64)/LSTM(128,64)/TCN, kp7, 40f, seed=42)

| 아키텍처 | Flash(KiB) | event MinP | FN | FP | fall_pr | nfall_pr | fall_rc |
|---------|-----------|-----------|----|----|---------|---------|---------|
| **GRU(128,64)** | 537 | **0.9241** | 18 | 13 | 0.979 | 0.924 | 0.971 |
| LSTM(128,64) | 681 | 0.9181 | 19 | 19 | 0.969 | 0.918 | 0.969 |
| TCN [32,32,64,96] | — | 0.8983 | 24 | 20 | 0.968 | 0.898 | 0.961 |
| TCN-lg [64,64,128,128] | — | 0.8740 | 31 | 17 | 0.972 | 0.874 | 0.950 |

> 기준: P27-vm0(val_video_min_pr) = **0.9241** (GRU(128,64), seed=42)

### GRU vs LSTM 체크포인트 친화성 (Phase 30/31)

| 아키텍처 | val_loss | val_video_min_pr | val_event_min_pr |
|---------|---------|-----------------|-----------------|
| GRU(128,64) | 0.9087 | **0.9241** ← 최적 | 0.9227 |
| LSTM(128,64) | **0.9181** ← 최적 | 0.9038 | 0.8651 |

→ **GRU와 LSTM은 최적 checkpoint 방향이 반대**. 아키텍처-체크포인트 친화성 존재.

**TCN 특성 분석**:
- Causal dilated Conv1D 기반 → window 단위 처리, stateful 불가
- STM32N6 배포 시 Conv transpose 연산 지원 미확인 → **배포 제외**
- 더 큰 TCN(tcn-lg)이 오히려 성능 하락 (FN=31, 최악)

**결론**:
- **GRU(128,64) 채택**: 성능·Flash·MACC·stateful streaming 모두 최적
- LSTM: GRU 대비 Flash 27% 증가, event MinP 소폭 낮음, val_loss 의존성 높음
- TCN: 성능·배포 양쪽에서 열세 → 제외

---

## 2.4 아키텍처 변인: 단방향 vs 양방향 GRU

**Bidirectional GRU**: 미래 프레임까지 참조하여 양방향 처리

| 방식 | 배포 가능 | 성능 |
|------|----------|------|
| Unidirectional | ✅ stateful streaming 가능 | 기준 |
| Bidirectional | ❌ 미래 프레임 필요, stateful 불가 | Phase 4에서 탐색 후 제외 |

**결론**: Bidirectional은 STM32N6 on-device 실시간 추론 불가 → 전 실험 단방향 고정.

---

## 2.5 아키텍처 변인: Velocity Feature (속도 특징)

**Velocity feature**: 주요 관절의 프레임 간 속도(vy, vx) 추가 (kp13 → 74피처)

### 비교 결과 (Phase 37/38)

| 구성 | test window MinP | event_vote MinP | event FP | event FN |
|------|-----------------|----------------|---------|---------|
| kp13 (45feat) | 0.9404 | 0.8816 | 28 | 7 |
| kp13+vel (74feat) | 0.9390 | **0.8991** | **23** | 6 |
| no-vel α=0.65 (45feat) | 0.9345 | 0.9079 | 21 | 12 |

> window MinP: velocity 소폭 하락(-0.0014), event FP: velocity로 -5 개선

**STM32N6 펌웨어 구현 비용 (velocity 추가 시)**:
- 입력 텐서: (1,40,45) → (1,40,74) — 64% 증가
- 정규화 통계: 45개 → 74개 쌍
- `PosePipeline_Update()`: velocity 계산 블록 ~80줄 추가
- GRU Conv1D 가중치: 입력 채널 45→74 → **크기 약 +35%**

**결론**: velocity는 event FP를 5건 줄이지만 구현 복잡도 대비 이득이 제한적.
`no-vel + α=0.65` 조합(P38-nv-a65)이 event MinP 0.9079로 velocity보다 우수하면서 45피처 유지.
**최종 채택: kp13 no-velocity (45 피처)**

---

## 2.6 모델 배포 사양 요약 (STedgeAI analyze 기준)

| 모델 | Flash (KiB) | RAM 활성화 (KiB) | MACC/window | 비고 |
|------|-----------|----------------|------------|------|
| GRU(64,32) kp13 | ~138 | ~12 | ~1.4M | 추정값 |
| **GRU(128,64) kp13** | **590** | **33** | **5.84M** | P40 배포 대상 |
| GRU(256,128) kp13 | ~2,200 | ~120 | 16.7M | MACC 과다 |
| LSTM(128,64) kp7 | 681 | 40 | 6.82M | GRU 대비 비효율 |
| GRU(64,32) v26 (구 배포 PoC) | ~150 | 2.8 | 37,378 | kp7 기반 구형 |

> Flash 레이아웃: 0x70680000 이후 ~60MB 여유 → GRU(128,64) 590KiB 문제없음
> MACC: STM32N6 Cortex-M55 @ 400MHz 기준 초당 ~1.6G op → 5.84M MACC ≈ 3.65ms/window

---

## 2.7 Phase 41/42 — GRU(64,32) vs GRU(128,64) 재비교

Phase 40에서 GRU(128,64) + stateful fine-tuning으로 ev_minpr=0.9543 달성.
Phase 42에서 class-balanced split 기준으로 두 크기를 직접 비교.

| Hidden | Window | ev_minpr | val_minpr (학습중) | Flash (KiB) | MACC |
|---|---|---|---|---|---|
| [64,32]  | 40 | 0.9035 | — | ~150 | ~1.4M |
| [64,32]  | 60 | 0.9167 | — | 275 | 4,141K |
| [128,64] | 40 | 0.8991 | ~0.9604 | 590 | 5,844K |
| [128,64] | 60 | 0.9035 | — | 590 | 8,764K |

**핵심 관찰**: GRU(128,64)이 val 학습 단계에서 높은 성능을 보이나 event-level test에서 GRU(64,32) 대비 열세.
val→event 갭이 GRU(128,64)에서 더 크게 발생 (과적합 또는 window↔event 평가 불일치).

| 항목 | GRU(64,32) | GRU(128,64) |
|---|---|---|
| Flash | 275K (w=60 기준) | 590K |
| MACC | 4.14M (w=60) | 5.84M~8.76M |
| ev_minpr | 0.9167 | 0.8991~0.9035 |
| 권장 | ✅ **채택** | ❌ 탈락 |

**Phase 41/42 결론**: GRU(64,32)로 확정. Phase 40의 GRU(128,64) 우위는 stateful fine-tuning 효과이며, 동일 훈련 조건에서는 GRU(64,32)가 Flash 53% 절감하면서 동급 이상 성능.

## 2.8 Phase 41 — TCN vs GRU 재확인 (class-balanced split)

Phase 30(splits_v2 기준) 이후 Phase 41에서 class-balanced split으로 재확인.

| Architecture | kp7 ev_minpr | kp13 ev_minpr | Post-processing |
|---|---|---|---|
| GRU(64,32) | 0.8904 | 0.9035 | v10k6 |
| TCN(64,32) | 0.8684 | 0.8684 | v7k5 |
| 차이 | GRU **+0.022** | GRU **+0.035** | |

Phase 30 결과(GRU(128,64) vs TCN: +0.025)와 일관됨. 모든 조건에서 GRU 우위. **TCN 최종 탈락 확정**.
