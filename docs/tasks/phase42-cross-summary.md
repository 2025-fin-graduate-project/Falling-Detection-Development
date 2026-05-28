# Phase 42 Summary — Cross-Variable Validation

**기간**: 2026-05  
**상태**: 완료  
**스크립트**: `scripts/train_window_phase37.py`  
**결과 디렉토리**: `results/phase42_cross/`

---

## 목적

Phase 41 ablation 결과를 교차 검증하고 누락된 조합을 실험:
1. kp17 × w60 조합 효과 (각각 P41에서 독립 실험만 수행됨)
2. raw preprocessing이 다른 feature set에서도 유효한지
3. GRU(128,64)가 GRU(64,32) 대비 실제로 이득을 주는지

---

## 실험 결과

### 전체 8개 실험

| 모델 | kp | w | prep | vel | hidden | ev_minpr | Flash |
|---|---|---|---|---|---|---|---|
| P42-kp13-w40      | kp13 | 40 | filt | ✗ | [64,32]  | 0.9035 | — |
| P42-raw-kp7-w60   | kp7  | 60 | raw  | ✗ | [64,32]  | 0.9123 | — |
| P42-raw-kp17-w40  | kp17 | 40 | raw  | ✗ | [64,32]  | 0.9123 | — |
| P42-raw-kp13-vel  | kp13 | 40 | raw  | ✓ | [64,32]  | 0.8728 | — |
| **P42-kp17-w60**  | kp17 | 60 | filt | ✗ | [64,32]  | **0.9211** | 290K |
| P42-vel-kp13-w60  | kp13 | 60 | filt | ✓ | [64,32]  | 0.8684 | — |
| P42-kp13-w40-h128 | kp13 | 40 | filt | ✗ | [128,64] | 0.8991 | 590K |
| P42-kp13-w60-h128 | kp13 | 60 | filt | ✗ | [128,64] | 0.9035 | 590K |

### 주요 발견

**kp17 × w60 교차 효과 (P42-kp17-w60)**  
- ev_minpr=0.9211 → P41+P42 전체 최고
- P41에서 kp17(w=40)=0.9123, kp13(w=60)=0.9167 → 교차 시 상승 확인
- Flash=290K, MACC=4.37M, Activation=31.5K → STM32N6 배포 적합

**Raw preprocessing 재검증**  
- kp7-raw-w60(0.9123) vs kp7-raw-w40(0.9167): raw에서도 w=40이 w=60보다 높음
- kp17-raw-w40(0.9123) = kp17-filtered-w40(0.9123): kp17에서 전처리 차이 없음
- raw+velocity: 0.8728로 크게 하락 → raw에서 velocity는 더욱 해로움

**GRU(128,64) 효과**  
- h128 모델이 val 단계에서 높은 성능(최대 0.9604) 보이나
- event-level test에서 GRU(64,32) 대비 오히려 열세
- Flash 2.1× 증가(275→590K)에 이점 없음 → GRU(64,32) 최종 채택

---

## 배포 분석 대상 (STedgeAI analyze, stm32n6)

| 모델 | Flash (KiB) | MACC | Activation (KiB) |
|---|---|---|---|
| P42-kp17-w60      | 290 | 4,371K | 31.5 |
| P42-kp13-w40-h128 | 590 | 5,844K | 33.0 |
| P42-kp13-w60-h128 | 590 | 8,764K | 48.0 |

---

## P41 + P42 통합 Top 5 (ev_minpr 기준)

| 순위 | 모델 | ev_minpr | Flash | 특이사항 |
|---|---|---|---|---|
| 1 | P42-kp17-w60    | 0.9211 | 290K | 전체 최고 |
| 2 | P41-kp13-w60    | 0.9167 | 275K | 가장 가벼운 상위권 |
| 2 | P41-raw-kp7-w40 | 0.9167 | 253K | 최소 Flash |
| 4 | P41-kp17-w40    | 0.9123 | 290K | — |
| 4 | (6개 모델 동률)  | 0.9123 | 다양 | — |

---

## 결론

- **최종 채택 모델**: P42-kp17-w60 (GRU(64,32), kp17, w=60, filtered)
- **경량 대안**: P41-raw-kp7-w40 (GRU(64,32), kp7, w=40, raw, Flash=253K)
- **GRU(128,64) 탈락**: 성능 이득 없고 Flash 2.1× 증가
- **Velocity 탈락**: 모든 조합에서 이득 미미하거나 해로움

Phase 43에서 INT8 양자화 성능 측정 및 논문용 시각화 생성.
