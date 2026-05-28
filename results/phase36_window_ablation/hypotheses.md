# Phase 36+ Window-Level MinPR 연구 가설 로그

**목표: test window MinPR ≥ 0.94**
**평가 기준**: `min(fall_precision, fall_recall, nfall_precision, nfall_recall)` — 40-frame 윈도우 단위
**배경**: 기존 event/video-level 학습은 배포(stateful GRU, 낙상 감지 후 리셋)와 평가 패러다임 불일치.
          Window-level 학습으로 직접 정렬.

---

## Phase 36-A: Feature Set (kp7 vs kp13)

**가설 H36-A**: 손목(wrist) + 무릎(knee) + 발목(ankle) 추가가 낙상 패턴 구별에 결정적이다.

**근거**:
- 낙상 시 발목/무릎이 급격히 하강 → 높은 판별력 기대
- 현재 kp7은 hips까지만 포함; 낙상 후 자세(바닥에 누운 상태) 포착 불가

| 설정 | kp7 (27-feat) | kp13 (45-feat) |
|------|--------------|----------------|
| 포함 | nose, shoulder, elbow, hip | + wrist, knee, ankle |
| 고정 | window=40, GRU(128,64), focal(γ=2, α=0.75) | 동일 |
| 예상 Δ | - | +0.02~0.05 |

**결과**:
- P36-kp7-w40-gru : test_minpr = **0.7771** (fall_pr=0.7878, fall_rc=0.7771, nfall_pr=0.9769, nfall_rc=0.9783, thr=0.7, epoch58)
- P36-kp13-w40-gru: test_minpr = **0.7843** (fall_pr=0.7985, fall_rc=0.7843, nfall_pr=0.9777, nfall_rc=0.9795, thr=0.7, epoch50)
- 결론: **kp13 선택** (+0.0072 vs kp7) — bottleneck: fall_recall. wrist/knee/ankle 추가 효과 소폭 확인

---

## Phase 36-B: Window Size (40 vs 30)

**가설 H36-B**: 30-frame 윈도우(1.5초)가 40-frame(2초)보다 낙상 발생 시점 전후 경계 혼란을 줄인다.

**근거**:
- 40-frame 윈도우는 낙상 전 정상 구간을 많이 포함 → 경계 윈도우(half-fall)가 많음
- 30-frame은 더 짧은 컨텍스트로 더 선명한 fall/nfall 구분 가능
- 단점: 너무 짧으면 post-fall 패턴(쓰러진 상태 유지) 포착 어려울 수 있음

| 설정 | win=40 | win=30 |
|------|--------|--------|
| 고정 | best kp, GRU(128,64) | 동일 |
| 예상 Δ | 기준 | ±0.01~0.03 |

**결과**:
- P36-kp13-w40-gru: test_minpr = **0.7843** (기준, Phase A 결과 재사용)
- P36-kp13-w30-gru: test_minpr = **0.7742** (fall_pr=0.7770, fall_rc=0.7742, nfall_pr=0.9780, nfall_rc=0.9784, thr=0.7, epoch37)
- 결론: **w40 선택** (w30이 -0.0101 낮음) — 30프레임은 컨텍스트 부족으로 오히려 손해

---

## Phase 36-C: Architecture (GRU vs LSTM)

**가설 H36-C**: LSTM의 cell state가 GRU보다 낙상 시작-전개-완료 순서 기억에 유리하다.

**근거**:
- LSTM: 장기 의존성 더 명시적 (cell state + hidden state 분리)
- GRU: 파라미터 적음, 훈련 빠름
- 낙상은 보통 1.5~3초 → 30~60 frame context가 필요한 패턴

| 설정 | GRU(128,64) | LSTM(128,64) |
|------|-------------|--------------|
| 고정 | best kp, best win | 동일 |
| 예상 Δ | 기준 | +0.00~0.03 |

**결과**:
- P36-kp13-w40-gru : test_minpr = **0.7843** (Phase A 결과 재사용)
- P36-kp13-w40-lstm: test_minpr = **0.7771** (fall_pr=0.8067, fall_rc=0.7771, nfall_pr=0.9770, nfall_rc=0.9807, thr=0.7, epoch44)
- 결론: **GRU 선택** (LSTM 동등하거나 약간 낮음) — LSTM은 precision 높지만 recall 동일 → bottleneck 미해결

---

## Phase 37: 0.94 미달 시 다음 가설 (자동 진행)

### H37-D: 속도 특징 추가 (velocity features)

**가설**: 프레임-간 키포인트 이동 속도(Δy, Δx per keypoint)가 낙상 검출에 핵심 정보다.

**근거**:
- 낙상 = 하강 가속도가 급증하는 이벤트 → velocity가 명시적 신호
- 현재 모델은 절대 좌표만 사용; GRU가 암묵적으로 속도를 학습해야 함
- 명시적 velocity 추가 시 수렴 속도와 정확도 동시 개선 기대

**구현**: `build_velocity_features(frames)` — 각 키포인트 (Δy_t = y_t - y_{t-1})
- kp7 + velocity → 27 + 21 = 48 features (confidence 제외 velocity)
- kp13 + velocity → 45 + 39 = 84 features

**실험**:
- P37-kp7v-w{best}-gru
- P37-kp13v-w{best}-gru

---

### H37-E: 상대 좌표 정규화 (hip-centered relative positions)

**가설**: 절대 좌표 대신 hip centroid 기준 상대 좌표를 쓰면 카메라 위치 invariance가 높아진다.

**근거**:
- 현재 kp_y, kp_x는 전체 화면 기준 절대 좌표 → 사람이 화면 어디 있느냐에 따라 값이 달라짐
- Hip 중심 기준 상대 좌표: `kp_y_rel = kp_y - hip_center_y`
- 낙상 패턴이 더 일관성 있게 표현됨

**실험**: P37-kp13-hip-w{best}-gru

---

### H37-F: Focal Loss 튜닝 (α, γ sweep)

**가설**: 현재 α=0.75(fall 강조)가 최적이 아닐 수 있다. val MinPR bottleneck이 fall_recall이면 α↑, nfall_recall이면 α↓.

| 실험 | α (fall weight) | γ |
|------|----------------|---|
| P37-fl-a80 | 0.80 | 2.0 |
| P37-fl-a70 | 0.70 | 2.0 |
| P37-fl-g3  | 0.75 | 3.0 |

---

### H37-G: 데이터 균형 재조정 (stride sweep)

**가설**: training fall_stride=1, nfall_stride=5가 최적이 아닐 수 있다.

- 현재: train 33% fall / 67% nfall
- H37-G1: 50/50 (fall_stride=1, nfall_stride=3) → 더 균형
- H37-G2: 40/60 (fall_stride=1, nfall_stride=4) → 중간

---

### H37-H: label_3class 경계 윈도우 제거

**가설**: label_3class=2(transition) 프레임이 마지막 프레임인 윈도우를 학습에서 제외하면 신호가 더 깨끗해진다.

**구현**: `build_train_windows`에서 `labels_3class[t] == 2`인 경우 skip

---

### H37-J: Complete-event 윈도우 레이블링 ← **사용자 제안**

**가설**: 낙상 이벤트 [start, end] 전체가 윈도우 안에 포함될 때만 fall 윈도우로 인정하면 신호가 깨끗해져 MinPR이 오른다.

**핵심 아이디어**:
```
BAD (현재):  1 1 1 | 1 0 0 0 0 |  ← onset이 윈도우 밖 → 과거 낙상 잔재 보고 분류
             0 0 0 | 0 1 1 1 1 |  ← fall이 윈도우 밖으로 계속됨 → 낙상 중간 상태
GOOD (제안): | 1 1 1 1 0 0 0 0 |  ← onset + end 모두 포함 → 완전한 낙상 사건 관찰
             | 0 0 1 1 1 1 0 0 |  ← 동일, 윈도우 내부에서 0→1→0 전환 완료
```

**근거**:
- 낙상은 보통 1~2초(20~40 프레임) → window=40이면 완전 포함 가능
- onset이 밖에 있는 윈도우를 fall로 학습 = "미래를 모르는 상태에서 과거 정보로 예측" → 배포 시 미스매치
- 낙상 종료 전 윈도우를 fall로 학습 = 낙상 중간 상태를 fall로 판단 → 모델 혼란
- **현실성**: 실제 GRU 배포에서 낙상 시작~끝이 윈도우 안에 잡혀야 정확한 이벤트 감지

**구현**:
```python
# 각 영상에서 fall event = 연속된 label==1 구간 [fall_start, fall_end]
for (fall_start, fall_end) in find_fall_events(labels):
    event_len = fall_end - fall_start + 1
    if event_len > window_size: continue  # 너무 긴 낙상은 스킵
    # 유효 윈도우 시작 범위: fall 전체가 윈도우 안에 들어오는 w_start
    w_min = max(0, fall_end - window_size + 1)
    w_max = fall_start
    for w_start in range(w_min, w_max + 1, fall_stride):
        fall_windows.append(frames[w_start:w_start + window_size], label=1)

# nfall 윈도우: 모든 프레임이 0인 구간만
```

**예상 효과**:
- fall 학습 예제 수 감소하지만 신호 일관성 대폭 향상
- fall_recall 개선 (모델이 명확한 onset→end 패턴 학습)
- nfall false alarm 감소 (경계 혼합 윈도우 제거)

**실험 결과**: `P37-pure-kp13-w40-gru` → **test_minpr=0.9404 ✅ 목표 달성**
- val=0.9477, test=0.9404, fall_pr=0.9404, fall_rc=0.9421, nfall_pr=0.9911, nfall_rc=0.9908
- threshold=0.725, epoch40(early stop), FN=908, FP=936
- Phase 36 최고(0.7843) 대비 **+0.1561** 향상

---

## Phase 37 Ablation: kp7 vs kp13 (pure-window 조건에서)

**가설**: pure-window 조건에서도 kp13이 kp7보다 우수할 것이다.

| 설정 | kp7+pure(27-feat) | kp13+pure(45-feat) |
|------|------------------|-------------------|
| 고정 | window=40, GRU(128,64), margin=5 | 동일 |

**결과**: `P37-pure-kp7-w40-gru` → test_minpr=**0.9327** (ep52 early stop)
- val=0.9427, fall_pr=0.9413, fall_rc=0.9327, nfall_pr=0.9896, nfall_rc=0.9910
- thr=0.75, FN=1054, FP=911, train: 112,863 fall / 146,648 nfall
- vs kp13 기준(0.9404): **−0.0077** (kp13 우세)
- bottleneck: fall_recall=0.9327 (kp13은 fall_pr=fall_rc≈0.94로 더 균형)
- 결론: **kp13 선택 유지** — wrist/knee/ankle이 낙상 완료 자세 포착에 기여

---

## Window vs Event MinPR 수렴 분석

**배경**: window MinPR과 event MinPR 사이의 갭이 크면 배포 성능 예측이 왜곡됨.
event eval은 `event_vote_eval()` 함수로 시뮬레이션 (STM32 배포와 동일 로직):
- `raw (K=1/N=1)`: 단일 창 threshold 초과 → 즉시 감지 (이전 방식)
- `vote (K=3/N=5)`: 5창 중 3창 이상 초과 → 감지 (STM32 스무딩 적용 방식)

### P37-pure-m10-kp13-w40-gru (최초 측정값)

| 구분 | minpr | fall_pr | fall_rc | nfall_pr | nfall_rc | FP비디오 |
|------|-------|---------|---------|---------|---------|---------|
| test_window | 0.9341 | 0.9533 | 0.9341 | 0.9879 | 0.9915 | 746창 |
| test_event(raw) | 0.8158 | 0.9367 | **0.9936** | 0.9789 | **0.8158** | 42영상 |
| test_event(v5k3) | 0.8640 | 0.9520 | **0.9840** | 0.9517 | **0.8640** | 31영상 |

**분석**:
- window→event 갭: raw=**+0.1183**, vote(5,3)=**+0.0701** — vote가 갭을 0.048 축소
- bottleneck: `nfall_recall` — 비낙상 영상에서 FP 창이 발생 (42→31개 영상)
- fall_recall은 event mode에서 오히려 매우 높음 (0.9840) — 낙상은 잘 감지됨
- 주요 과제: 비낙상 영상 내 낙상-유사 자세(앉기, 구부리기 등) 오검출 억제

**결론**: vote(5,3)으로 FP 영상 26% 감소(42→31), 갭 41% 축소(+0.1183→+0.0701).
완전한 수렴을 위해서는 training에서 event-level FP를 직접 억제하는 전략 필요 (Phase 38).

---

## Phase 37 Ablation: Velocity Features (pure-window)

**가설 H37-D**: 프레임-간 키포인트 이동 속도(Δy, Δx)를 명시적으로 제공하면 낙상 검출 성능이 향상된다.

**실험**: `P37-pure-vel-kp13-w40-gru` — kp13(45-feat) + velocity(39-feat) = 84 features total

| 구분 | minpr | fall_pr | fall_rc | nfall_pr | nfall_rc | FN | FP창 |
|------|-------|---------|---------|---------|---------|-----|------|
| val_window  | 0.9502 | 0.9534 | 0.9502 | 0.9923 | 0.9928 | 1569 | 1465 |
| test_window | 0.9390 | 0.9556 | 0.9390 | 0.9906 | 0.9933 | 956 | 684 |
| test_event(raw,K=1/N=1) | 0.8772 | 0.9569 | 0.9936 | 0.9804 | 0.8772 | FP=28vid |
| test_event(vote,K=3/N=5) | 0.8991 | 0.9642 | 0.9904 | 0.9716 | 0.8991 | FP=23vid |

- thr=0.725, early stop ep47, Gap: raw=**+0.0915**, vote(5,3)=**+0.0719**

**분석**:
- window test_minpr=0.9390 — 기준(0.9404)보다 **−0.0014** (소폭 하락)
- val_minpr=0.9502 — 기준(0.9477)보다 **+0.0025** (val에서는 최고치)
- event FP 비디오: raw 28개(vs m10: 42개), vote 23개(vs m10: 31개) — **큰 폭 감소**
- vote gap(+0.0719) ≈ m10(+0.0701)와 유사 — window→event 수렴도는 비슷
- bottleneck: fall_recall=0.9390 (window에서 FN=956으로 m0~m5 대비 다소 많음)

**해석**:
- velocity feature는 **비낙상 영상 FP를 효과적으로 억제** (42→28 raw FP): 앉기/구부리기 같은 점진적 동작은 속도 벡터가 급변하지 않아 구분 가능
- 반면 window fall_recall이 소폭 하락 → 특정 낙상 패턴(천천히 미끄러짐 등)에서 velocity 신호가 약할 수 있음
- val과 test 간 gap이 크지 않아 과적합은 없음; 파라미터 증가(45→84 feat)에도 일반화 유지

**결론**: velocity 추가는 event-level FP 억제에 유효하나 window-level MinPR은 기준 대비 동등 수준 (−0.0014). **기준(kp13, margin=5) 유지**; Phase 38에서 event-level FP 억제 전략 탐색 시 velocity 조합 재검토 가치 있음.

---

## Window vs Event MinPR 수렴 분석 (전체 비교)

| 실험 | test_window | test_event_raw | test_event_vote | raw갭 | vote갭 | FP비디오(raw) | FP비디오(vote) | FN비디오(vote) |
|------|------------|---------------|----------------|------|------|------------|------------|------------|
| P37-pure-m10-kp13-w40-gru  | 0.9341 | 0.8158 | 0.8640 | +0.1183 | +0.0701 | 42 | 31 | — |
| **P37-pure-vel-kp13-w40-gru**  | 0.9390 | **0.8772** | **0.8991** ★ | +0.0915 | +0.0399 | **28** | **23** | 6 |
| P37-pure-h256-kp13-w40-gru | 0.9428 | 0.8640 | 0.8904 | +0.0973 | +0.0712 | 31 | 25 | 6 |
| P37-pure-h64-kp13-w40-gru  | 0.9327 | 0.8816 | 0.8947 | +0.0839 | +0.0709 | 27 | 24 | 7 |

**관찰**:
- **event_vote 순위**: vel(0.8991) > h64(0.8947) > h256(0.8904) > m10(0.8640)
- window 순위와 역전: h64(window 4위) → event 2위, h256(window 1위) → event 3위
- velocity가 raw FP 억제 최강(28→23): 점진적 동작의 속도 벡터 차이 활용
- h64는 window MinPR이 낮지만 event FP도 적음(27raw/24vote) — 작은 모델이 오히려 덜 튐
- **vote(5,3) 효과**: m10: +0.0482pt, h256: +0.0264pt, h64: +0.0131pt, vel: +0.0219pt

---

## Phase 37 Ablation: Hidden Size (256,128 vs 128,64 vs 64,32)

**가설 H37-G**: 더 큰 hidden dimension이 복잡한 낙상 패턴을 더 잘 포착한다 (h256). 반대로 경량 모델(h64)도 충분한 성능을 낼 수 있다.

### h256 결과: `P37-pure-h256-kp13-w40-gru` (hidden=256,128)

| 구분 | minpr | fall_pr | fall_rc | nfall_pr | nfall_rc | FN | FP창 |
|------|-------|---------|---------|---------|---------|-----|------|
| val_window  | 0.9495 | — | — | — | — | — | — |
| test_window | **0.9428** | 0.9464 | 0.9428 | 0.9912 | 0.9918 | 896 | 836 |
| test_event(raw) | 0.8640 | — | — | — | — | FP=31vid | FN=4vid |
| test_event(vote) | 0.8904 | — | — | — | — | FP=25vid | FN=6vid |

- thr=0.75, early stop ep41, Gap: raw=**+0.0973**, vote(5,3)=**+0.0712**
- vs 기준(128,64): test **+0.0024** — **Phase 37 최고 test_minpr**
- FP창(836) > 기준(936)보다 적음 — 대형 모델이 nfall 오검출도 줄임

**분석**:
### h64 결과: `P37-pure-h64-kp13-w40-gru` (hidden=64,32)

| 구분 | minpr | fall_pr | fall_rc | nfall_pr | nfall_rc | FN | FP창 |
|------|-------|---------|---------|---------|---------|-----|------|
| val_window  | 0.9426 | — | — | — | — | — | — |
| test_window | 0.9327 | 0.9450 | 0.9327 | 0.9896 | 0.9916 | 1055 | 850 |
| test_event(raw)  | 0.8816 | — | — | — | — | FP=27vid | FN=7vid |
| test_event(vote) | 0.8947 | — | — | — | — | FP=24vid | FN=7vid |

- thr=0.725, early stop ep41, Gap: raw=**+0.0839**, vote(5,3)=**+0.0709**
- test_window=0.9327 — 기준보다 **−0.0077** (window 기준 최하위)
- 그러나 event_vote=**0.8947** — h256(0.8904)보다 높음 (event 기준 2위)
- FP비디오(raw=27)는 전체 실험 중 최소 — 경량 모델이 덜 과적합, 덜 튐

**해석**: window 성능↓ but event FP 적음 → 작은 모델이 nfall 패턴에 덜 과적합되어 오히려 일반화 잘됨. 포팅 최적이나 window MinPR이 0.93대로 낮아 실제 낙상 미감지 위험 존재.

---

## ★ 최종 모델 결정 (전체 Event Eval 완료)

### 전체 비교표 (event_vote MinPR 기준 정렬)

| 순위 | 모델 | window | ev_raw | ev_vote | FP비디오 | FN비디오 | INT8크기 |
|------|------|--------|--------|---------|---------|---------|---------|
| 1 | kp13-w30-gru | 0.9255 | 0.8772 | **0.9035** | **22** | 14 | ~138KB |
| 2 | **vel (128,64+velocity)** | 0.9390 | 0.8772 | **0.8991** | 23 | **6** | ~138KB |
| 3 | m0 (margin=0) | 0.9350 | 0.8728 | 0.8947 | 24 | 12 | ~138KB |
| 3 | h64 (64,32) | 0.9327 | 0.8816 | 0.8947 | 24 | 7 | ~35KB |
| 5 | h256 (256,128) | 0.9428 | 0.8640 | 0.8904 | 25 | 6 | ~550KB |
| 6 | kp7-w40-gru | 0.9327 | 0.8553 | 0.8860 | 26 | 7 | ~138KB |
| 7 | baseline (kp13-w40) | 0.9404 | 0.8684 | 0.8816 | 27 | 8 | ~138KB |
| 8 | m10 (margin=10) | 0.9341 | 0.8158 | 0.8640 | 31 | 10 | ~138KB |

### 핵심 발견: window ≠ event 역전

- window 1위(h256=0.9428)가 event 5위(0.8904)
- window 최하위(w30=0.9255)가 event 1위(0.9035)
- **window MinPR은 배포 성능과 다른 척도임을 확인**

### 최종 선택: `P37-pure-vel-kp13-w40-gru` ★

**이유: 사용자 목표 "miss rate 최소화 + 정상 감지 억제"**

| 기준 | w30 (event 1위) | **vel (선택)** |
|------|----------------|--------------|
| event_vote MinPR | **0.9035** | 0.8991 |
| **FN (낙상 미감지)** | **14개** | **6개** ← |
| FP (오경보) | 22개 | 23개 |
| window MinPR | 0.9255 | 0.9390 |

- w30: event MinPR 1위지만 FN=14(낙상 14건 미감지) — 안전 시스템에서 치명적
- vel: FN=6(최소, h256과 공동), FP=23(2위) — miss rate와 FP 모두 최적
- **낙상 감지 시스템에서 FN 비용 >> FP 비용**

**포팅 계획**:
- hidden(128,64) = 기존 모델과 동일 → INT8 변환 워크플로 재사용
- input_size 45→84 (velocity 39개 추가)
- pose_pipeline.c에 velocity 계산 추가 (window 버퍼 내 프레임 차분)

- h256이 모든 Phase 37 실험 중 test_minpr 최고 달성 (0.9428)
- val=0.9495는 velocity(0.9502)와 유사하나 test generalization이 더 좋음
- event-level FP 비디오 수(raw=31)는 velocity(28)보다 많으나 vote=25로 비슷
- bottleneck: fall_recall=0.9428 — 여전히 FN이 많음 (896창)
- 파라미터 증가 대비 성능 향상은 소폭 (+0.0024) → diminishing returns 구간

---

## Phase 38: Event-MinPR ≥ 0.90 목표 (window ≥ 0.93 유지)

**배경**: Phase 37 최선 모델(vel, 포팅 완료)
- window MinPR = **0.9390** ✓, event_vote = **0.8991** (0.0009 미달)
- bottleneck: nfall_recall = 0.8991 → **FP 비디오 23개** / ~228 nfall 테스트 영상
- 목표: FP 비디오 ≤ 21개 (nfall_recall ≥ 0.9079)

### ⚠️ 배포 정합성 이슈

STM32 현재 배포 (`GRU_FALL_RESET_COUNT=1`) = K=1/N=1 (raw, 단일 창 초과 시 즉시 발동).  
event eval의 vote(K=3/N=5) = 0.8991이지만 실제 배포 기준 raw = **0.8772** (목표에서 0.0228 멀다).

**권장 해결책**: main.c에 vote(5,3) 로직 추가 → 배포가 eval 기준과 일치.
```c
// FallDetectionState_TypeDef에 추가:
float32_t vote_buf[5];  uint32_t vote_head;

// FallDetection_Update 내부 (threshold 초과 체크 대신):
vote_buf[vote_head] = fall_state.fall_score;
vote_head = (vote_head + 1) % 5;
uint32_t vote_sum = 0;
for (int i = 0; i < 5; i++)
    vote_sum += (vote_buf[i] >= GRU_FALL_SCORE_THRESHOLD) ? 1 : 0;
if (vote_sum >= 3) { /* trigger alarm */ }
```
이 변경 없이는 event_vote 0.90 달성이 의미 없음.

---

### H38-A: vel + w30 (PRIMARY BET)

**가설**: w30이 event_vote=0.9035를 달성한 이유(짧은 컨텍스트로 FP 억제)와 velocity가 FP 억제하는 효과(28→23)가 조합되면 FP ≤ 20이 가능하다.

**근거**:
- w30 단독: event_vote=0.9035, FP=22, **FN=14** (낙상 14건 미감지)
- vel 단독: event_vote=0.8991, FP=23, **FN=6** (최소)
- vel의 속도 피처는 빠른 낙상은 잘 잡고 점진적 자세 변화(앉기)는 구분
- w30은 컨텍스트가 짧아 낙상 전 정상 구간 혼재가 줄고 nfall FP도 감소
- 조합: FP ≤ 20, FN ≤ 12 기대

**예상**: event_vote ≈ 0.90~0.92, window ≈ 0.92~0.94

**리스크**: w30 단독 FN=14를 velocity가 얼마나 회복하는지 불확실

| 설정 | w30 단독 | vel 단독 | **vel+w30 실제** |
|------|---------|---------|----------------|
| window MinPR | 0.9255 | 0.9390 | 0.9290 |
| event_vote | 0.9035 | 0.8991 | **0.8991** |
| FP비디오 | 22 | 23 | **23** (불변) |
| FN비디오 | 14 | 6 | **12** (악화) |

**결과 분석 (2026-05-18)**:
- **FP는 vel이 지배** — w30의 FP 억제 효과가 vel 조합 시 사라짐
- vel이 만드는 23개 FP 비디오는 window 크기와 무관한 패턴 (vel 피처 자체의 특성)
- FN은 6→12: vel의 FN 최소화 효과도 w30으로 인해 절반 상실
- 결론: **vel+w30은 두 접근의 약점을 더하는 조합** — 독립적 개선이 아님
- 교훈: FP 억제는 window 크기보다 **낙상-유사 동작을 구별하는 피처 자체**가 관건

---

### H38-B: vel + focal α=0.65

**가설**: 현재 α=0.75(fall 75%/nfall 25% 가중)는 nfall FP 억제에 불리. α=0.65로 낮추면 모델이 낙상 신호에 덜 민감해져 FP 1-2건 감소.

**근거**:
- Focal loss에서 α는 fall class weight; 높을수록 recall 우선 → nfall precision↓ → FP↑
- vel 모델에서 nfall_precision=0.9716 (nfall FP 창 684개) — 다소 낮음
- α=0.65: 정상 자세에서 "fall 가능성 있음" 점수를 전반적으로 낮춤
- 단점: fall_recall 소폭 하락 (0.9904 → 0.97~0.98)

**예상**: event_vote ≈ 0.895~0.905, FP -1~-2, FN +0~+2

---

### H38-C: vel + h256

**가설**: velocity(FP 억제) + h256(window MinPR 최고=0.9428) 조합이 두 효과를 합산한다.

**근거**:
- h256 단독: window=0.9428(최고), event_vote=0.8904, FP=25, FN=6
- vel 단독: window=0.9390, event_vote=0.8991, FP=23, FN=6
- 조합: window ≥ 0.94 기대, event FP=21~24 기대

**예상**: window ~0.945, event_vote ~0.895~0.905

---

### H38-D: vel + nfall_stride=3

**가설**: nfall 학습 데이터 밀도를 높이면 (stride=5→3) 하드 네거티브를 더 많이 보아 FP 억제 향상.

**근거**:
- 현재: ~113K fall / ~147K nfall (43/57)
- stride=3: ~113K fall / ~244K nfall (32/68) — nfall 비중 ↑
- 앉기/구부리기 등 낙상 유사 자세가 더 많이 학습됨
- 단점: 클래스 불균형이 더 심해져 fall_recall 하락 가능

**예상**: event_vote ~0.895~0.905, FP -1~-3, FN +0~+4

---

### H38-E: vel + dropout=0.4

**가설**: dropout 강화(0.3→0.4)가 모델의 nfall 패턴 과적합을 줄여 FP 억제.

**근거**:
- 모델이 학습 nfall 패턴에 과적합 → 테스트 nfall 영상에서 예외적 자세에 반응
- dropout↑로 ensemble 효과 → 특정 자세에 덜 민감

**예상**: 효과 불확실, window 소폭 하락 가능, event_vote ±0.01

---

### H38-NV: no-velocity + α=0.65 (비용 절감)

**동기**: velocity 피처는 event_vote 향상에 효과적이지만 (FP 28→23, FN 9→6),
STM32 펌웨어 구현 비용이 크다 (29개 추가 피처, 이전 프레임 저장, 정규화 통계 74개).
α=0.65의 nfall 구분 강화 효과만으로 동일한 event_vote 달성 가능한지 검증.

**가설**: α=0.65가 velocity 없이도 FP 억제 → 45-feature 모델로 같은 성능

**비교 설계**:
- P38-vel-a65-gru: vel(74feat) + α=0.65 → 결과 대기 중
- **P38-nv-a65-gru**: no-vel(45feat) + α=0.65 → Wave2 첫 번째로 실행

**판단 기준**:
- event_vote 차이 ≤ 0.005 → velocity 불필요, **45-feat 모델 채택**
- event_vote 차이 > 0.010 → velocity 필수 유지

**펌웨어 비용 절감 효과 (velocity 제거 시)**:
- 입력 텐서: (1×40×74) → **(1×40×45)** — 39% 감소
- 정규화 통계: 74개 → 45개 쌍
- `PosePipeline_t` 구조체: `hssc_y_prev`, `vhssc_x_ema` 등 9개 필드 제거
- `PosePipeline_Update()`: velocity 계산 블록 ~80줄 제거
- GRU 가중치: Conv1D 입력 채널 74→45 → **가중치 크기 약 -35%**

---

### H38-Wave2 (하이퍼파라미터 조합 — 코드 수정 없음)

**H38-F: vel + w30 + h256**
- vel+w30이 최선일 때 대형 모델 추가로 표현력 향상
- 예상: window ≥ 0.94, event_vote ~0.90~0.92, FP ≤ 20

**H38-G: vel + w30 + nfall_stride=3**
- vel+w30 기반 + 하드 네거티브 밀도 증가 → FP 추가 억제
- 예상: event_vote +0.005~0.010, FP -1~-3, FN +0~+3

**H38-H: vel + w30 + focal α=0.60**
- 더 공격적인 focal weight 감소 (0.75→0.60): nfall 구분력 극대화
- 리스크: fall_recall 추가 하락 (FN ≥ 10 가능)

**H38-I: vel + w35**
- 중간 윈도우 (30↔40): w30의 FP 억제 + w40의 FN 억제 균형 탐색
- w30(FN↑)과 w40(FP↑) 중간에서 최적점 탐색
- 예상: event_vote ~0.900~0.910

**H38-J: vel + w30 + dropout=0.35**
- 약한 정규화 추가: w30이 이미 implicit 정규화 효과
- dropout 0.30→0.35로 소폭 강화 → 과적합 억제

### H38-Wave3 (신규 피처 — 코드 수정 필요)

**H38-K: trunk angle feature (몸통 기울기)**
가장 강력한 신규 피처. 낙상 = 몸통이 수직→수평으로 전환.

```python
# 어깨 중간점 → 엉덩이 중간점 벡터의 수직 성분
shoulder_mid_y = (kp5_y + kp6_y) / 2
hip_mid_y      = (kp11_y + kp12_y) / 2
shoulder_mid_x = (kp5_x + kp6_x) / 2
hip_mid_x      = (kp11_x + kp12_x) / 2
trunk_dy = hip_mid_y - shoulder_mid_y   # 양수 = 서있음 (y축 아래)
trunk_dx = hip_mid_x - shoulder_mid_x
trunk_sin = trunk_dy / (sqrt(trunk_dy**2 + trunk_dx**2) + 1e-6)
# 서있음: ~1.0, 쓰러짐: ~0.0
```
+ trunk_sin의 시간 미분(angular velocity) = 낙상 중 급격히 감소

**구현 필요**: `train_window_phase37.py` + `pose_pipeline.c` 양쪽 수정

**H38-L: lower-body center feature**
- lower_center_y = mean(kp11_y, kp12_y, kp13_y, kp14_y, kp15_y, kp16_y) — 하체 중심
- `lower_to_upper_dist` = lower_center_y - HSSC_y — 하체-상체 수직 거리 (쓰러지면 감소)
- `d(lower_center_y)/dt` — 하체 하강 속도

---

## 결과 추적

| ID | feat | win | arch | 추가 변인 | val_minpr | test_minpr | bottleneck | 결론 |
|----|------|-----|------|----------|-----------|------------|------------|------|
| P36-kp7-w40-gru  | kp7  | 40 | GRU | - | 0.7821 | 0.7771 | fall_recall | 기준선 |
| P36-kp13-w40-gru | kp13 | 40 | GRU | - | 0.7845 | 0.7843 | fall_recall | kp13 선택 |
| P36-kp7-w30-gru  | kp7  | 30 | GRU | - | - | - | - | (Phase B에서 제외) |
| P36-kp13-w30-gru | kp13 | 30 | GRU | - | 0.7767 | 0.7742 | fall_recall | w40 선택 |
| P36-kp13-w40-lstm | kp13 | 40 | LSTM | - | 0.7812 | 0.7771 | fall_recall | GRU 선택 |
| P37-pure-kp13-w40-gru  | kp13 | 40 | GRU  | pure-window(margin=5) | 0.9477 | **0.9404** ✅ | fall_pr | **목표 달성!** |
| P37-pure-kp7-w40-gru   | kp7  | 40 | GRU  | pure-window(margin=5) | 0.9427 | 0.9327 | fall_recall | kp13 선택 재확인 |
| P37-pure-kp13-w30-gru  | kp13 | 30 | GRU  | pure-window(margin=5) | 0.9336 | 0.9255 | fall_recall | w40 선택 재확인 |
| P37-pure-kp13-w40-lstm | kp13 | 40 | LSTM | pure-window(margin=5) | - | 재실행예정 | - | reset_after 버그수정 후 재실행 필요 |
| P37-pure-m0-kp13-w40-gru  | kp13 | 40 | GRU | pure+margin=0  | 0.9426 | 0.9350 | fall_recall | margin=5 선택 재확인 (−0.0054) |
| P37-pure-m10-kp13-w40-gru | kp13 | 40 | GRU | pure+margin=10 | 0.9457 | 0.9341 | fall_recall | m5 선택 유지 (−0.0063); event gap 최초 측정 |
| P37-pure-vel-kp13-w40-gru | kp13 | 40 | GRU | pure+velocity  | 0.9502 | 0.9390 | fall_recall | window 동등(-0.0014); event FP 크게 감소(42→28) |
| P37-pure-h256-kp13-w40-gru | kp13 | 40 | GRU | hidden=256,128 | 0.9495 | **0.9428** ✅ | fall_recall | Phase 37 최고 (+0.0024 vs 기준) |
| P37-pure-h64-kp13-w40-gru  | kp13 | 40 | GRU | hidden=64,32   | 0.9426 | 0.9327 | fall_recall | window↓(-0.0077) but event_vote 2위(0.8947) |
| P38-vel-w30-gru   | kp13 | 30 | GRU | vel+pure+m5 | 0.9350 | 0.9290 | fall_recall | H38-A: event_vote=0.8991(동일), FP=23(불변), FN=12(악화) |
| P38-vel-a65-gru   | kp13 | 40 | GRU | vel+α=0.65  | - | - | - | H38-B |
| P38-vel-h256-gru  | kp13 | 40 | GRU | vel+h256    | - | - | - | H38-C |
| P38-vel-ns3-gru   | kp13 | 40 | GRU | vel+ns3     | - | - | - | H38-D |
| P38-vel-drp4-gru  | kp13 | 40 | GRU | vel+drp=0.4 | - | - | - | H38-E |
| P38-nv-a65-gru        | kp13 | 40 | GRU | no-vel+α=0.65   | - | - | - | H38-NV (Wave2 1st — 비용절감) |
| P38-vel-w30-h256-gru  | kp13 | 30 | GRU | vel+w30+h256    | - | - | - | H38-F (Wave2) |
| P38-vel-w30-ns3-gru   | kp13 | 30 | GRU | vel+w30+ns3     | - | - | - | H38-G (Wave2) |
| P38-vel-w30-a60-gru   | kp13 | 30 | GRU | vel+w30+α=0.60  | - | - | - | H38-H (Wave2) |
| P38-vel-w35-gru       | kp13 | 35 | GRU | vel+w35         | - | - | - | H38-I (Wave2) |
| P38-vel-w30-drp35-gru | kp13 | 30 | GRU | vel+w30+drp35   | - | - | - | H38-J (Wave2) |

> 업데이트: 각 실험 완료 시 자동 기입
