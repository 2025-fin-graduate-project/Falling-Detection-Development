# Phase 32 Summary: GRU × val_loss 심화 탐색

작성일: 2026-05-16

## 목적

Phase 30(GRU val_loss=0.9087) 기준으로 세 방향 탐색:
1. Focal loss 제거 → cross-entropy(CE)로 fall gradient 억제 해소
2. GRU+focal+val_loss 시드 다양성 (seed=42만 테스트됨)
3. GRU(256,128) + val_loss: 더 큰 모델, unbiased checkpoint 조합

## 결과

| ID | 손실 | 크기 | Seed | EventMinP | VidMinP | FN | FP | fall_pr | nfall_pr | fall_rc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `P32-ce-42` | CE | 128,64 | 42 | **0.9138** | 0.9138 | 17 | 20 | 0.9679 | 0.9258 | 0.9726 |
| `P32-fl-1` | Focal | 128,64 | 1 | 0.9110 | 0.9149 | 21 | 17 | 0.9724 | 0.9110 | 0.9662 |
| `P32-256-42` | Focal | 256,128 | 42 | 0.8966 | 0.8966 | 19 | 24 | 0.9617 | 0.9163 | 0.9694 |
| `P32-ce-1` | CE | 128,64 | 1 | 0.8941 | 0.8979 | 25 | 21 | 0.9660 | 0.8941 | 0.9597 |
| `P32-fl-3` | Focal | 128,64 | 3 | 0.8921 | 0.8996 | 26 | 17 | 0.9722 | 0.8921 | 0.9581 |
| `P32-fl-2` | Focal | 128,64 | 2 | 0.8903 | 0.8979 | 26 | 21 | 0.9659 | 0.8903 | 0.9581 |

## 전체 기준 비교

| 실험 | 아키텍처 | 손실 | Checkpoint | Seed | EventMinP |
|---|---|---|---|---:|---:|
| P27-vm0 | GRU(128,64) | Focal | val_video_min_pr | 42 | **0.9241** |
| P30-lstm | LSTM(128,64) | Focal | val_loss | 42 | 0.9181 |
| P32-ce-42 | GRU(128,64) | CE | val_loss | 42 | 0.9138 |
| P32-fl-1 | GRU(128,64) | Focal | val_loss | 1 | 0.9110 |
| P30-gru | GRU(128,64) | Focal | val_loss | 42 | 0.9087 |

## 분석

### CE vs Focal (val_loss 기준)
- seed=42: CE(0.9138) > Focal(0.9087) → CE가 FN 22→17 감소, FP 13→20 증가
- seed=1:  Focal(0.9110) > CE(0.8941) → seed-dependent 역전
- 손실함수 효과가 seed에 종속 → 일관된 방향성 없음

### GRU(256,128)+val_loss
- 0.8966 < GRU(128,64)+val_loss 0.9087 — **크기 증가가 오히려 역효과**
- FP=24(최다): 더 큰 모델이 오히려 non-fall 경계를 느슨하게 학습
- val_loss 환경에서 256,128 과적합 의심

### GRU+val_loss 상한선
- 6개 실험 중 최고 0.9138 — 어떤 조합도 0.92 미달
- val_loss+GRU의 구조적 상한선이 ~0.91-0.914 수준으로 보임

### 핵심 미탐색 조합
- **GRU + CE + val_video_min_pr**: P27-vm0(0.9241)의 focal → CE 교체
  - CE가 val_loss에서 GRU FN을 22→17로 감소시킨 효과가 val_video_min_pr에서도 재현될 경우
  - P27-vm0 FN=18이 ~14-15로 감소 → 0.93+ 가능성
- **LSTM + CE + val_loss**: P30-lstm(0.9181 with focal)의 CE 버전 미탐색

## 결론

1. **GRU+val_loss 방향 소진**: 최고 0.9138, P27-vm0(0.9241) 미달
2. **CE가 focal보다 GRU+val_loss seed=42에서 유리**: +0.005, 단 seed-dependent
3. **대형 GRU(256,128) val_loss 역효과**: capacity 증가 불필요
4. **전체 최고**: P27-vm0(0.9241) 유지

## 다음 단계 (Phase 33)

핵심 미탐색 조합 탐색:
- GRU(128,64) + CE + val_video_min_pr (seed=42, 1, 0)
- LSTM(128,64) + CE + val_loss (seed=42, 1)
