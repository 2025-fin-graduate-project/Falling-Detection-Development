# Handoff Document — Fall Detection GRU Optimization

## 현재 상태 (2026-05-14 01:xx)

### 실행 중인 프로세스
- **B-GRU-raw-v3** 학습 중
  - log: `results/baselines_phase0/B-GRU-raw-v3.log`
  - 설정: GRU(128,64) + Conv1D×2, stride=2(균형), no focal, dropout=0.3, noise=0.02, quiet mode
  - 데이터: `dataset/train.csv`, data_scope=no_by

### 실험 결과 비교
| 버전 | test F1 | Recall | Precision | 비고 |
|------|---------|--------|-----------|------|
| v1 | 0.8749 | 0.9378 | 0.8199 | GRU(64,32), 기본 설정 |
| v2 | 0.8875 | 0.9289 | 0.8496 | Conv1D+GRU(128,64), focal loss |
| v3 | 진행 중 | — | — | 균형 stride, no focal |

---

## 목표
- **Fall precision ≥ 0.90**
- **Non-fall precision ≥ 0.90**
- (둘 다 동시에 — 통합 precision이 아님)
- 오버피팅 없음

---

## 핵심 코드 변경사항 (이미 적용됨)
`scripts/train_baseline.py`:
- `conv_pre_layers`, `conv_pre_filters`, `conv_pre_kernel` 파라미터 추가 → Conv1D 전처리 블록
- `min_val_precision` → threshold 선택 시 **fall + non-fall precision 둘 다** 제약
- `select_threshold()`: fall_prec AND nfall_prec 모두 ≥ min_val_precision 조건

---

## 다음 실험 계획 (v3 완료 후 순서대로)

v3 결과 확인 후:
- **목표 달성** → 완료
- **미달** → 아래 순서로 진행

| 버전 | 주요 변경 | 명령 핵심 |
|------|-----------|-----------|
| v4 | stride=1 (완전 균형) + precision≥0.90 강화 | `--train-negative-stride 1 --min-val-precision 0.90` |
| v5 | splits_v2 데이터 (이상치 제거본) + all directions | `--train-csv dataset/splits_v2/train.csv --data-scope all` |
| v6 | Bidirectional GRU | `--bidirectional` |
| v7 | focal loss + 균형 stride 조합 (alpha=0.5) | `--focal-loss --focal-alpha 0.5 --train-negative-stride 2` |
| v8 | 더 큰 모델 GRU(256,128) | `--gru-units 256,128` |
| v9 | patience 늘리고 더 학습 | `--early-stop-patience 20 --epochs 150` |
| v10 | 최종 앙상블 threshold 튜닝 | threshold sweep 수동 조정 |

---

## 공통 베이스 명령어 템플릿
```bash
uv run python scripts/train_baseline.py \
  --experiment-id B-GRU-raw-vN \
  --model-type gru --preprocessing raw \
  --train-csv dataset/train.csv --val-csv dataset/val.csv --test-csv dataset/test.csv \
  --feature-set kp12 --data-scope no_by \
  --gru-units 128,64 \
  --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5 \
  --train-negative-stride 2 \
  --dropout-rate 0.3 --noise-std 0.02 \
  --early-stop-patience 15 \
  --min-val-precision 0.90 \
  --quiet \
  --output-root results/baselines_phase0 \
  2>&1 | tee results/baselines_phase0/B-GRU-raw-vN.log
```

---

## 재시작 후 할 일
1. `tail -5 results/baselines_phase0/B-GRU-raw-v3.log` → v3 완료 여부 확인
2. 완료됐으면 `grep "video-level test" results/baselines_phase0/B-GRU-raw-v3.log` → 결과 확인
3. 목표 미달 시 위 표에서 다음 버전 실행
