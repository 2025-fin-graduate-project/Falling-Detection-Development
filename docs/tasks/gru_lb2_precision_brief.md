# Agent Brief — GRU LB-2 Precision Push

**작성일**: 2026-05-14  
**목표**: GRU (TFLite INT8 양자화 이전) LB-2 기준 **MinP ≥ 0.93**  
**MinP 정의**: `min(FallP, NFallP)` — 낙상 정밀도와 비낙상 정밀도 중 낮은 쪽  
**배경**: 이 지표는 "낙상 이벤트 감지" 성능 기준 (넘어진 후 정지 상태 감지 X)

---

## 1. 현재 성과 요약

### GRU LB-2 최고 결과 (이 이상을 넘어야 함)

| ID | Arch | Win | KP | F1 | MinP | FallP | NFallP |
|----|------|-----|----|----|------|-------|--------|
| P1-v10 | GRU(256,128) bidir | 60f | kp12 | 0.9220 | 0.8903 | 0.8903 | 0.9167 |
| P2-v02 | GRU(128,64) bidir | 60f | kp12 | 0.9207 | 0.8906 | 0.9036 | 0.8906 |
| P2-v03 | GRU(128,64) bidir+focal | 60f | kp12 | 0.9164 | **0.8905** | 0.8905 | 0.8961 |
| P2-v01 | GRU(256,128) bidir+focal | 60f | kp12 | 0.9179 | 0.8824 | 0.9031 | 0.8824 |

**현재 GRU best MinP = 0.8906** → 목표까지 +0.039 필요

### 비교 기준: TCN best (이미 달성)

| ID | Arch | Win | MinP | 비고 |
|----|------|-----|------|------|
| P2-v06 | TCN[64,64,128,128]+focal | 60f | 0.9044 | 현재 전체 최고 |

TCN 기준에서도 0.93에는 못 미침. TCN Phase 5 (30f window)가 현재 실험 중.

---

## 2. 이미 실행/예약된 실험 (중복 금지)

다음은 별도 체인에서 이미 실행되고 있음. 결과가 나올 때까지 중복 실험 하지 말 것:

| 실험 | 커버하는 내용 | 출력 경로 |
|------|-------------|-----------|
| Phase 3 (P3-v01~v08) | GRU bidir, LB-2/LB-3, 30f/40f, [3,9)s | `results/gru_phase3_2s/` |
| Phase 4 (P4-v01~v08) | TCN+focal, KP ablation (minimal/kp7/kp12/all × 60f/30f) | `results/gru_phase4_kp/` |
| Phase 5 (P5-v01~v06) | TCN+focal+30f, std/large channels, kp7/minimal | `results/tcn_phase5_30f/` |

---

## 3. 미시도 전략 — 우선순위 순

### 전략 A: Short Window × KP Reduction (즉시 실행 가능)

**가설**: 30f 윈도우는 낙상 동작 구간(~21프레임) 포착률을 4.2% → 89%로 높임.  
여기에 노이즈 관절(손목 등)을 제거한 kp7 또는 minimal을 결합하면 GRU도 TCN 수준 달성 가능.

Phase 3은 `kp12 + 30f + bidir/focal` 만 다룸. 다음 조합이 비어 있음:

| 실험 ID | 내용 | 핵심 변인 |
|---------|------|----------|
| G6-v01 | GRU(256,128) bidir+focal, 30f, kp7 | 권장 KP × short window |
| G6-v02 | GRU(256,128) bidir+focal, 30f, minimal | 최소 KP × short window |
| G6-v03 | GRU(256,128) bidir+focal, 30f, kp12, larger (512,256) | 더 큰 용량 |
| G6-v04 | GRU(256,128) bidir+focal, 60f, kp7 | KP만 줄이고 window 유지 |

출력 경로: `results/gru_phase6_kp_win/`

### 전략 B: Hard Negative Mining (train_baseline.py 수정 필요)

**가설**: GRU가 FP를 내는 "어려운 정상 구간"(정상인데 낙상으로 예측)을 val set에서 수집해  
다음 학습의 training set에 hard negative로 추가하면 FallP(비낙상 정밀도 오류 감소) 향상.

구현 방법:
1. 1차 모델로 val set 추론 → FP 창 인덱스 수집
2. FP 창에 2× sample weight 부여 or train CSV에 별도 hard_neg 플래그 컬럼 추가
3. `train_baseline.py`에 `--hard-neg-weight` 옵션 추가

출력 경로: `results/gru_phase6_hardneg/`

### 전략 C: GCN+GRU Hybrid (train_baseline.py 신규 모델 구현 필요)

**가설**: 17개 관절의 skeleton graph 구조(뼈대 연결 관계)를 명시적으로 학습하면  
현재 GRU가 놓치는 낙상 방향별 패턴(전/후/측면)을 포착 가능.

구현 개요:
- 각 프레임에서 1~2 layer Spatial GCN: `A @ X @ W` (A = 고정 skeleton adjacency matrix)
- 프레임별 GCN embedding을 시계열로 쌓아 GRU에 입력
- Global feature (HSSC/VHSSC/AHSSC)는 GRU output과 concat 후 Dense classifier
- `--model-type gcn_gru`로 분기

```
입력: (batch, 60frames, 17joints, 3~5channels[x,y,score,dx,dy])
  └→ per-frame SpatialGCN → (batch, 60, hidden_dim)
  └→ GRU → (batch, hidden_dim)
  └→ concat global_feat → Dense(2) → Softmax
```

**TFLite INT8 호환 주의**: `tf.linalg.matmul` 기반 GCN은 TFLite 변환 가능.  
`torch_geometric` 등 외부 라이브러리 사용 금지. TF/Keras 순수 ops만 사용.

출력 경로: `results/gru_phase6_gcn/`

---

## 4. 실험 설정 — 고정값 (변경 불가)

```bash
# 공통 고정값 (CLAUDE.md 준수)
--preprocessing filtered
--train-csv dataset/splits_v2_filtered/train.csv
--val-csv   dataset/splits_v2_filtered/val.csv
--test-csv  dataset/splits_v2_filtered/test.csv
--label-column label          # LB-2 고정
--data-scope all              # 모든 방향 포함
--dropout-rate 0.3
--noise-std 0.02
--train-negative-stride 2
--early-stop-patience 15
--epochs 100
--min-val-precision 0.90
--model-type gru
--conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5  # conv-pre 제거 금지
```

> `conv-pre` 제거 금지: P2-v07 ablation에서 MinP 0.8987 → 0.8571로 급락 확인됨.

### 30f 윈도우 설정 (전략 A)

```bash
--target-steps 30
--window-start-sec 3.0
--window-end-sec   9.0
```

### 60f 윈도우 설정 (기존과 동일)

```bash
--target-steps 60
--window-start-sec 5.0
--window-end-sec   9.0
```

---

## 5. 전략 A 실험 스크립트 템플릿

브랜치: `experiment/gru-lb2-precision-phase6`

```bash
#!/usr/bin/env bash
# Phase 6 — GRU LB-2 Precision Push
# Target: MinP ≥ 0.93

set -uo pipefail
_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="${_SITE}/nvidia/cudnn/lib:${_SITE}/nvidia/cufft/lib:${_SITE}/nvidia/cusolver/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OUTROOT="results/gru_phase6_kp_win"
SUMMARY="$OUTROOT/summary.log"
mkdir -p "$OUTROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }

run_exp() {
    local id="$1"; shift
    local logfile="$OUTROOT/${id}.log"
    if [[ -f "$OUTROOT/$id/metrics.json" ]]; then
        log "SKIP  $id — already complete"; return 0
    fi
    log "START $id"
    if uv run python scripts/train_baseline.py \
            --experiment-id "$id" --output-root "$OUTROOT" --quiet "$@" \
            2>&1 | tee "$logfile"; then
        log "OK    $id"
    else
        log "FAIL  $id"
    fi
}

BASE=(
    --model-type gru
    --gru-units 256,128 --bidirectional
    --conv-pre-layers 2 --conv-pre-filters 64 --conv-pre-kernel 5
    --focal-loss --focal-gamma 2.0 --focal-alpha 0.25
    --preprocessing filtered
    --train-csv dataset/splits_v2_filtered/train.csv
    --val-csv   dataset/splits_v2_filtered/val.csv
    --test-csv  dataset/splits_v2_filtered/test.csv
    --label-column label --data-scope all
    --dropout-rate 0.3 --noise-std 0.02
    --train-negative-stride 2
    --early-stop-patience 15 --epochs 100
    --min-val-precision 0.90
)
WIN30=(--target-steps 30 --window-start-sec 3.0 --window-end-sec 9.0)
WIN60=(--target-steps 60 --window-start-sec 5.0 --window-end-sec 9.0)

# G6-v01: kp7 × 30f (권장 KP + short window)
run_exp G6-v01 "${BASE[@]}" "${WIN30[@]}" --feature-set kp7

# G6-v02: minimal × 30f (최소 KP + short window)
run_exp G6-v02 "${BASE[@]}" "${WIN30[@]}" --feature-set minimal

# G6-v03: kp12 × 30f, GRU(512,256) — larger capacity
run_exp G6-v03 "${BASE[@]}" "${WIN30[@]}" --feature-set kp12 --gru-units 512,256

# G6-v04: kp7 × 60f — KP만 줄이고 window는 기존 유지
run_exp G6-v04 "${BASE[@]}" "${WIN60[@]}" --feature-set kp7

log "=== Phase 6 KP×Win results ==="
# ... (summary 출력 블록)
```

---

## 6. 성공 기준 및 평가

```bash
# 결과 확인 명령
python3 -c "
import json, glob
for f in sorted(glob.glob('results/gru_phase6_*/**/metrics.json', recursive=True)):
    m = json.load(open(f))
    tv = m['metrics'].get('test_video', {})
    minp = tv.get('min_precision', 0)
    mark = ' ★★' if minp >= 0.93 else (' ★' if minp >= 0.91 else '')
    print(f'{f.split(\"/\")[-2]:12s}  F1={tv.get(\"f1\",0):.4f}  MinP={minp:.4f}{mark}')
"
```

| MinP 범위 | 판정 |
|-----------|------|
| ≥ 0.95 | 목표 초과 달성 ★★ |
| 0.93~0.95 | 목표 달성 ★ |
| 0.91~0.93 | 부분 개선, 추가 실험 필요 |
| < 0.91 | 퇴보, 원인 분석 후 중단 |

---

## 7. 실험 완료 후 처리

1. `results/gru_phase6_*/summary.log` 저장 확인
2. best model을 `docs/` 산하에 결과 표로 기록
3. 브랜치 `experiment/gru-lb2-precision-phase6` → PR to `dev`
4. MinP ≥ 0.93 달성 시 INT8 양자화 결과(`test_int8` 지표)도 함께 보고

---

## 8. 참고 파일

| 파일 | 내용 |
|------|------|
| `CLAUDE.md` | 전체 실험 파라미터 규칙 |
| `scripts/train_baseline.py` | 모델 구현 및 실험 러너 |
| `docs/analysis/dataset_analysis_report.md` | KP 선정 분석 (§3 키포인트 선정) |
| `docs/colab-training-analysis-and-gcn-plan.md` | hard negative mining, GCN+GRU 구현 가이드 |
| `docs/stm32n6_fall_detection_strategy.md` | TFLite INT8 배포 제약 |
| `results/gru_phase2_arch/` | Phase 2 GRU 기준 결과 |
| `results/gru_phase3_2s/` | Phase 3 30f 결과 (완료 후 참조) |
