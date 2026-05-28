# Handoff — Falling Model Development

작성 시각: 2026-05-16 01:30 KST

## 현재 브랜치와 Git 상태

- 현재 브랜치: `codex/class-balanced-splits`
- 원격 push 완료된 최신 커밋:
  - `7a03775 docs: align agent instructions`
  - 직전 주요 커밋:
    - `9983277 eval: add event minpr checkpoint run`
    - `75aa6ca docs: add per-phase experiment summaries`
    - `41fb924 eval: add event-threshold class-balanced GRU run`
    - `e60c8dd dataset: add class-balanced split options`

현재 남아 있는 변경/미추적 파일은 이전 작업 또는 다른 agent/user 작업으로 보이며, 이번 handoff에서 건드리지 않았다.

```text
 M docs/tasks/phase11-loop-strategy.md
?? scripts/run_overnight.sh
?? scripts/run_phase12_focal_lb3.sh
?? scripts/run_phase13_alpha_negstride.sh
?? scripts/run_phase14_alpha_sweep.sh
?? scripts/run_phase15_alpha25_sweep.sh
?? scripts/run_phase16_short_epochs.sh
?? scripts/run_phase17_velocity_features.sh
?? scripts/run_phase18_tcn.sh
?? scripts/run_phase19_gru256.sh
?? scripts/util/build_filtered_v2_splits_kv.py
```

## tmux 상태

현재 tmux 세션:

```text
auto-work
codex
phase21-event-minpr
```

`phase20-class-balanced`는 완료 후 종료됐다.

현재 실행 중인 핵심 세션:

```bash
tmux attach -t phase21-event-minpr
```

상태:

- Phase21 `P21-v01` 실행 중
- 데이터 로딩 및 window 생성 완료
- GPU에 올라가 학습 중
- 로그:
  - `results/phase21_event_minpr_checkpoint/summary.log`
  - `results/phase21_event_minpr_checkpoint/P21-v01.log`

최근 로그 요약:

```text
[2026-05-16 01:23:03] Phase 21 시작
[2026-05-16 01:25:03] GPU ready
[2026-05-16 01:25:03] START P21-v01
train rows=1,211,093 videos=6,039
val rows=344,292 videos=1,714
test rows=172,209 videos=853
train windows=353,150 positive=213,662 videos=6,039
val windows=70,274 positive=30,106 videos=1,714
test windows=34,973 positive=15,069 videos=853
```

## 데이터셋 상태

새 class-balanced split 생성 완료:

- Raw split: `dataset/splits_v2_class_balanced/`
- Filtered split: `dataset/splits_v2_class_balanced_filtered/`

분포:

| Split | Videos | Rows | Positive frame ratio |
| --- | ---: | ---: | ---: |
| train | 6039 | 1,211,093 | 0.07492 |
| val | 1714 | 344,292 | 0.07596 |
| test | 853 | 172,209 | 0.07595 |

검증:

- train/val/test 간 video overlap: 0
- split 기준: `direction,video_label`
- frame label balancing 적용

생성 명령:

```bash
python3 scripts/util/build_v2_dataset.py \
  --input dataset/final_dataset.csv \
  --out-dir dataset/splits_v2_class_balanced \
  --stratify-keys direction,video_label \
  --balance-frame-labels \
  --label-column label \
  --positive-labels 1 \
  --seed 42

python3 scripts/util/build_filtered_v2_splits.py \
  --src-dir dataset/splits_v2_class_balanced \
  --dst-dir dataset/splits_v2_class_balanced_filtered
```

`dataset/*`는 `.gitignore` 대상이라 커밋하지 않았다.

## 평가 기준 정리

현재 논의된 기준:

- 단순 `video-level`만 주 지표로 두면 낙상 영상에서 엉뚱한 구간의 positive도 TP가 될 수 있다.
- 반대로 `event-level`만 주 지표로 두면 후처리/threshold가 event metric에 과하게 맞춰져 비정상적으로 좋아 보일 수 있다.
- 현재 실험 방향:
  - P20: `val_loss`로 epoch 선택, threshold 선택은 `event`
  - P21: epoch 선택과 threshold 선택 모두 `val_event_min_pr`

권장 보고 방식:

- `test_video`와 `test_event_video`를 모두 보고한다.
- threshold 선택 기준과 checkpoint 기준을 반드시 같이 기록한다.
- 최종 판단은 FP/FN count와 confusion matrix를 함께 본다.

## Phase20 완료 결과

실험 디렉토리: `results/phase20_class_balanced_gru`

공통 설정:

- dataset: `dataset/splits_v2_class_balanced_filtered`
- GRU 계열
- threshold selection: `--threshold-eval-level event`
- checkpoint monitor: `val_loss`
- event tolerance: 2 windows

| ID | 변경점 | Test Event MinPR | Test Video MinPR | Threshold / mc | Event CM |
| --- | --- | ---: | ---: | --- | --- |
| `P20-v01` | GRU(128,64), alpha=0.25, neg_stride=2 | 0.9121 | 0.9160 | 0.525 / 2 | TN=218 FP=14 FN=21 TP=600 |
| `P20-v02` | alpha=0.15 | 0.8934 | 0.8971 | 0.450 / 3 | TN=218 FP=14 FN=26 TP=595 |
| `P20-v03` | neg_stride=1 | 0.9142 | 0.9181 | 0.500 / 3 | TN=213 FP=19 FN=20 TP=601 |
| `P20-v04` | GRU(96,48) | 0.8966 | 0.8966 | 0.500 / 4 | TN=208 FP=24 FN=24 TP=597 |

판단:

- P20 best는 `P20-v03`.
- 기존 P9O-v01과 test split이 달라 직접 동등 비교는 조심해야 한다.
- 새 split과 event threshold 기준에서는 `P20-v03`이 가장 균형이 좋다.
- `P20-v02` alpha=0.15는 FN 증가로 실패.
- `P20-v04` compact GRU는 성능 손실이 커서 우선순위 낮음.

## Phase21 진행 중

실험 디렉토리: `results/phase21_event_minpr_checkpoint`

목적:

- P20은 threshold만 event 기준으로 골랐다.
- Phase21은 학습 중 best epoch 선택도 `val_event_min_pr`로 바꿔, 진짜 event-level MinPR에 맞춰 학습한다.

코드 변경:

- `scripts/train_baseline.py`
  - `--checkpoint-monitor val_event_min_pr` 추가
  - `ValMinPRCallback`이 epoch마다 validation event-level MinPR sweep
  - best event MinPR epoch의 weight 복원

실행 스크립트:

- `scripts/run_phase21_event_minpr_checkpoint.sh`

실험 계획:

| ID | 설정 | 상태 |
| --- | --- | --- |
| `P21-v01` | P20-v01 대응, GRU(128,64), neg_stride=2 | 실행 중 |
| `P21-v02` | P20-v03 대응, GRU(128,64), neg_stride=1 | 대기 |

확인 명령:

```bash
tmux attach -t phase21-event-minpr
tail -n 80 results/phase21_event_minpr_checkpoint/summary.log
tail -n 80 results/phase21_event_minpr_checkpoint/P21-v01.log
```

결과 요약 명령:

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path("results/phase21_event_minpr_checkpoint")
for d in sorted(root.glob("P21-v*")):
    mj = d / "metrics.json"
    if not mj.exists():
        print(d.name, "running/no metrics")
        continue
    m = json.loads(mj.read_text())
    thr = m.get("threshold_selection", {})
    tv = m.get("metrics", {}).get("test_video", {})
    te = m.get("metrics", {}).get("test_event_video", {})
    print(
        d.name,
        "event", round(te.get("min_pr", -1), 4),
        "video", round(tv.get("min_pr", -1), 4),
        "thr", thr.get("threshold"),
        "mc", thr.get("min_consecutive"),
        "eval", thr.get("eval_level"),
    )
PY
```

## 문서화 상태

완료된 Phase 11~19는 별도 docs 파일로 작성 및 커밋 완료:

- `docs/tasks/phase11-summary.md`
- `docs/tasks/phase12-summary.md`
- `docs/tasks/phase13-summary.md`
- `docs/tasks/phase14-summary.md`
- `docs/tasks/phase15-summary.md`
- `docs/tasks/phase16-summary.md`
- `docs/tasks/phase17-summary.md`
- `docs/tasks/phase18-summary.md`
- `docs/tasks/phase19-summary.md`

Phase20은 완료됐으므로 다음 담당자는 아래 문서를 추가해야 한다:

- `docs/tasks/phase20-summary.md`
- `results/phase20_class_balanced_gru/strategy.md` 업데이트 또는 생성

Phase21은 아직 실행 중이므로 완료 후에만 문서화한다:

- `docs/tasks/phase21-summary.md`
- `results/phase21_event_minpr_checkpoint/strategy.md`

공통 규칙은 `AGENTS.md`와 `CLAUDE.md`에 반영돼 있다. 완료된 phase는 반드시 `results/<phase>/strategy.md`와 `docs/tasks/phaseNN-summary.md`를 남긴다.

## 다음 액션

1. `phase21-event-minpr` tmux에서 `P21-v01` 완료 여부 확인.
2. `P21-v01` 완료 후 `test_event_video`, `test_video`, CM 비교.
3. `P21-v02`까지 완료되면 P20 best인 `P20-v03`과 비교.
4. Phase20 문서화:
   - `docs/tasks/phase20-summary.md`
   - `results/phase20_class_balanced_gru/strategy.md`
5. Phase21 완료 후 문서화.
6. 다음 가설 후보:
   - robust feature clipping
   - derivative smoothing 강화
   - confidence threshold 재생성 sweep
   - event metric은 주 지표가 아니라 제약 또는 보조 지표로 둘지 재검토
