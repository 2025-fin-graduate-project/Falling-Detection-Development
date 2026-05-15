# Phase 10/11 Hard-Negative MinPR Plan

## Current State

Phase 9 changed the primary selection metric from `MinP` to:

```text
minPR = min(Fall Precision, NonFall Precision, Fall Recall, NonFall Recall)
```

The best current candidate is `P9O-v01`:

```text
model: GRU(128,64), kp7, 40 frames, 2xConv64, focal alpha=0.25
postprocess: threshold=0.525, min_consecutive=3
test CM: [[226, 19], [20, 646]]
test MinPR: 0.9187
```

This is very close to the first target `test MinPR >= 0.92`, but the result is not yet robust:

- The test split needs roughly one fewer FN, or a similarly small FP/FN shift, to pass.
- Validation MinPR for this setting is only `0.8848`.
- Phase 10 postprocessing sweep selected by validation did not improve test MinPR.
- The recurring bottleneck is `NonFall Recall`, meaning normal videos are still being classified as fall.

## Key Diagnosis

The model is not limited by fall detection capacity. Fall metrics are already high:

```text
P9O-v01 test:
FallP  = 0.9714
FallR  = 0.9700
NFallP = 0.9187
NFallR = 0.9224
```

Further increasing model sensitivity or fall-side weighting is likely to increase false positives. The next phase should explicitly defend the non-fall class.

## Why Simple Postprocessing Is Not Enough

The Phase 10 postprocessing sweep tested these video-level rules on `P9O-v01`:

- `threshold + min_consecutive`
- `threshold + min_consecutive + min_positive_count`
- `threshold + min_consecutive + positive_ratio`
- `top-k mean score`
- hysteresis with low/high thresholds

Best rule selected by validation:

```text
rule: threshold=0.475, min_consecutive=4
val  MinPR=0.8938, FP=51, FN=44
test MinPR=0.8980, FP=25, FN=17
```

Best test-oracle rule remained effectively the current rule:

```text
rule: threshold=0.525, min_consecutive=3
val  MinPR=0.8848, FP=50, FN=56
test MinPR=0.9187, FP=19, FN=20
```

This suggests the remaining false positives are not just short score spikes. Some normal videos likely contain sustained fall-like windows.

## Phase 10 Goal

Build an evidence report for false-positive and false-negative videos, then use it to define hard-negative training inputs.

Phase 10 should not start by adding more random model variants. It should answer:

1. Which normal videos are false positives?
2. Are false-positive scores short spikes or sustained regions?
3. Are false positives concentrated by camera, direction, source, or low keypoint confidence?
4. Which fall videos are false negatives under stricter non-fall-safe settings?
5. Can the high-scoring normal windows be mined as hard negatives for retraining?

## Phase 10 Deliverables

### 1. FP/FN Timeline Report

Add a script, tentatively:

```text
scripts/report_phase10_error_timelines.py
```

Inputs:

```text
--exp-dir results/phase9_minpr_open/P9O-v01
--output-dir results/phase10_error_analysis/P9O-v01
--threshold 0.525
--min-consecutive 3
```

Outputs:

```text
false_positive_videos.csv
false_negative_videos.csv
video_score_summary.csv
fp_score_timelines.csv
fn_score_timelines.csv
```

Recommended per-video fields:

```text
video_id
true_label
pred_label
direction
max_score
top3_mean
top5_mean
positive_count
positive_ratio
max_consecutive
first_positive_window
last_positive_window
mean_keypoint_confidence
min_keypoint_confidence
```

The timeline CSV should keep at least:

```text
video_id
window_index
window_start_frame
window_start_sec
score
binary_at_selected_threshold
label
direction
confidence_mean
```

### 2. Hard-Negative Mining Dataset

From false-positive normal videos, mine high-score negative windows.

Initial mining rule:

```text
true video label == non-fall
score >= 0.40
or in top 10 windows per false-positive video
```

Output:

```text
results/phase10_error_analysis/P9O-v01/hard_negative_windows.csv
```

Recommended fields:

```text
video_id
window_start_frame
window_start_sec
score
rank_in_video
feature_set
target_steps
source_split
```

This file should not replace the dataset. It should be an auxiliary mining artifact used by the next training phase.

### 3. MinPR Checkpointing

Current training uses `val_loss` for early stopping and best-weight restoration. That can miss the best epoch for video-level MinPR.

Add a validation callback that:

1. Predicts validation scores at each epoch or every `N` epochs.
2. Sweeps threshold and `min_consecutive`.
3. Computes video-level `val_min_pr`.
4. Saves the model weights when `val_min_pr` improves.
5. Restores the best `val_min_pr` checkpoint before final test evaluation.

Suggested option:

```text
--checkpoint-monitor val_video_min_pr
```

Keep `val_loss` as the default for normal training, but use `val_video_min_pr` for Phase 10/11 candidates.

### 4. Hard-Negative Training Modes

Add one conservative training option at a time.

Candidate options:

```text
--hard-negative-windows results/phase10_error_analysis/P9O-v01/hard_negative_windows.csv
--hard-negative-repeat 2
```

or:

```text
--hard-negative-video-ids results/phase10_error_analysis/P9O-v01/false_positive_videos.csv
--hard-negative-stride 1
```

Preferred first implementation:

- Keep normal training windows unchanged.
- Oversample windows from false-positive videos by lowering their negative stride to `1`.
- Avoid adding test-derived artifacts into final training. Use validation false positives first for principled training selection.

## Phase 11 Training Candidates

Only start these after the Phase 10 error report exists.

Common base:

```text
GRU(128,64)
kp7
40 frames
2xConv64
focal alpha=0.25
dropout=0.3
noise_std=0.02
min_consecutive_values=1,2,3,4,5,7,9
threshold_count=37
checkpoint_monitor=val_video_min_pr
```

Candidate set:

```text
P11-v01: base + val_video_min_pr checkpointing only
P11-v02: P11-v01 + validation-FP-video negative stride=1
P11-v03: P11-v01 + mined hard-negative windows repeat=2
P11-v04: P11-v03 + normal class weight multiplier
P11-v05: P11-v03 + calibrated logit-margin score sweep
```

Do not use label 3 / LB-3 for this phase. The model target remains fallen-vs-non-fallen binary detection.

## Acceptance Criteria

Primary gate:

```text
float test video-level MinPR >= 0.92
```

Stability checks:

```text
val video-level MinPR should improve over P9O-v01 baseline
test gain should not come with a large validation drop
Fall Recall should stay >= 0.95
NonFall Recall should improve or remain >= 0.92
```

Deployment checks for passing candidates:

```text
STedgeAI analyze
STedgeAI host eval with INT8 threshold reselect
```

INT8/STedgeAI remains a deployment-readiness metric, not the first gate.

## Current Recommendation

Proceed in this order:

1. Generate `P9O-v01` FP/FN timeline report.
2. Mine validation false-positive hard negatives.
3. Add `val_video_min_pr` checkpointing.
4. Run P11 candidates using the same `P9O-v01` architecture.
5. Only add data if hard-negative mining and MinPR checkpointing fail to pass `0.92` robustly.
