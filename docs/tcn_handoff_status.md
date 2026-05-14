# TCN Handoff Status

## Purpose

This note summarizes the current TCN baseline work, the active evaluation logic, and the main conflicts that must be carried into the next handoff.

## Current State

- Branch: `codex/tcn-baseline`
- Active work centered on TCN baselines for STM32N6 deployment.
- GPU on WSL is functional.
- INT8 export is now working from the tracked Keras model path.
- The earlier LB-3 evaluation rule that treated `fallen` as a trigger was identified as a mismatch for event detection and has been revised in code.

## Fixed Formulas

### 1. LB-2 binary event model

For a 60-frame window:

```text
window_label = max(frame_labels)
positive = 1
score = p(class 1)
alarm = score >= threshold for min_consecutive windows
```

This is the standard fall-event baseline.

### 2. Old LB-3 rule that caused the conflict

The earlier LB-3 setup used:

```text
window_label = max(frame_labels)
positive_labels = [1, 2]
fall_score = p(class 1) + p(class 2)
```

This was valid for a "falling or fallen state" detector, but not for a pure fall-event detector. It allowed `fallen` to dominate when a window contained both falling and fallen frames.

### 3. Revised LB-3 event rule

The revised event rule is:

```text
if any(frame_label == 1):
    window_label = 1
elif any(frame_label == 2):
    window_label = 2
else:
    window_label = 0
```

And the alarm trigger becomes:

```text
positive_labels = [1]
score = p(class 1)
alarm = score >= threshold for min_consecutive windows
```

Class 2 remains a contextual post-fall state, not the sole trigger for a new event alarm.

### 4. Threshold selection

For validation windows:

```text
video_alarm = any(filtered_window_score >= threshold over min_consecutive windows)
```

The threshold and consecutive count are selected by a 2D sweep on validation data, maximizing F1 subject to precision constraints.

## Progress Summary

### Best completed results before the LB-3 event-rule correction

| Experiment | Dataset | Test F1 | Fall P | Non-fall P | Min P | Notes |
|---|---|---:|---:|---:|---:|---|
| `TCN-SAFE-v03` | raw LB-2 | 0.9088 | 0.8904 | 0.8711 | 0.8711 | Strongest LB-2 safe model |
| `TCN-VAR-v01` | filtered LB-2 | 0.9196 | 0.9048 | 0.8854 | 0.8854 | Best LB-2 filtered model |
| `TCN-VAR-v02` | raw LB-3 | 0.9686 | 0.9540 | 0.9491 | 0.9491 | High score, but old LB-3 logic favored fallen-state detection |
| `TCN-TUNE-v02` | raw LB-3 | 0.9722 | 0.9582 | 0.9585 | 0.9582 | Best float/INT8 baseline before event-rule correction |

### INT8 export status

- `model.keras` -> `model_int8.tflite` works when using `TFLiteConverter.from_keras_model(model)`.
- The earlier `from_concrete_functions()` path hit `READ_VARIABLE` failures.
- Pooling heads were changed from `Flatten` to `Reshape` to reduce dynamic-shape ops in the export graph.

## Conflict Summary

### 1. LB-3 semantic conflict

The project originally treated LB-3 as:

```text
1 = falling
2 = fallen
positive = [1, 2]
```

That is not wrong for a general fall-state classifier, but it is wrong for a pure fall-event detector when the desired trigger must fire on `falling`.

### 2. Model interpretation conflict

The high-scoring LB-3 models were mostly detecting `fallen` context. This produced strong event-level metrics, but it was not the desired semantics for alarm logic.

### 3. Deployment conflict

The core deployment target is STM32N6. That means:

- Prefer INT8-safe operators.
- Avoid unsupported dilation and dynamic graph tricks.
- Keep the final event rule simple enough to port into C.

## What To Carry Forward

1. Use `falling_priority` for LB-3 training windows.
2. Use `positive_labels = [1]` for the event alarm.
3. Keep `class 2` only as post-fall context.
4. Re-run the LB-3 candidates under the new event rule before selecting a final baseline.
5. Preserve the INT8 export path from Keras.

## Files That Matter

- [scripts/train_baseline.py](/home/min/Falling-Detection-Development/scripts/train_baseline.py)
- [scripts/auto_tcn_experiments.py](/home/min/Falling-Detection-Development/scripts/auto_tcn_experiments.py)
- [docs/tcn-baseline-auto-plan.md](/home/min/Falling-Detection-Development/docs/tcn-baseline-auto-plan.md)
- [CLAUDE.md](/home/min/Falling-Detection-Development/CLAUDE.md)

