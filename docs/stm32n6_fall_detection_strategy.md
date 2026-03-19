# STM32N6 Fall Detection Model Development Strategy

## 1. Purpose

This document defines the engineering direction for developing a fall detection model that will eventually run on `STM32N6`, while training and evaluation are performed separately in `Colab` or another workstation environment.

The immediate goal is not just to train a model that scores well offline. The real goal is to train a model that:

- achieves strong detection quality on unseen videos
- remains deployable through the ST Edge AI toolchain
- survives `int8` quantization with limited accuracy loss
- uses operators that map well to the `STM32N6 Neural-ART NPU`

This document is intended to become the reference guide for subsequent notebook development, model comparison, export, and deployment validation.

## 2. Executive Summary

The practical conclusion from the ST documentation is clear:

- `TCN` should be the primary deployment model.
- `GRU` should be kept as an accuracy baseline, not as the main deployment target.
- The production model should be designed around `static input shapes`, `quantization-friendly layers`, and `NPU-supported operators`.
- The `95%+` target should be treated as a system-level target on a clearly defined metric, not as a vague training accuracy number.

The most realistic path is:

1. Build a clean `TCN` baseline with a fixed input window.
2. Build a matched `GRU` baseline on the same data split for comparison.
3. Push the `TCN` to strong float performance first.
4. Constrain the model architecture so it can be exported to `ONNX`, quantized to `int8`, and validated in the ST toolchain.
5. Optimize recall, false alarms, and latency together rather than optimizing frame accuracy alone.

## 3. Current Dataset Understanding

Based on the local dataset file [`dataset/final_dataset.csv`](/home/min/Workspace/Graduate-Project/Falling-Model-Development/dataset/final_dataset.csv):

- Total rows: `4,093,620`
- Total videos: `20,380`
- Total columns: `59`
- Label distribution:
  - label `0`: `3,778,880` frames
  - label `1`: `314,740` frames
- Positive-video distribution:
  - videos containing at least one positive frame: `15,273`
  - videos with only negative frames: `5,107`

The schema indicates:

- metadata columns: `video_id`, `frame`, `time_sec`
- pose features: `kp0` to `kp16` with `x`, `y`, `score`
- engineered features: `HSSC_y`, `HSSC_x`, `RWHC`, `VHSSC`
- target: `label`

This means the working feature set is naturally:

- `51` keypoint-derived pose values
- `4` engineered features
- total model input dimension: `55`

## 3.1 Operational Assumption for This Project

There is evidence that fall timing metadata existed during dataset preparation, but it was not carried cleanly into the current modeling pipeline.

For the current development phase, the project will use the following operational assumption:

- each clip is treated as a `10-second` sequence sampled at `15 fps`
- the effective monitoring segment is fixed to `5s` through `9s`
- this gives a fixed `4-second` input region
- therefore the model input is standardized to `60` frames

In frame indices, the default monitoring region is:

- start frame: `75`
- end frame: `134`
- length: `60` frames

This is a pragmatic decision for consistency and development speed. It is not the same as saying the metadata problem is solved. It means the project temporarily replaces unreliable event-start metadata with a deterministic shared monitoring window.

Practical consequence:

- the training notebooks should slice the input clip to `5~9s` first
- all models should assume a fixed input tensor equivalent to `60 x 55`
- any future recovery of reliable onset metadata should be treated as a later data-quality upgrade, not as a blocker for current model development

## 4. Why the Hardware Constraints Matter

`STM32N6` deployment is governed by the ST Edge AI stack, not just by whether a model runs in PyTorch.

### 4.1 Key deployment constraints from ST

- The `Neural-ART NPU` is for `inference-only`.
- Input and output tensors must be `static`.
- Mixed precision hybrid execution is not the preferred path; weights and activations should be quantized.
- Efficient NPU deployment is based on `int8` quantized models.
- If an operation stays in `float32`, it is executed in software on the host CPU rather than on the NPU.

Reference:

- [STM32Cube AI Studio documentation](https://wiki.st.com/stm32mcu/wiki/AI:STM32Cube_AI_Studio_documentation)
- [ST Neural-ART NPU - Supported operators and limitations](https://stedgeai-dc.st.com/assets/embedded-docs/stneuralart_operator_support.html)
- [Quantized models / quantization](https://stedgeai-dc.st.com/assets/embedded-docs/quantization.html)

### 4.2 Quantization implications

ST documents make the following points relevant to this project:

- deployment should assume `8-bit` quantization
- `ONNX` quantized deployment uses the `QDQ` pattern
- post-training quantization is possible, but quantization-aware design is safer if accuracy drops too much
- representative calibration data is required for static quantization
- dynamic-shape and dynamic-quantization style assumptions are not a good fit for this target

Practical implication:

- model architecture must be chosen with quantization in mind from the first notebook version
- large recurrent or normalization-heavy architectures that only work well in float are risky

Reference:

- [Quantization](https://stedgeai-dc.st.com/assets/embedded-docs/quantization.html)
- [ST Neural-ART NPU - Supported operators and limitations](https://stedgeai-dc.st.com/assets/embedded-docs/stneuralart_operator_support.html)

## 5. Operator-Level Analysis

The most important distinction is:

- `toolbox-supported` means the model can be parsed and converted
- `NPU-mapped` means the operation actually runs efficiently on the accelerator

These are not the same thing.

### 5.1 Operators that are favorable for this project

The following operator families are compatible with a deployment-oriented `TCN` design:

- `Conv1D / Conv`
- `BatchNorm1D / BatchNormalization`
- `Add`
- `Concat`
- `Pad / ConstantPad1D`
- `Relu`
- `Sigmoid / Logistic`
- `Flatten`
- `Gemm / Dense / FullyConnected`
- `Softmax`
- `AveragePool` and `GlobalAveragePool` with care

Relevant ST references:

- [ONNX toolbox support](https://stedgeai-dc.st.com/assets/embedded-docs/supported_ops_onnx.html)
- [TensorFlow Lite toolbox support](https://stedgeai-dc.st.com/assets/embedded-docs/supported_ops_tflite.html)
- [Keras toolbox support](https://stedgeai-dc.st.com/assets/embedded-docs/supported_ops_keras.html)
- [ST Neural-ART NPU - Supported operators and limitations](https://stedgeai-dc.st.com/assets/embedded-docs/stneuralart_operator_support.html)

### 5.2 Why TCN fits better than GRU

From the ST documentation:

- `ONNX Conv` includes `Conv1D/Conv2D`
- `QLinearConv` is supported
- `BatchNorm1D/BatchNorm2D` are mapped
- `Pad`, `Concat`, `Add`, `Relu`, `Gemm`, `Softmax` are supported or have hardware mapping paths

That is nearly the exact operator set needed for a deployment-friendly `TCN`.

By contrast:

- `GRU` appears in `ONNX` and `Keras` support documents
- but the `Neural-ART NPU` operator mapping does not list `GRU`
- ST also documents special constraints for `stateful LSTM/GRU`, including `batch-size = 1` and unsupported `return_state`
- recurrent state buffers require dedicated runtime handling

Practical conclusion:

- `GRU` is acceptable as a research baseline
- `GRU` is not the preferred deployment architecture for `STM32N6`
- `TCN` is the correct primary design path

Reference:

- [ONNX toolbox support](https://stedgeai-dc.st.com/assets/embedded-docs/supported_ops_onnx.html)
- [Keras toolbox support](https://stedgeai-dc.st.com/assets/embedded-docs/supported_ops_keras.html)
- [Keras stateful LSTM/GRU support](https://stedgeai-dc.st.com/assets/embedded-docs/keras_lstm_stateful.html)
- [ST Neural-ART NPU - Supported operators and limitations](https://stedgeai-dc.st.com/assets/embedded-docs/stneuralart_operator_support.html)

### 5.3 Operators to avoid or minimize

The following are risky, unnecessary, or likely to reduce deployment efficiency:

- custom layers
- unsupported post-processing inside the network graph
- recurrent-state-dependent runtime logic unless there is a strong reason
- float-only normalization or exotic activation chains
- dynamic shape logic
- excessive reshape/transposition complexity
- unsupported or CPU-fallback-heavy operators such as `Mean`, `Softplus`, `Softsign`, `Upsample`, or other float fallback paths when a simpler equivalent exists

Design rule:

- if a simpler equivalent exists using `Conv`, `Relu`, `Add`, `BatchNorm`, `Flatten`, `Dense`, prefer that version

## 6. Recommended Model Direction

### 6.1 Primary model: deployment-oriented TCN

Recommended first production architecture family:

- input shape: `(60, 55)`
- 3 to 5 temporal convolution blocks
- kernel size: prioritize `3`
- dilation schedule: mild, such as `1, 2, 4, 8`
- channels: start small, such as `32 -> 64 -> 64 -> 96`
- residual connections: allowed only if implemented with simple `Add`
- normalization: `BatchNorm1D` if needed
- activation: `ReLU`
- output head:
  - option A: last-timestep binary logit
  - option B: pooled sequence representation followed by small `Dense` head

Recommended first search range:

- window sizes:
  - primary fixed setting: `60` frames from the `5~9s` monitoring region
  - secondary ablation only if needed: `32`, `48`, `64`, `80`
- base channels: `24`, `32`, `48`
- depth: `3`, `4`, `5`
- dropout: `0.0` to `0.2`

### 6.2 Secondary model: GRU baseline

Recommended only as a comparison model:

- 1 or 2 `GRU` layers
- hidden sizes: `32`, `64`, `96`
- small dense classifier head
- no complex attention stack in the first phase

Purpose:

- answer whether recurrent memory adds enough value to justify the deployment pain
- establish whether the simpler `TCN` is already competitive

If `GRU` beats `TCN` only slightly in float performance but collapses after export or quantization, `TCN` still wins.

## 7. What “95%+ Accuracy” Should Mean

The project should not use raw frame accuracy as the main target.

With this dataset, frame-level accuracy can be misleading because:

- negative frames are the majority
- many fall sequences contain long negative regions
- a model can score high accuracy while still missing the actual fall onset

The target should be redefined into deployment-relevant metrics.

### 7.1 Primary metrics

For model selection, prioritize:

- `macro F1`
- `fall-class recall`
- `fall-class precision`
- `video-level detection rate`
- `false alarm rate per video`
- inference latency after export

### 7.2 Suggested acceptance criteria

Recommended development target:

- frame-level overall accuracy: `>= 95%`
- fall recall: `>= 95%`
- fall precision: `>= 92%`
- macro F1: `>= 0.94`
- unseen-video detection rate: `>= 95%`

These values are project targets, not guarantees. The important point is that all of them should be evaluated on `unseen videos`, not randomly shuffled frames.

## 8. Data Split Strategy

This is critical.

The dataset must be split by `video_id`, not by frame.

Reason:

- frames from the same video are highly correlated
- random frame split causes leakage
- leakage produces unrealistically high metrics and invalidates the result

Recommended split policy:

- train: `70%`
- validation: `15%`
- test: `15%`
- all splits grouped by `video_id`
- if possible, preserve label presence ratio across splits at the video level

Secondary hold-out:

- keep a final untouched deployment validation subset
- use it only after architecture and threshold selection are complete

## 9. Monitoring Window Strategy

Because the original fall-onset metadata is not currently reliable inside the modeling pipeline, the project will use a fixed monitoring segment.

Default rule:

- treat each sample as a `10-second` clip
- crop the effective model input to the `5~9s` segment
- feed only this `60-frame` region to the model

Why this is the current best choice:

- it standardizes all inputs to a static shape
- it matches STM32N6 deployment constraints well
- it removes ambiguity from the current pipeline
- it allows fast iteration while avoiding ad hoc per-sample event alignment logic

This should be treated as the project default until reliable onset metadata is restored.

## 10. Windowing Strategy

Fall detection is a temporal problem. The model must learn pre-fall, transition, and post-fall movement patterns.

Recommended approach:

- for the current phase, use the fixed `5~9s` monitoring region as the primary window
- if additional experiments are needed later, create fixed windows only inside valid clip boundaries
- label each window using the last frame or a short-horizon rule
- prevent windows from crossing video boundaries

Recommended experiments:

- last-frame label
- any-positive-in-window label
- horizon-aware label, such as positive if a fall occurs in the next `k` frames

For deployment, the best practical option in this project is:

- fixed `60-frame` monitoring input
- probability threshold with short temporal smoothing outside the model
- avoid unnecessary dynamic sliding logic in the embedded path unless later experiments prove it is necessary

## 11. Feature Engineering Direction

The current feature space already contains:

- raw keypoint positions
- keypoint confidence
- engineered body geometry features

To improve the chance of reaching the target, add sequence-friendly derived signals in the notebook pipeline:

- frame-to-frame keypoint velocity
- frame-to-frame keypoint acceleration
- torso angle change
- hip-to-shoulder displacement
- center-of-mass approximation
- confidence masking or confidence-weighted features

Important rule:

- any new feature added for training must also be reproducible in the embedded preprocessing pipeline

Do not add a feature that cannot be reproduced on-device.

## 12. Strategy for Actually Reaching 95%+

### 12.1 Phase 1: build a trustworthy baseline

First objective:

- make the evaluation trustworthy before trying to optimize the score

Checklist:

- grouped split by `video_id`
- reproducible seed
- class-balanced training sampler or weighted loss
- confusion matrix on unseen videos
- precision/recall curves
- threshold tuning on validation only

### 12.2 Phase 2: attack the class imbalance directly

The frame distribution is strongly imbalanced.

Recommended countermeasures:

- weighted cross entropy
- focal loss experiments
- oversampling positive windows
- hard negative mining
- balanced batch construction

Preferred order:

1. weighted loss
2. balanced window sampling
3. focal loss if false negatives remain high

### 12.3 Phase 3: improve the labels and event logic

Model quality is often limited by label quality rather than architecture.

Investigate:

- whether positive labels start too late
- whether pre-fall frames should be included
- whether transition frames should receive softened labels

High-value option:

- convert strict frame classification into event-aware classification using short forecasting or onset-aware labels

This often improves practical fall detection more than simply adding layers.

### 12.4 Phase 4: optimize thresholding for deployment

A binary classifier is not finished when training ends.

Tune:

- decision threshold
- minimum consecutive positive windows
- cooldown logic after alarm
- temporal smoothing over output probabilities

These steps should be outside the neural network if possible, because ST explicitly favors removing custom post-processing from the model graph.

Reference:

- [ST Neural-ART NPU - Supported operators and limitations](https://stedgeai-dc.st.com/assets/embedded-docs/stneuralart_operator_support.html)

### 12.5 Phase 5: quantization and export early, not late

Do not wait until the final week to test export.

Recommended milestone order:

1. float training baseline
2. export to `ONNX`
3. static quantization prototype
4. ST validation in toolchain
5. quantization-aware refinement if needed

If the float model cannot survive export and int8 quantization, it is not a valid candidate.

## 13. Architecture Rules for the Notebooks

The future `Colab` notebooks should follow these rules.

### 13.1 Common pipeline rules

- use the same grouped split logic in both notebooks
- use the same metric functions in both notebooks
- use the same export and calibration path in both notebooks
- keep preprocessing deterministic and documented
- save model config, thresholds, and feature schema together
- slice every sample to the shared `5~9s` monitoring region unless an experiment explicitly states otherwise

### 13.2 TCN notebook rules

- implement a pure PyTorch `TCN` using simple operators
- avoid external layers that complicate export unless necessary
- prefer explicit `Conv1d`, `BatchNorm1d`, `ReLU`, `Dropout`, `Linear`
- keep causal padding logic simple and test export early

### 13.3 GRU notebook rules

- treat it strictly as a baseline notebook
- do not allow the architecture to drift into a complex research model
- if it depends on runtime state management or export instability, stop escalating complexity

## 14. Concrete Development Plan

### 14.1 Notebook order

1. `tcn_stm32n6_baseline.ipynb`
2. `gru_baseline.ipynb`
3. `tcn_quantization_export.ipynb`

### 14.2 TCN experiment ladder

Run this order:

1. fixed `5~9s` monitoring input, shallow TCN
2. fixed `5~9s` monitoring input, moderate TCN
3. weighted loss
4. balanced window sampler or balanced clip sampler
5. feature augmentation with velocity and acceleration
6. threshold tuning
7. export and quantization check

### 14.3 GRU experiment ladder

Run this order:

1. 1-layer GRU, hidden `32`
2. 1-layer GRU, hidden `64`
3. 2-layer GRU, hidden `64`
4. threshold tuning
5. export feasibility check

Decision rule:

- if `GRU` is not materially better than `TCN` on unseen-video metrics, discontinue it as a deployment candidate

## 15. Risks

Main risks in this project:

- frame leakage due to incorrect split
- inflated accuracy from class imbalance
- model architecture that cannot be quantized cleanly
- high offline score but poor unseen-video detection
- strong float model that falls apart after `int8` export
- excessive dependence on preprocessing that is not reproducible on-device
- hidden label noise caused by incomplete or inconsistently transferred fall-onset metadata

## 16. Final Guidance

The correct development posture is:

- build for deployment from day one
- use `TCN` as the main path
- use `GRU` only as a comparison baseline
- evaluate on unseen videos
- standardize inputs to the shared `5~9s` monitoring region for now
- optimize recall, false alarms, and latency together
- validate export and quantization early

If the team follows this document, the resulting notebooks will be much more likely to produce a model that is not only accurate in `Colab`, but also realistic for `STM32N6`.

## 17. Reference Links

Core ST references:

- [STM32Cube AI Studio documentation](https://wiki.st.com/stm32mcu/wiki/AI:STM32Cube_AI_Studio_documentation)
- [Quantization](https://stedgeai-dc.st.com/assets/embedded-docs/quantization.html)
- [ONNX toolbox support](https://stedgeai-dc.st.com/assets/embedded-docs/supported_ops_onnx.html)
- [TensorFlow Lite toolbox support](https://stedgeai-dc.st.com/assets/embedded-docs/supported_ops_tflite.html)
- [Keras toolbox support](https://stedgeai-dc.st.com/assets/embedded-docs/supported_ops_keras.html)
- [Keras stateful LSTM/GRU support](https://stedgeai-dc.st.com/assets/embedded-docs/keras_lstm_stateful.html)
- [ST Neural-ART NPU - Supported operators and limitations](https://stedgeai-dc.st.com/assets/embedded-docs/stneuralart_operator_support.html)

Local project references:

- [dataset/final_dataset.csv](/home/min/Workspace/Graduate-Project/Falling-Model-Development/dataset/final_dataset.csv)
- [colab/tcn_v1_260317.ipynb](/home/min/Workspace/Graduate-Project/Falling-Model-Development/colab/tcn_v1_260317.ipynb)
