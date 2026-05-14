# ST Edge AI — Neural-ART NPU 연산자 지원 현황

> 작성일: 2026-05-14  
> 참고: ST Edge AI Core Technology Documentation, ST Community  
> 목적: STM32N6 배포 전략 수립 및 보고서 참고

---

## 1. Neural-ART NPU 개요

STM32N6에 탑재된 ST 자체 설계 NPU. **INT8 scale/offset 포맷으로 양자화된 연산만 NPU에서 직접 가속**되며, 미지원 연산은 Cortex-M55 CPU로 자동 fallback된다. ST Edge AI Core 툴이 연산자 단위로 스케줄링을 결정한다.

---

## 2. Neural-ART NPU 지원 연산자 목록 (TFLite 기준, 38종)

| 카테고리 | 연산자 |
|---|---|
| Convolution | `CONV_2D`, `DEPTHWISE_CONV_2D`, `TRANSPOSE_CONV` |
| Pooling | `AVERAGE_POOL_2D`, `MAX_POOL_2D` |
| Fully Connected | `FULLY_CONNECTED` |
| Activation | `RELU`, `RELU6`, `LEAKY_RELU`, `PRELU`, `HARD_SWISH`, `LOGISTIC`, `TANH` |
| Element-wise | `ADD`, `MUL`, `SUB`, `ABS`, `CEIL` |
| Reshape / Slice | `RESHAPE`, `SQUEEZE`, `EXPAND_DIMS`, `STRIDED_SLICE`, `SPLIT`, `SPLIT_V` |
| Matrix | `BATCH_MATMUL`, `TRANSPOSE` |
| Data | `CONCATENATION`, `PAD`, `PACK`, `UNPACK`, `SPACE_TO_DEPTH`, `RESIZE_NEAREST_NEIGHBOR` |
| 정규화 / 변환 | `(RE)QUANTIZE`, `CAST` |
| 논리 | `EQUAL`, `LOGICAL_AND`, `LOGICAL_NOT`, `LOGICAL_OR` |

> 출처: [ST Neural-ART NPU - Supported operators and limitations](https://stm32ai-cs.st.com/assets/embedded-docs/stneuralart_operator_support.html)

---

## 3. GRU / LSTM 지원 현황

| 항목 | 상태 | 비고 |
|---|---|---|
| Neural-ART NPU 가속 | **미지원** | GRU/LSTM 연산자 없음 |
| ST Edge AI Core 변환 | **지원** | Keras stateful LSTM/GRU (initial support) |
| 실행 위치 | **Cortex-M55 CPU** | fallback |
| 지원 배치 크기 | `batch_size=1` 만 | 추론 시 단일 샘플 |
| `return_state` | 미지원 | — |
| `return_sequences` | 지원 (마지막 레이어 제외 시) | 일반적 스택 GRU 구성 가능 |
| INT8 실행 | CPU에서 int8 또는 float32 fallback | 플랫폼에 따라 다름 |

> 출처: [Keras stateful LSTM/GRU support](https://stedgeai-dc.st.com/assets/embedded-docs/keras_lstm_stateful.html),  
> [GRU layer support on ST Edge AI (Community)](https://community.st.com/t5/edge-ai/gru-layer-support-on-st-edge-ai/td-p/803047)

---

## 4. 우리 GRU 모델의 연산자별 실행 위치

현재 모델 구조: `Conv1D(causal) × 2 → GRU(256) → GRU(128) → [TemporalAttention] → Dense → Softmax`

| 레이어 | TFLite 연산자 | 실행 위치 | 비고 |
|---|---|---|---|
| Conv1D (causal) | `CONV_2D` | **NPU** | TFLite에서 CONV_2D로 변환됨 |
| BatchNormalization | (CONV에 fold) | **NPU** | 양자화 시 Conv에 흡수 |
| GRU (256, 128) | GRU (미지원) | **CPU** | Cortex-M55 fallback |
| LayerNormalization | 분해 연산 | **CPU** | GRU 블록 내 |
| TemporalAttention | `FULLY_CONNECTED` + element-wise | **NPU/CPU 혼합** | Dense는 NPU, reduce_sum 등은 CPU |
| Dense (head) | `FULLY_CONNECTED` | **NPU** | |
| Softmax | `SOFTMAX` (or LOGISTIC) | CPU 가능성 | 미지원 목록에 SOFTMAX 없음 |

---

## 5. 배포 가능성 평가

### 결론: 배포 가능, GRU는 CPU 실행이 정상 시나리오

- **변환 가능**: ST Edge AI Core가 Keras GRU → TFLite INT8 변환 지원 (initial support)
- **NPU 미가속**: GRU 레이어가 CPU에서 실행되는 것은 **예상 범위 내**
- **지연 시간**: 15fps 윈도우 기준 추론 허용 시간 ~67ms. Cortex-M55 400MHz에서 INT8 GRU(256+128 units, 60 steps) 실행 가능

### 실험 전략에 미치는 영향

| 항목 | 내용 |
|---|---|
| 현재 GRU 실험 방향 | 변경 없음 — NPU 미가속이어도 배포 목표 유지 |
| GRU units 크기 중요성 | CPU에서 직접 실행되므로 `256,128` vs `128,64` ablation이 latency 관점에서 실용적 의미 있음 |
| TCN 비교 실험 의의 | TCN은 CONV_2D 기반 → NPU 풀 가속 → latency 측면 유리. 동일 정확도라면 TCN이 배포 효율 우위 |
| TemporalAttention | Dense 기반 구현이므로 NPU 일부 활용 가능. GRU 대비 추가 overhead 미미 |
| INT8 목표 (F1 0.9x) | CPU fallback에서도 INT8 실행 가능, 정확도 목표 유지 |

---

## 6. 참고 링크

- [ST Neural-ART NPU - Supported operators](https://stm32ai-cs.st.com/assets/embedded-docs/stneuralart_operator_support.html)
- [Keras stateful LSTM/GRU support](https://stedgeai-dc.st.com/assets/embedded-docs/keras_lstm_stateful.html)
- [ST Edge AI Core Technology Documentation](https://stedgeai-dc.st.com/assets/embedded-docs/index.html)
- [Keras toolbox support](https://stedgeai-dc.st.com/assets/embedded-docs/supported_ops_keras.html)
- [TFLite toolbox support](https://stedgeai-dc.st.com/assets/embedded-docs/supported_ops_tflite.html)
- [GRU layer support on ST Edge AI (Community)](https://community.st.com/t5/edge-ai/gru-layer-support-on-st-edge-ai/td-p/803047)
