# Colab Training Analysis and GCN Adoption Plan

작성일: 2026-05-13

## 1. 분석 범위

이 문서는 `colab/*.ipynb`에 저장된 출력, `artifacts/gru_v26_final_notebook/run_metadata.json`, `model/run_metadata(1).json`, 그리고 현재 학습 스크립트 구조를 기준으로 낙상 감지 모델의 학습 현황과 성능 개선 방향을 정리한다.

현재 프로젝트의 주 입력은 MoveNet 계열 17개 관절의 좌표/신뢰도(`kp0_y`, `kp0_x`, `kp0_s` ... `kp16_*`)와 파생 특징(`HSSC_y`, `HSSC_x`, `RWHC`, `VHSSC`, 일부 노트북의 `AHSSC`)이다. 대부분의 실험은 길이 60 또는 100의 sliding window를 사용하고, `video_id` 기반 group split으로 데이터 누수를 줄이는 구조다.

## 2. Colab 학습 결과 요약

| 실험 | 클래스 | 핵심 설정 | 저장 출력 기준 결과 | 해석 |
| --- | --- | --- | --- | --- |
| `gru_v26_training_final.ipynb` | 2-class | 60 step, GRU 64/32, threshold 0.8, positive weight 1.5 | Test accuracy 0.8808, precision 0.8714, recall 0.8334, macro F1 0.8761 | precision은 괜찮지만 threshold 0.8에서 recall 손실이 크다. |
| `gru_v26_training.ipynb` | 2-class | 60 step, GRU 64/32, threshold 0.7, positive weight 0.7 | Test accuracy 0.8827, precision 0.8617, recall 0.8517, macro F1 0.8787 | v26 final보다 recall은 개선되지만 전체 성능 차이는 작다. |
| `gru_v27_2class_training-1.ipynb` | 2-class | 60 step, GRU 64/32 + BatchNorm, self-labeling, threshold 0.55, min consecutive 5 | raw test accuracy 0.90, post-processed test accuracy 0.92, macro F1 0.90, Fall F1 0.86 | 저장 출력 기준 최고 성능. threshold와 연속 프레임 규칙이 큰 효과를 냈다. |
| `tcn_v28_training.ipynb` | binary eval, near-fall mining 포함 | 100 step, dilated TCN, hard negative/near-fall mining | Test accuracy 0.8935, precision 0.8926, recall 0.9550, macro F1 0.8757 | recall은 매우 높지만 precision/오탐 부담이 남는다. 학습 accuracy와 validation loss 간 격차가 커 과적합 신호가 있다. |
| `model/run_metadata(1).json` | 2-class | `final_dataset_1Euro.csv` | Test accuracy 0.4216, recall 0.0, predicted positive rate 0.0 | 정상 동작하지 않은 산출물로 봐야 한다. normalization mean/std에 NaN이 포함되어 있어 입력 결측 처리나 전처리 저장 과정 점검이 필요하다. |

저장된 출력이 없는 노트북도 일부 있다. `cnn_gru_v1_2class_training.ipynb`, `lstm_v1_2class_training.ipynb`, `rnn_v1_2class_training.ipynb`, `tcn_v29_class2_training.ipynb`, `tcn_v29_class3_training.ipynb`, `gru_v27_3class_training-1.ipynb`는 코드 구조는 확인됐지만 신뢰할 만한 최종 출력이 저장되어 있지 않아 위 표의 정량 비교에서는 제외했다.

## 3. 현재 학습 흐름의 강점

1. **입력 특징이 실시간 배포에 적합하다.** 17개 관절 좌표와 신뢰도, 상체 중심, 세로 속도, bbox 비율 중심이라 카메라 기반 실시간 낙상 감지에 필요한 정보가 압축되어 있다.
2. **`video_id` 기준 split을 사용한다.** 프레임 또는 window 단위 무작위 split보다 누수 가능성이 낮다.
3. **threshold sweep과 min-consecutive 후처리가 이미 효과를 보였다.** `gru_v27_2class_training-1.ipynb`에서 raw test accuracy 0.90이 후처리 후 0.92로 상승했다.
4. **TCN 계열은 recall 확보에 강하다.** `tcn_v28_training.ipynb`는 test recall 0.9550을 기록해 놓치는 낙상을 줄이는 방향의 baseline으로 가치가 있다.

## 4. 주요 문제점

1. **성능 비교 기준이 완전히 통일되어 있지 않다.** 노트북마다 `target_steps`, label mode, positive stride, negative stride, threshold, min consecutive, class weight가 다르다. 따라서 accuracy만 보고 모델 우열을 판단하기 어렵다.
2. **threshold가 모델 성능을 크게 좌우한다.** v26은 threshold 0.7/0.8 변화만으로 recall이 0.8517에서 0.8334로 내려간다. 최종 모델 선택은 반드시 validation 기반 threshold 탐색과 test 고정 평가를 분리해야 한다.
3. **일부 전처리 산출물에 NaN 문제가 있다.** `model/run_metadata(1).json`의 normalization mean/std에 NaN이 포함되고, 모든 샘플을 normal로 예측했다. `final_dataset_1Euro.csv` 생성 후 좌표/파생 특징 결측을 검사해야 한다.
4. **3-class 실험은 아직 정량 근거가 부족하다.** 3-class 노트북은 코드상 `label_3class`를 사용하지만 최종 classification report가 저장되어 있지 않다. `Falling`과 `Fallen` 분리가 오탐 감소에 도움이 되는지 아직 결론을 내리기 어렵다.
5. **window 단위 지표만으로 제품 품질을 판단하기 어렵다.** 실제 낙상 감지는 video/event 단위 탐지 지연, false alarm per hour, 낙상 시작 전후 검출 시점이 중요하다.

## 5. 성능 개선 방향

### 5.1 즉시 우선순위

1. **GRU v27 2-class를 현재 기준 모델로 고정한다.**
   - 저장 출력 기준 test accuracy 0.92, macro F1 0.90으로 가장 안정적이다.
   - 같은 split, 같은 dataset에서 v26/v27/TCN을 다시 평가해 baseline table을 고정한다.

2. **평가 코드를 통합한다.**
   - 모델별 공통 지표: accuracy, precision, recall, macro F1, specificity, false positive rate, false negative rate, MCC, PR-AUC.
   - 운영 지표: false positives per hour, event-level recall, median detection delay.
   - threshold/min-consecutive는 validation에서만 선택하고, test에서는 선택된 값을 고정한다.

3. **NaN/결측 방어를 전처리 파이프라인에 넣는다.**
   - `kp*_x`, `kp*_y`, `HSSC_*`, `RWHC`, `VHSSC`, `AHSSC`에 대해 split 전 `isna()` 카운트와 video별 결측 비율을 저장한다.
   - One Euro Filter 적용 후 좌표가 전부 NaN이 되는 video가 있는지 확인한다.
   - 결측 관절은 무조건 0으로 채우기보다 confidence score와 함께 mask/presence feature를 유지하는 편이 낫다.

4. **후처리를 모델 선택 과정에 포함한다.**
   - `threshold ∈ [0.05, 0.95]`, `min_consecutive ∈ [1, 5]` grid는 이미 효과가 입증됐다.
   - 낙상 경보 목적이면 recall 하한을 먼저 정하고 macro F1 또는 false positive rate로 tie-break한다.

### 5.2 모델/데이터 실험

1. **3-class를 이벤트 상태 추적용으로 재평가한다.**
   - `Normal`, `Falling`, `Fallen`을 직접 분류한 뒤 운영 알람은 `Falling` 또는 `Falling+Fallen` 확률로 만든다.
   - 기대 효과는 `Fallen` 정지 상태와 일상 정지 자세를 구분해 오탐을 줄이는 것이다.

2. **hard negative mining을 GRU v27에도 적용한다.**
   - TCN v28의 높은 recall은 hard negative/near-fall mining의 영향일 수 있다.
   - false positive가 많이 나는 정상 video 구간을 validation에서 수집해 다음 학습의 normal-hard class 또는 sample weight로 반영한다.

3. **입력 특징을 보강한다.**
   - 현재 v26 metadata는 `AHSSC`가 빠져 있고, 일부 노트북은 `AHSSC`를 포함한다. 상체 중심의 가속도는 낙상 순간 구분에 유용하므로 공통 feature set에 포함 여부를 실험한다.
   - 신뢰도 기반 관절 mask, shoulder/hip angle, body tilt, bbox height velocity를 추가 후보로 둔다.

4. **경량 배포 제약을 별도 평가한다.**
   - Keras 성능뿐 아니라 TFLite 변환 성공 여부, int8 quantization 후 성능, STM32N6 추론 latency와 peak memory를 함께 비교한다.

## 6. GCN 도입 가능성

결론부터 말하면 **도입 가능성이 높고, vanilla GCN보다 ST-GCN 계열이 현재 문제에 더 적합하다.**

현재 입력은 17개 관절이 있는 skeleton sequence다. 일반 GRU/TCN은 17개 관절을 feature vector로 펼쳐서 처리하므로 관절 간 물리적 연결 구조를 명시적으로 알지 못한다. 반면 GCN/ST-GCN은 관절을 node, 뼈대 연결을 edge로 두고 frame 내부의 공간 관계와 frame 간 시간 관계를 함께 학습한다.

참고 문헌상 GCN은 graph-structured data에 직접 동작하는 신경망으로 제안되었고, ST-GCN은 skeleton sequence의 spatial/temporal pattern을 함께 학습하기 위해 제안되었다. 낙상 감지에서도 skeleton과 ST-GCN을 결합해 관절 의존성을 활용하려는 연구가 있다.

### 6.1 현재 데이터로 구성 가능한 그래프

MoveNet/COCO 17 keypoint 기준으로 다음 edge를 기본 skeleton topology로 둘 수 있다.

| 부위 | edge |
| --- | --- |
| 얼굴/어깨 | nose-eye, eye-ear, shoulder-shoulder |
| 팔 | shoulder-elbow, elbow-wrist |
| 몸통 | shoulder-hip, hip-hip |
| 다리 | hip-knee, knee-ankle |

각 frame의 node feature는 최소 `[x, y, score]`이고, 추가로 관절별 velocity를 붙이면 `[x, y, score, dx, dy]`가 된다. `HSSC`, `RWHC`, `VHSSC`, `AHSSC`는 graph pooling 이후 global feature로 concat하는 방식이 자연스럽다.

### 6.2 권장 아키텍처

첫 실험은 다음 순서가 현실적이다.

1. **ST-GCN-lite**
   - 입력 shape: `(batch, time=60, joints=17, channels=3~5)`
   - spatial graph conv: 고정 adjacency matrix 기반 `A @ X @ W`
   - temporal conv: `Conv1D` 또는 depthwise temporal conv
   - global average pooling 후 dense classifier

2. **GCN + GRU hybrid**
   - 각 frame에서 1~2 layer GCN으로 관절 embedding 생성
   - frame별 embedding을 pooling해 `(batch, time, hidden)`으로 변환
   - 기존 GRU v27 head를 재사용
   - 장점: 현재 GRU 평가/후처리 코드와 가장 잘 연결된다.

3. **3-class ST-GCN**
   - 2-class에서 성능이 확인된 뒤 `Normal/Falling/Fallen`으로 확장한다.
   - `Falling` recall과 `Normal -> Falling` 오탐을 핵심 지표로 본다.

### 6.3 배포 관점 주의사항

PyTorch Geometric, DGL 같은 외부 GNN 라이브러리는 연구 실험에는 편하지만 TFLite/STM32 배포에는 불리하다. 배포 후보는 Keras/TensorFlow 기본 연산만 사용해 구현하는 것이 좋다.

권장 구현 방식은 다음과 같다.

```python
# 개념 예시: batch/time/joint/channel 입력에 고정 adjacency를 적용
x = tf.einsum("ij,btjc->btic", adjacency, x)
x = tf.keras.layers.Dense(hidden)(x)
```

다만 `einsum`이 TFLite/ST Edge AI에서 항상 원하는 형태로 내려간다고 가정하면 안 된다. 실제 배포 후보는 `tf.matmul` 또는 reshape + `Dense`/`Conv2D` 기반으로 변환 가능성을 먼저 확인해야 한다.

## 7. 제안 실험 로드맵

1. **Baseline 재현**
   - `gru_v27_2class_training-1.ipynb` 설정을 스크립트화한다.
   - 동일 split에서 threshold/min-consecutive sweep 결과와 test report를 JSON으로 저장한다.

2. **데이터 검증**
   - `final_dataset.csv`, `final_dataset_1Euro.csv`, `master_3class.sqlite`에 대해 feature별 NaN/Inf 리포트를 만든다.
   - 결측이 있는 video_id와 class 분포를 확인한다.

3. **ST-GCN-lite prototype**
   - 2-class, 60 step, 동일 split, 동일 후처리로 GRU v27과 비교한다.
   - 목표: GRU v27 macro F1 0.90 이상 또는 같은 recall에서 false positive rate 감소.

4. **GCN + GRU hybrid**
   - ST-GCN-lite가 비슷한 성능을 보이면 GRU hybrid로 경량화와 안정성을 확인한다.
   - 목표: GRU v27 대비 parameter/FLOPs 증가를 제한하면서 precision 개선.

5. **배포 검증**
   - Keras -> TFLite fp32 -> TFLite int8 순서로 변환한다.
   - 변환 후 동일 test set에서 metric drift를 측정한다.
   - STM32N6 도구에서 unsupported op 여부와 latency를 확인한다.

## 8. 다음 작업 제안

1. `scripts/evaluate_training_outputs.py` 형태의 공통 평가 스크립트를 추가한다.
2. `scripts/check_dataset_quality.py`로 NaN/Inf/결측 관절/label 분포를 자동 리포트한다.
3. `scripts/train_stgcn_lite.py`를 추가해 Keras 기반 ST-GCN-lite를 먼저 실험한다.
4. GRU v27, TCN v28, ST-GCN-lite를 같은 표로 비교하는 `docs/model-comparison.md`를 생성한다.

## 9. 참고 자료

- Kipf, Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017: https://arxiv.org/abs/1609.02907
- Yan, Xiong, Lin, "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition", AAAI 2018: https://arxiv.org/abs/1801.07455
- Sensors 2023, "Skeleton-Based Fall Detection with Multiple Inertial Sensors Using Spatial-Temporal Graph Convolutional Networks": https://www.mdpi.com/1424-8220/23/4/2153
