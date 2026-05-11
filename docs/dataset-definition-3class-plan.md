# Fall Detection Dataset Definition and 3-Class Plan

## 목적

이 문서는 `STM32N6`용 낙상 감지 모델 개발의 1번 작업인 `데이터셋 기준 고정`을 위해 작성한다.

이번 프로젝트의 목표는 기존 `fall / non-fall` 이진 분류에서 확장하여 아래 `3-class` 기준을 사용하는 것이다.

- `0 = non_fall`
- `1 = almost_fall`
- `2 = fall`

---

## 1. 현재 데이터셋 상태

확인 대상:

- [final_dataset.csv](/home/min/Workspace/Graduate-Project/Falling-Model-Development/dataset/final_dataset.csv)
- [final_dataset.sqlite](/home/min/Workspace/Graduate-Project/Falling-Model-Development/dataset/final_dataset.sqlite)

현재 확인된 사실:

- 총 row 수: `4,093,620`
- 총 video 수: `20,380`
- clip 길이: 대부분 약 `10초`
- clip당 frame 수: `201`
- 현재 컬럼 수: `59`
- 현재 label 컬럼은 `label` 단일 컬럼
- 현재 label 값은 `0`, `1` 두 개만 존재

현재 label 분포:

- `0`: `3,778,880`
- `1`: `314,740`

즉, 현재 데이터셋은 **이진 분류 전용**이며, `almost_fall` 클래스는 아직 데이터셋에 반영되어 있지 않다.

추가 해석:

- 동일 동작이 여러 카메라 각도에서 촬영된 `multi-view clip` 구조로 보인다.
- 따라서 이 데이터셋은 `여러 시점에서 낙상 / 비낙상 검출` 성능을 평가하는 데 적합하다.

---

## 2. 현재 feature schema

현재 feature 구조는 아래와 같다.

- 메타데이터:
  - `video_id`
  - `frame`
  - `time_sec`

- keypoint feature:
  - `kp0_y`, `kp0_x`, `kp0_s`
  - ...
  - `kp16_y`, `kp16_x`, `kp16_s`

- engineered feature:
  - `HSSC_y`
  - `HSSC_x`
  - `RWHC`
  - `VHSSC`

- target:
  - `label`

총 입력 feature 수:

- keypoint 기반 `51`
- engineered `4`
- 합계 `55`

이 구조는 현재 `TCN input = (time, feature) = (60, 55)` 또는 배포 기준 `(1, 55, 60)`으로 사용하기 적합하다.

---

## 3. 이번 프로젝트의 3-class 정의

이번 프로젝트에서는 label 의미를 아래와 같이 고정한다.

### `0 = non_fall`

정의:
- 정상 보행
- 일상 동작
- 앉기, 서기, 방향 전환 등 일반 동작
- 균형 상실 없이 자세가 안정적으로 유지되는 상태

활용:
- `non_fall` clip은 단순 배경 데이터가 아니라 `낙상이 아닌 다양한 정상/유사동작`을 포함하는 hard negative 데이터로 사용한다.
- 즉, 비낙상 데이터에서도 모델은 매 윈도우마다 `fall 여부가 아님`을 검출할 수 있어야 한다.

### `1 = almost_fall`

정의:
- 낙상 직전의 위험 상태
- 중심이 크게 무너지지만 최종적으로 완전한 낙상으로 이어지지 않는 상태
- 급격한 기울어짐, 휘청거림, 미끄러짐 후 회복, 비정상적 하강 후 회복 등이 포함될 수 있음

중요 조건:
- 실제 `fall`과 구별 가능해야 한다.
- 단순한 빠른 동작이나 앉기 동작이 여기에 섞이면 안 된다.

### `2 = fall`

정의:
- 신체 중심이 급격히 하강하고
- 바닥 또는 낮은 자세로 붕괴되며
- 회복 없이 낙상 상태에 도달한 경우

---

## 3.1 멀티뷰와 윈도우 검출 기준

이번 프로젝트는 `영상 클립 전체를 한 번에 1개 라벨로만 보는 방식`보다 아래 구조를 우선한다.

- 여러 각도에서 촬영된 clip을 각각 독립 샘플로 사용
- 각 clip 내부를 `window` 단위로 잘라 낙상 여부를 판단
- 따라서 한 video 전체가 아니라 `시간 구간별 검출`이 가능해야 함

의미:

- 여러 각도별로 `fall / almost_fall / non_fall` 검출 성능을 확인할 수 있다.
- `non_fall` clip에서도 각 window는 명확한 음성 샘플로 사용된다.
- 필요하면 하나의 clip 안에서 초반은 `non_fall`, 중간은 `almost_fall`, 후반은 `fall`처럼 나뉠 수 있다.

즉, 이번 모델은 `clip classification`보다 `window-level temporal classification`에 가깝다.

---

## 4. 핵심 문제

현재 데이터셋에는 `almost_fall` 라벨이 없다.

따라서 지금 상태로는 아래만 가능하다.

- `non_fall vs fall` 이진 분류

지금 상태로는 아래는 불가능하다.

- `non_fall / almost_fall / fall` 3-class 학습

즉, `almost_fall` 클래스를 쓰려면 **라벨 재설계 또는 재생성 단계가 반드시 필요하다.**

---

## 5. 3-class를 만들기 위한 현실적인 방법

우선순위는 아래 순서가 맞다.

### 방법 A. 원본 영상 기준 재라벨링

가장 권장되는 방법이다.

절차:
- 원본 video 단위로 낙상 시점 전후를 다시 확인
- `almost_fall` 구간을 사람이 직접 지정
- frame 또는 segment 단위로 새 라벨 생성

장점:
- 가장 정확하다.
- 논문/보고서에서 근거를 설명하기 쉽다.

단점:
- 시간이 많이 든다.

### 방법 B. 기존 fall 양성 구간을 시간대별로 3단계 분할

원본 onset 메타데이터가 있거나 복구 가능할 때 사용할 수 있다.

예시:
- 안정 구간 -> `non_fall`
- 낙상 직전 불안정 구간 -> `almost_fall`
- 실제 붕괴 및 착지 구간 -> `fall`

장점:
- 자동화 가능성이 있다.

단점:
- onset 정보가 부정확하면 라벨 오염이 심해진다.

### 방법 C. 기존 이진 라벨 기반의 임시 규칙 생성

임시 실험용으로만 가능하다.

예시:
- `VHSSC`, 자세 기울기, hip/shoulder 위치 변화량 등으로 위험 구간을 추정
- 기존 `fall` 양성 근처 프레임 중 일부를 `almost_fall`로 재할당

장점:
- 빠르게 프로토타입 가능

단점:
- 모델이 결국 사람이 만든 규칙을 다시 배우는 구조가 될 수 있다.
- 보고서에서 정당화가 약하다.

결론:
- **최종 보고서용 기준은 A 또는 B가 적합하다.**
- **C는 빠른 실험용 보조 수단으로만 사용한다.**

---

## 6. 프로젝트 기준 권장 데이터 단위

### clip 단위

- 원본 video는 약 `10초`, `201 frame`
- 동일 동작의 여러 각도가 존재할 수 있으므로 카메라 시점별 clip을 각각 하나의 독립 입력으로 취급한다.

### monitoring window 단위

현재 파이프라인과 가장 잘 맞는 기본 입력은 다음과 같다.

- 기본 구간: `5초 ~ 9초`
- 길이: `4초`
- 사용 frame 수: `60`

이 window는 실제 검출 단위다.

즉:

- 여러 각도별로 window 단위 낙상 검출 가능
- 비낙상 clip에서도 모든 window는 음성 검출 샘플로 활용 가능

### label 단위

최종 학습은 아래 둘 중 하나로 고정해야 한다.

- frame label -> sliding window label
- segment label -> clip/window label

3-class 기준에서는 `window label` 정의가 특히 중요하다.

권장 방식:

- 윈도우 내부에 `fall`이 포함되면 `fall`
- `fall`은 없고 `almost_fall`이 포함되면 `almost_fall`
- 나머지는 `non_fall`

우선순위 규칙:

- `fall > almost_fall > non_fall`

실무 해석:

- 전체 clip 라벨보다 `window 라벨`이 우선이다.
- 비낙상 clip은 모든 window가 `non_fall`이어야 한다.
- 낙상 clip은 시간 구간에 따라 `non_fall -> almost_fall -> fall`로 분할될 수 있다.

---

## 7. split 기준

데이터 누수 방지를 위해 split은 반드시 `video_id` 기준으로 해야 한다.

권장 split:

- train: `70%`
- validation: `15%`
- test: `15%`

조건:

- 같은 `video_id`가 여러 split에 동시에 들어가면 안 된다.
- class 비율을 split별로 기록해야 한다.

---

## 8. 정확도 목표 해석

이번 프로젝트의 `95%` 목표는 단순 train accuracy가 아니다.

권장 기준:

- primary: test accuracy `95% 이상`
- 함께 확인:
  - macro F1
  - class-wise precision
  - class-wise recall
  - confusion matrix

특히 `almost_fall` 클래스가 추가되면 accuracy 하나만으로 모델 품질을 평가하면 안 된다.

이유:

- `almost_fall`은 `non_fall`과 경계가 가까워 가장 헷갈리기 쉽다.
- `fall`만 잘 맞고 `almost_fall`을 무시하는 모델은 실제 안전 시스템에서 불완전하다.

---

## 9. 당장 필요한 후속 작업

### 즉시 해야 할 일

1. `almost_fall` 라벨 생성 기준을 확정
2. 원본 데이터 또는 메타데이터에서 재라벨링 가능 여부 확인
3. 새 라벨 컬럼 이름 결정
4. 3-class 분포를 다시 계산
5. 학습 스크립트를 binary -> multiclass 구조로 수정
6. multi-view clip을 split할 때 같은 원본 동작이 train/test에 동시에 섞이지 않는지 확인

### 권장 label 컬럼

- 기존 컬럼 유지: `label`
- 새 컬럼 추가: `label_3class`

권장 매핑:

- `0 = non_fall`
- `1 = almost_fall`
- `2 = fall`

기존 `label`은 호환성 때문에 유지하는 편이 안전하다.

---

## 10. 현재 결론

현재 데이터셋은 `fall / non_fall` 이진 분류 기준으로만 정리되어 있다.

따라서 다음 문장을 현재 기준으로 확정할 수 있다.

- 현재 feature 구조는 `TCN` 학습과 `STM32N6` 배포에 적합하다.
- 현재 라벨 구조는 `3-class` 학습에 적합하지 않다.
- `almost_fall` 클래스를 추가하려면 라벨 재정의가 선행되어야 한다.

즉, `모델 학습`보다 먼저 해야 할 진짜 1번 작업은 다음이다.

- **3-class 라벨 정의 확정**
- **라벨 생성 방식 확정**
- **split 기준 고정**

그리고 그 기준은 아래 문장으로 요약할 수 있다.

- 여러 각도의 영상 clip에서 낙상 / 비낙상을 검출할 수 있어야 한다.
- 비낙상 데이터 역시 `fall이 아님`을 학습시키는 핵심 음성 데이터다.
- 최종 모델은 clip-level보다 window-level 검출 모델로 설계하는 것이 맞다.
