---
theme: seriph
title: Falling Model Development Progress
info: |
  ## Falling Model Development
  중간 진행 상황 발표
class: text-center
drawings:
  persist: false
transition: slide-left
mdc: true
duration: 12min
fonts:
  sans: Noto Sans KR
  serif: Noto Serif KR
  mono: JetBrains Mono
---

<style>
:root {
  --slidev-theme-primary: #0f766e;
  --slidev-theme-accent: #b45309;
  --fm-ink: #0f172a;
  --fm-muted: #475569;
  --fm-line: #cbd5e1;
  --fm-soft: #f8fafc;
}

.slidev-layout {
  color: var(--fm-ink);
}

h1 {
  color: #0f172a;
  font-weight: 800;
  letter-spacing: -0.02em;
}

h2, h3 {
  color: #134e4a;
}

p, li, td, th {
  color: var(--fm-ink);
}

.muted {
  color: var(--fm-muted);
}

.hero {
  background:
    radial-gradient(circle at top right, rgba(15, 118, 110, 0.18), transparent 32%),
    radial-gradient(circle at bottom left, rgba(180, 83, 9, 0.12), transparent 30%),
    linear-gradient(135deg, #f8fafc 0%, #eff6ff 45%, #fefce8 100%);
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 18px;
}

.card {
  border: 1px solid var(--fm-line);
  border-radius: 18px;
  padding: 16px 18px;
  background: rgba(255, 255, 255, 0.78);
  box-shadow: 0 8px 30px rgba(15, 23, 42, 0.05);
}

.kpi {
  font-size: 1.8rem;
  font-weight: 800;
  line-height: 1.1;
  color: #134e4a;
}

.kpi-label {
  font-size: 0.95rem;
  color: var(--fm-muted);
}

.mini {
  font-size: 0.88rem;
  color: var(--fm-muted);
}

.section-title {
  font-size: 0.95rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: #0f766e;
  margin-bottom: 8px;
}

.timeline {
  display: grid;
  gap: 10px;
  margin-top: 16px;
}

.timeline-item {
  display: grid;
  grid-template-columns: 130px 1fr;
  gap: 14px;
  align-items: start;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(203, 213, 225, 0.7);
}

.timeline-date {
  font-family: "JetBrains Mono", monospace;
  color: #0f766e;
  font-size: 0.92rem;
}

.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 700;
  background: #ccfbf1;
  color: #115e59;
}

.warn {
  background: #fef3c7;
  color: #92400e;
}

.danger {
  background: #fee2e2;
  color: #991b1b;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
}

th, td {
  border-bottom: 1px solid rgba(203, 213, 225, 0.7);
  padding: 10px 12px;
  vertical-align: top;
}

th {
  text-align: left;
  color: #134e4a;
}
</style>

---
layout: cover
class: hero
---

# 낙상 감지 모델 개발 진행 상황

STM32N6 배포를 목표로 한 시계열 분류 모델 개발 보고

<div class="mt-10 text-lg muted">
Graduate Project · Falling Model Development
</div>

<div class="mt-18 grid grid-cols-3 gap-4 text-left">
  <div class="card">
    <div class="section-title">Target</div>
    <div class="kpi">STM32N6</div>
    <div class="kpi-label">NPU 배포 가능한 구조 우선</div>
  </div>
  <div class="card">
    <div class="section-title">Current Focus</div>
    <div class="kpi">TCN / GRU</div>
    <div class="kpi-label">v2, v2.5 학습 파이프라인 구축</div>
  </div>
  <div class="card">
    <div class="section-title">This Talk</div>
    <div class="kpi">개발 현황</div>
    <div class="kpi-label">완료 사항, 구조 변화, 다음 단계</div>
  </div>
</div>

---

# 발표 개요

1. 프로젝트 목표와 제약
2. 데이터와 입력 정의
3. 현재까지 구현한 개발 내용
4. 모델 버전별 변화
5. 배포 관점에서의 정리
6. 남은 과제와 다음 단계

---

# 프로젝트 목표

<div class="card-grid">
  <div class="card">
    <div class="section-title">문제</div>
    <div class="text-xl font-bold mb-2">낙상 구간을 안정적으로 탐지해야 함</div>
    <div class="mini">단순 오프라인 정확도보다 실제 영상에서의 탐지 성능과 오경보 억제가 중요</div>
  </div>
  <div class="card">
    <div class="section-title">배포 조건</div>
    <div class="text-xl font-bold mb-2">STM32N6 / ST Edge AI 제약 고려</div>
    <div class="mini">정적 입력, int8 친화 구조, NPU 지원 연산 중심 설계 필요</div>
  </div>
  <div class="card">
    <div class="section-title">개발 방향</div>
    <div class="text-xl font-bold mb-2">TCN을 주 모델로, GRU를 비교 기준으로 유지</div>
    <div class="mini">정확도 비교는 하되 배포 용이성을 함께 평가</div>
  </div>
  <div class="card">
    <div class="section-title">이번 발표 범위</div>
    <div class="text-xl font-bold mb-2">모델 성능 확정보다 개발 진행상황 보고</div>
    <div class="mini">파이프라인 정비, 모델 버전업, 노트북 실험 기반 구축 중심</div>
  </div>
</div>

---

# 데이터와 입력 정의

<div class="grid grid-cols-2 gap-6 mt-6">
  <div class="card">
    <div class="section-title">현재 데이터 이해</div>
    <div class="kpi">4,093,620 rows</div>
    <div class="kpi-label">20,380개 영상, 59개 컬럼</div>
    <ul class="mt-4">
      <li>포즈 기반 입력: `kp0 ~ kp16`의 `x / y / score`</li>
      <li>추가 특징: `HSSC_y`, `HSSC_x`, `RWHC`, `VHSSC`</li>
      <li>최종 입력 차원: `55` features</li>
    </ul>
  </div>
  <div class="card">
    <div class="section-title">시계열 입력 표준화</div>
    <div class="kpi">60 x 55</div>
    <div class="kpi-label">고정 길이 시퀀스 입력</div>
    <ul class="mt-4">
      <li>초기 v2: `5s ~ 9s` 구간을 리샘플링한 clip 기반 학습</li>
      <li>현재 v2.5: `4s ~ 10s` 모니터링 구간에서 sliding window 생성</li>
      <li>모델 비교를 위해 train/val/test는 `video_id` 기준 분리</li>
    </ul>
  </div>
</div>

---

# 개발 진행 타임라인

<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-date">초기 단계</div>
    <div>
      <div class="font-bold">기본 학습 파이프라인 정리</div>
      <div class="mini">CSV 또는 SQLite에서 입력을 읽고, 정규화, 그룹 분할, 학습/평가/저장을 한 번에 수행하는 구조 구축</div>
    </div>
  </div>
  <div class="timeline-item">
    <div class="timeline-date">v2</div>
    <div>
      <div class="font-bold">TCN v2 / GRU v2 구현</div>
      <div class="mini">clip 단위 학습, 클래스 가중치 적용, TFLite export 옵션, 주요 분류 지표 저장 기능 추가</div>
    </div>
  </div>
  <div class="timeline-item">
    <div class="timeline-date">v2.5</div>
    <div>
      <div class="font-bold">sliding-window supervision 도입</div>
      <div class="mini">모니터링 구간 내부에서 여러 윈도우를 생성하고, `last_frame` 또는 `segment_max` 기준으로 라벨링</div>
    </div>
  </div>
  <div class="timeline-item">
    <div class="timeline-date">최근</div>
    <div>
      <div class="font-bold">Colab 실험 노트북 및 threshold sweep 추가</div>
      <div class="mini">실험 재현성과 후처리 임계값 탐색을 위한 노트북 기반 실험 환경까지 연결</div>
    </div>
  </div>
</div>

---

# 현재 구현된 학습 파이프라인

| 구성 요소 | 구현 내용 | 의미 |
| --- | --- | --- |
| 입력 소스 | `CSV`, `SQLite` 모두 지원 | 대용량 데이터 처리 경로를 유연하게 유지 |
| 데이터 분할 | `GroupShuffleSplit`으로 `video_id` 단위 분리 | 같은 영상이 train/test에 동시에 들어가는 문제 방지 |
| 정규화 | train split 기준 평균/표준편차 계산 | 평가 데이터 누수 방지 |
| 불균형 대응 | class weight 적용 | 양성 구간 희소성 대응 |
| 산출물 저장 | `.keras`, 메타데이터, 시각화 파일 저장 | 실험 비교와 추적 가능 |
| 배포 준비 | optional TFLite export | STM32 배포 흐름으로 이어질 수 있는 기반 확보 |

---

# 모델 버전별 변화

| 버전 | 핵심 아이디어 | 현재 의미 |
| --- | --- | --- |
| `TCN v2` | 고정 clip 입력에 대한 deployment-oriented baseline | NPU 친화 연산 중심 기준 모델 |
| `GRU v2` | 동일 입력 조건에서의 recurrent baseline | 정확도 비교용 기준선 |
| `TCN v2.5` | 모니터링 구간 내 sliding window 학습 | 실제 낙상 시점을 더 세밀하게 반영하려는 개선 |
| `GRU v2.5` | 같은 supervision 방식으로 GRU 비교 | 구조 차이보다 supervision 차이를 분리해 보기 위함 |

<div class="mt-6 grid grid-cols-3 gap-4">
  <div class="card">
    <div class="badge">TCN 우선</div>
    <p class="mt-3 mini">`Conv1D`, `BatchNorm`, `Add`, `Dense` 위주의 구조는 ST 문서 기준으로 배포 적합성이 높음</p>
  </div>
  <div class="card">
    <div class="badge warn">GRU 비교군</div>
    <p class="mt-3 mini">GRU는 성능 비교에는 유효하지만 NPU 직접 매핑 관점에서는 제약이 더 큼</p>
  </div>
  <div class="card">
    <div class="badge">v2.5 핵심</div>
    <p class="mt-3 mini">clip 단일 라벨보다 window 단위 supervision이 실제 탐지 목적에 더 가까운 방향</p>
  </div>
</div>

---

# 왜 v2.5로 넘어갔는가

<div class="grid grid-cols-2 gap-6 mt-6">
  <div class="card">
    <div class="section-title">기존 v2 한계</div>
    <ul>
      <li>고정 구간 전체를 하나의 clip으로 처리해 시점 정보가 거칠게 반영됨</li>
      <li>낙상 시작 직전과 직후의 변화를 구분해 학습시키기 어려움</li>
      <li>실제 탐지는 특정 시점 판단인데, 학습은 clip 전체 라벨에 묶여 있었음</li>
    </ul>
  </div>
  <div class="card">
    <div class="section-title">v2.5 개선점</div>
    <ul>
      <li>모니터링 구간에서 `60-step` window를 다수 생성</li>
      <li>positive / negative stride를 분리해 샘플링 조절</li>
      <li>threshold sweep과 결합해 운영 임계값 탐색이 가능해짐</li>
    </ul>
  </div>
</div>

---

# 현재 저장소 기준 산출물

<div class="card-grid">
  <div class="card">
    <div class="section-title">학습 스크립트</div>
    <ul>
      <li>`scripts/train_tcn_v2.py`</li>
      <li>`scripts/train_gru_v2.py`</li>
      <li>`scripts/train_tcn_v25.py`</li>
      <li>`scripts/train_gru_v25.py`</li>
    </ul>
  </div>
  <div class="card">
    <div class="section-title">실험 노트북</div>
    <ul>
      <li>`colab/tcn_v2_training.ipynb`</li>
      <li>`colab/gru_v2_training.ipynb`</li>
      <li>`colab/tcn_v25_training.ipynb`</li>
      <li>`colab/gru_v25_training.ipynb`</li>
    </ul>
  </div>
  <div class="card">
    <div class="section-title">배포 흐름 준비</div>
    <ul>
      <li>TFLite export 함수 구현</li>
      <li>STM32N6 전략 문서 정리</li>
      <li>연산 제약 기반 설계 원칙 수립</li>
    </ul>
  </div>
  <div class="card">
    <div class="section-title">현재 확인된 artifact</div>
    <ul>
      <li>`artifacts/tcn_v25_smoke/tcn_v25.keras`</li>
      <li class="mini">즉, smoke 수준 저장 결과는 존재</li>
      <li class="mini">최종 비교용 실험 결과 표는 아직 별도 정리 필요</li>
    </ul>
  </div>
</div>

---

# 배포 관점에서의 현재 판단

<div class="grid grid-cols-3 gap-4 mt-6">
  <div class="card">
    <div class="badge">확정</div>
    <div class="text-lg font-bold mt-3">TCN 중심 전략 유지</div>
    <p class="mini mt-2">STM32N6 NPU 지원 연산과 정적 입력 구조를 고려하면 현재까지 가장 현실적인 주 경로</p>
  </div>
  <div class="card">
    <div class="badge warn">진행 중</div>
    <div class="text-lg font-bold mt-3">정확도와 threshold 운영값 탐색</div>
    <p class="mini mt-2">단순 accuracy가 아니라 recall, false alarm, 운영 임계값을 함께 봐야 함</p>
  </div>
  <div class="card">
    <div class="badge danger">미완료</div>
    <div class="text-lg font-bold mt-3">실제 STM32N6 정량 검증</div>
    <p class="mini mt-2">int8 양자화 후 정확도 유지, 지연시간, 메모리 사용량은 이후 단계에서 확인 필요</p>
  </div>
</div>

---

# 남은 과제

1. v2와 v2.5 결과를 같은 기준으로 정리한 비교표 작성
2. threshold sweep 결과를 바탕으로 운영 임계값 후보 도출
3. TCN float 성능이 충분한지 확인 후 양자화 실험 진행
4. TFLite 또는 ONNX 경로에서 ST Edge AI 도구체인 적합성 점검
5. STM32N6 탑재 전 latency, memory, false alarm까지 포함한 시스템 평가 수행

---
layout: center
class: hero
---

# 결론

현재까지의 개발은 "모델 하나를 학습했다" 수준이 아니라,
배포 가능한 낙상 감지 모델을 만들기 위한 **실험 파이프라인과 비교 구조**를 만든 단계입니다.

<div class="mt-8 text-xl font-bold">
핵심 진전: `TCN / GRU` baseline 구축 → `v2.5 sliding-window` 전환 → 배포 관점 기준 정리
</div>

<div class="mt-10 muted">
다음 발표에서는 성능 비교표와 threshold 결과까지 포함해 정량적으로 정리할 예정
</div>
