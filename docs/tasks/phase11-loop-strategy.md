# Phase 11+ 자율 루프 전략

## 핵심 원칙

- **같은 전략 내 파라미터 변형** → 같은 Phase, v-number 증가 (P11-v01, v02, ...)
- **접근법 자체가 새로울 때** → Phase 번호 증가 (Phase 12, 13, ...)
- **각 Phase의 results 디렉토리에 `strategy.md` 작성** (아래 형식)

---

## strategy.md 형식 (results/<phase-dir>/strategy.md)

```markdown
# Phase N Strategy

**핵심 가설**: 왜 이 접근이 효과적일 것이라 생각하는가

**이전 Phase 실패 원인**: (Phase N-1에서 무엇이 부족했는가)

**주요 변경점**:
- 변경 1
- 변경 2

**후보 실험**:
| ID | 변경점 |
|----|--------|
| PN-v01 | ... |

**종료 기준**: test MinPR ≥ 0.92
```

---

## 루프 절차 (매 이터레이션)

1. `screen -list` — 프로세스 생존 확인
2. `tail -40 <현재 phase log>` — 진행 상태 파악
3. 완료된 실험 `metrics.json` 분석
4. 판단:
   - 학습 진행 중 → 대기 (45~60분)
   - **목표 달성** → 루프 종료, 결과 요약
   - runner 종료 + **미달 + 후보 남음** → 병목 분석 후 runner 재시작 (same Phase)
   - runner 종료 + **미달 + 후보 소진** → 다음 Phase 설계 후 실행

---

## Phase 이력 및 계획

### Phase 11 (현재)
- **디렉토리**: `results/phase11_minpr_hardneg/`
- **핵심 가설**: val_video_min_pr 체크포인팅 + val FP 영상 hard-negative 오버샘플링으로 NFall 방어
- **후보**: v01(체크포인팅만) → v02(+hard-neg) → v03(+sweep) → v04(+focal α) → v05(+neg stride) → v06(+kp12) → v07(+256,128)

### Phase 12 (Phase 11 전체 실패 시)
아이디어 후보 (Phase 11 결과 보고 확정):
- **데이터 (LB-3 활용)**: 3-class 학습(0/1/2)으로 falling(1) vs fallen(2) 분리
  - ⚠️ 목표는 falling(1) 검출. fallen(2)은 이미 넘어진 상태라 독립 검출기로 의미 없음
  - 활용 방향: 3-class 학습 후 positive_labels=[1,2]로 평가 — 학습에서 label 분리해 falling 패턴 더 명시적으로 학습
  - 또는 label 2 구간을 non-fall처럼 취급해 hard-negative 역할로 활용 (검토 필요)
- **손실**: NFall recall 보조 손실
- **데이터**: hard-negative windows repeat 삽입 (repeat=3)
- **모델**: TCN (temporal feature 더 명시적)

---

## 결과 확인 명령어

```bash
# 완료된 실험 요약
python3 - results/phase11_minpr_hardneg 0.92 <<'EOF'
import json, sys
from pathlib import Path
outroot = Path(sys.argv[1]); target = float(sys.argv[2])
print(f"{'ID':<12} {'test_MinPR':>10} {'NFallP':>8} {'NFallR':>8} {'val_MinPR':>10}")
for d in sorted(outroot.iterdir()):
    mj = d / "metrics.json"
    if not mj.exists(): continue
    m = json.loads(mj.read_text())
    tv = m.get("metrics", {}).get("test_video", {})
    vv = m.get("metrics", {}).get("val_video", {})
    flag = " PASS" if tv.get("min_pr", 0) >= target else ""
    print(f"{d.name:<12} {tv.get('min_pr', float('nan')):>10.4f} {tv.get('nfall_precision', float('nan')):>8.4f} {tv.get('nfall_recall', float('nan')):>8.4f} {vv.get('min_pr', float('nan')):>10.4f}{flag}")
EOF

tail -30 results/phase11_minpr_hardneg_main.log
screen -list
```

## screen 재시작

```bash
screen -S phase11 -X quit
screen -dmS phase11 bash -c 'cd /home/min/Workspace/Graduate-Project/Falling-Model-Development && _SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])") && export LD_LIBRARY_PATH="$(find ${_SITE}/nvidia -maxdepth 2 -name lib -type d | tr "\n" ":"):/usr/local/cuda/lib64" && bash scripts/run_phase11_minpr_hardneg.sh 2>&1 | tee results/phase11_minpr_hardneg_main.log; echo "=== DONE at $(date) ===" | tee -a results/phase11_minpr_hardneg_main.log'
```
