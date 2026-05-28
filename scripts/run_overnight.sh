#!/usr/bin/env bash
# 오버나이트 마스터 스크립트 — Phase 16→17→18→19 순차 실행
# Phase 17은 splits_v2_filtered_kv 재구축 완료 후 실행
# 각 Phase가 MinPR≥0.92 달성하면 즉시 종료

set -uo pipefail

LOGFILE="results/overnight.log"
TARGET_MIN_PR="${TARGET_MIN_PR:-0.92}"
KV_TRAIN="dataset/splits_v2_filtered_kv/train.csv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

_SITE=$(uv run python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -n "$_SITE" ]]; then
    export LD_LIBRARY_PATH="$(find "${_SITE}/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':'):/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p results
log "=== Overnight run started ==="

any_phase_passed() {
    local target="$1"; shift
    for outroot in "$@"; do
        python3 - "$outroot" "$target" <<'PYEOF' 2>/dev/null && return 0
import json, sys
from pathlib import Path
outroot = Path(sys.argv[1])
target = float(sys.argv[2])
if not outroot.exists(): sys.exit(1)
for d in outroot.iterdir():
    mj = d / "metrics.json"
    if not mj.exists(): continue
    m = json.loads(mj.read_text())
    if m.get("skipped"): continue
    tv = m.get("metrics", {}).get("test_video", {})
    if float(tv.get("min_pr", 0.0)) >= target:
        print(f"PASS: {d.name} in {outroot.name}  test_min_pr={tv['min_pr']:.4f}")
        sys.exit(0)
sys.exit(1)
PYEOF
    done
    return 1
}

# ── Phase 16: epochs=30/patience=5, α sweep + multi-seed ────────────────────
log "=== Starting Phase 16 (short epochs α sweep) ==="
bash scripts/run_phase16_short_epochs.sh
if any_phase_passed "$TARGET_MIN_PR" results/phase16_short_epochs; then
    log "=== TARGET REACHED in Phase 16 — stopping ==="
    exit 0
fi
log "Phase 16 complete — MinPR < $TARGET_MIN_PR, continuing to Phase 17"

# ── Phase 17: velocity features (kp7kv) ─────────────────────────────────────
if [[ ! -f "$KV_TRAIN" ]]; then
    log "Building splits_v2_filtered_kv dataset (per-keypoint velocity)..."
    uv run python scripts/util/build_filtered_v2_splits_kv.py 2>&1 | tee -a "$LOGFILE"
    if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
        log "ERROR: dataset build failed — skipping Phase 17"
    else
        log "Dataset build complete"
    fi
fi

if [[ -f "$KV_TRAIN" ]]; then
    log "=== Starting Phase 17 (velocity features kp7kv) ==="
    bash scripts/run_phase17_velocity_features.sh
    if any_phase_passed "$TARGET_MIN_PR" results/phase17_velocity_features; then
        log "=== TARGET REACHED in Phase 17 — stopping ==="
        exit 0
    fi
    log "Phase 17 complete — MinPR < $TARGET_MIN_PR, continuing to Phase 18"
else
    log "SKIP Phase 17 — kv dataset not available"
fi

# ── Phase 18: TCN architecture ───────────────────────────────────────────────
log "=== Starting Phase 18 (TCN architecture) ==="
bash scripts/run_phase18_tcn.sh
if any_phase_passed "$TARGET_MIN_PR" results/phase18_tcn; then
    log "=== TARGET REACHED in Phase 18 — stopping ==="
    exit 0
fi
log "Phase 18 complete — MinPR < $TARGET_MIN_PR, continuing to Phase 19"

# ── Phase 19: GRU(256,128) ───────────────────────────────────────────────────
log "=== Starting Phase 19 (GRU 256,128) ==="
bash scripts/run_phase19_gru256.sh
if any_phase_passed "$TARGET_MIN_PR" results/phase19_gru256; then
    log "=== TARGET REACHED in Phase 19 — stopping ==="
    exit 0
fi
log "Phase 19 complete — MinPR < $TARGET_MIN_PR"

# ── Final summary ────────────────────────────────────────────────────────────
log "=== Overnight run complete ==="
log "=== Best results across all phases ==="
python3 - "$TARGET_MIN_PR" results/phase16_short_epochs results/phase17_velocity_features results/phase18_tcn results/phase19_gru256 <<'PYEOF' 2>/dev/null | tee -a "$LOGFILE"
import json, sys
from pathlib import Path
target = float(sys.argv[1])
roots = [Path(p) for p in sys.argv[2:]]
all_results = []
for root in roots:
    if not root.exists(): continue
    for d in sorted(root.iterdir()):
        mj = d / "metrics.json"
        if not mj.exists(): continue
        m = json.loads(mj.read_text())
        if m.get("skipped"): continue
        tv = m.get("metrics", {}).get("test_video", {})
        thr = m.get("threshold_selection", {})
        minpr = float(tv.get("min_pr", 0.0))
        cm = tv.get("confusion_matrix")
        fp = cm[0][1] if cm else "?"
        fn = cm[1][0] if cm else "?"
        all_results.append((minpr, root.name, d.name, fp, fn, thr.get("threshold"), thr.get("min_consecutive")))

all_results.sort(reverse=True)
for minpr, phase, exp, fp, fn, thr, mc in all_results[:15]:
    flag = "*** PASS ***" if minpr >= target else ""
    print(f"{phase}/{exp:<14} MinPR={minpr:.4f} FP={fp} FN={fn} thr={thr} mc={mc} {flag}")
PYEOF
