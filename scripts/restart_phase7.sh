#!/usr/bin/env bash
# P5-v06 완료 후 Phase 7 재실행용
# Usage: nohup bash scripts/restart_phase7.sh > results/gru_phase7_quant/restart.log 2>&1 &

set -uo pipefail
cd /home/min/Workspace/Graduate-Project/Falling-Model-Development

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for P5-v06 to complete..."
until [[ -f results/gru_phase5_compact/P5-v06/metrics.json ]]; do
    sleep 30
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] P5-v06 done. Launching Phase 7..."

# GPU 해제 대기 (5초)
sleep 5

nohup bash scripts/run_gru_phase7.sh >> results/gru_phase7_quant/nohup.log 2>&1
