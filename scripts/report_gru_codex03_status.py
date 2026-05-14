#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any


EXPERIMENTS = {
    "C3-v01": "30f-kp7",
    "C3-v02": "30f-minimal",
    "C3-v03": "30f-xl-kp12",
    "C3-v04": "60f-kp7",
    "C3-v05": "40f-kp7",
    "C3-v06": "30f-kp7-attn",
    "C3-v07": "30f-low-alpha",
    "C3-v08": "30f-valp92",
    "C3-v09": "40f-low-alpha-valp92",
    "C3-v10": "30f-last-frame",
}

SCREEN_EXPERIMENTS = {
    "C3S-v01": "30f-kp7-small",
    "C3S-v02": "30f-kp7-large",
    "C3S-v03": "40f-kp7-small",
    "C3S-v04": "30f-minimal",
    "C3S-v05": "30f-low-alpha",
    "C3S-v06": "60f-kp7-small",
    "C3S-v07": "30f-attn",
    "C3S-v08": "30f-last-frame",
}

DEPLOY_EXPERIMENTS = {
    "C4D-v01": "30f-kp7-128x64",
    "C4D-v02": "30f-kp7-96x48",
    "C4D-v03": "30f-minimal-128x64",
    "C4D-v04": "40f-kp7-128x64",
    "C4D-v05": "30f-kp7-ce-128x64",
}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def metric_row(exp_id: str, label: str, output_root: Path, target: float) -> tuple[str, float, float]:
    metrics = load_json(output_root / exp_id / "metrics.json")
    if not metrics:
        state = "RUNNING" if (output_root / f"{exp_id}.log").exists() else "PENDING"
        return f"{exp_id:8s} {label:22s} {state:>8s}", -1.0, -1.0

    tv = metrics.get("metrics", {}).get("test_video", {})
    minp = float(tv.get("min_precision", 0.0))
    f1 = float(tv.get("f1", 0.0))
    passed = "YES" if minp >= target else "NO"
    line = (
        f"{exp_id:8s} {label:22s} "
        f"F1={f1:.4f} Rec={float(tv.get('recall', 0.0)):.4f} "
        f"FallP={float(tv.get('precision', 0.0)):.4f} "
        f"NFallP={float(tv.get('nfall_precision', 0.0)):.4f} "
        f"MinP={minp:.4f} Pass={passed}"
    )
    return line, minp, f1


def tail(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Report codex/03 GRU experiment progress.")
    parser.add_argument("--output-root", type=Path, default=Path("results/gru_codex03_optimal"))
    parser.add_argument("--target-min-precision", type=float, default=0.92)
    parser.add_argument("--tail-lines", type=int, default=8)
    args = parser.parse_args()

    output_root = args.output_root
    if "deploy" in output_root.name:
        experiments = DEPLOY_EXPERIMENTS
    elif "screen" in output_root.name:
        experiments = SCREEN_EXPERIMENTS
    else:
        experiments = EXPERIMENTS
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("")
    print(f"[{now}] codex/03 GRU status")
    print(f"output_root={output_root} target_min_precision={args.target_min_precision:.4f}")

    completed = sorted(exp_id for exp_id in experiments if (output_root / exp_id / "metrics.json").exists())
    print(f"completed={len(completed)}/{len(experiments)}")

    best: tuple[float, float, str] | None = None
    for exp_id, label in experiments.items():
        line, minp, f1 = metric_row(exp_id, label, output_root, args.target_min_precision)
        print(line)
        metrics = load_json(output_root / exp_id / "metrics.json")
        if metrics and "test_int8_video" in metrics.get("metrics", {}):
            qv = metrics["metrics"]["test_int8_video"]
            print(
                f"{'':8s} {'INT8 video':22s} "
                f"F1={float(qv.get('f1', 0.0)):.4f} Rec={float(qv.get('recall', 0.0)):.4f} "
                f"FallP={float(qv.get('precision', 0.0)):.4f} "
                f"NFallP={float(qv.get('nfall_precision', 0.0)):.4f} "
                f"MinP={float(qv.get('min_precision', 0.0)):.4f}"
            )
        if minp >= 0.0 and (best is None or (minp, f1) > (best[0], best[1])):
            best = (minp, f1, exp_id)

    if best is not None:
        print(f"best={best[2]} MinP={best[0]:.4f} F1={best[1]:.4f}")

    summary_tail = tail(output_root / "summary.log", args.tail_lines)
    if summary_tail:
        print("summary_tail:")
        for line in summary_tail:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
