#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PHASE9_LABELS = {
    "P9S-v01": "gru-128x64-kp7-40f-2conv-focal",
    "P9S-v02": "gru-128x64-kp7-30f-2conv-focal",
    "P9S-v03": "gru-128x64-kp12-30f-2conv-focal",
    "P9S-v04": "gru-256x128-minimal-40f-focal",
    "P9S-v05": "tcn-light-kp7-30f",
    "P9S-v06": "tcn-light-kp7-40f",
    "P9S-v07": "tcn-medium-kp7-30f",
    "P9F-v01": "full-gru-128x64-kp7-40f-2conv-focal",
    "P9F-v02": "full-gru-128x64-kp7-30f-2conv-focal",
    "P9F-v03": "full-gru-128x64-kp12-30f-2conv-focal",
    "P9F-v04": "full-gru-256x128-minimal-40f-focal",
    "P9F-v05": "full-tcn-light-kp7-30f",
    "P9F-v06": "full-tcn-light-kp7-40f",
    "P9F-v07": "full-tcn-medium-kp7-30f",
}


@dataclass
class Row:
    exp_id: str
    label: str
    path: Path
    source: str
    min_pr: float
    fall_precision: float
    nfall_precision: float
    fall_recall: float
    nfall_recall: float
    f1: float
    threshold: float | None
    min_consecutive: int | None
    stedgeai_ok: bool
    state: str = "DONE"

    @property
    def sort_key(self) -> tuple[float, float, float, int, float]:
        return (
            self.min_pr,
            self.fall_recall,
            self.fall_precision,
            1 if self.stedgeai_ok else 0,
            self.f1,
        )


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0


def from_confusion_matrix(metric: dict[str, Any]) -> dict[str, float] | None:
    cm = metric.get("confusion_matrix")
    if not cm or len(cm) != 2 or len(cm[0]) != 2 or len(cm[1]) != 2:
        return None
    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    fall_p = safe_div(tp, tp + fp)
    nfall_p = safe_div(tn, tn + fn)
    fall_r = safe_div(tp, tp + fn)
    nfall_r = safe_div(tn, tn + fp)
    f1 = safe_div(2 * fall_p * fall_r, fall_p + fall_r)
    return {
        "fall_precision": fall_p,
        "nfall_precision": nfall_p,
        "fall_recall": fall_r,
        "nfall_recall": nfall_r,
        "min_pr": min(fall_p, nfall_p, fall_r, nfall_r),
        "f1": f1,
    }


def from_tp_counts(metric: dict[str, Any]) -> dict[str, float] | None:
    keys = {"tp", "tn", "fp", "fn"}
    if not keys.issubset(metric):
        return None
    tp, tn, fp, fn = (int(metric[key]) for key in ("tp", "tn", "fp", "fn"))
    fall_p = safe_div(tp, tp + fp)
    nfall_p = safe_div(tn, tn + fn)
    fall_r = safe_div(tp, tp + fn)
    nfall_r = safe_div(tn, tn + fp)
    f1 = safe_div(2 * fall_p * fall_r, fall_p + fall_r)
    return {
        "fall_precision": fall_p,
        "nfall_precision": nfall_p,
        "fall_recall": fall_r,
        "nfall_recall": nfall_r,
        "min_pr": min(fall_p, nfall_p, fall_r, nfall_r),
        "f1": f1,
    }


def metric_values(metric: dict[str, Any]) -> dict[str, float]:
    recomputed = from_confusion_matrix(metric) or from_tp_counts(metric)
    if recomputed:
        return recomputed
    fall_p = float(metric.get("precision", metric.get("fall_precision", 0.0)))
    nfall_p = float(metric.get("nfall_precision", 0.0))
    fall_r = float(metric.get("recall", 0.0))
    nfall_r = float(metric.get("nfall_recall", 0.0))
    return {
        "fall_precision": fall_p,
        "nfall_precision": nfall_p,
        "fall_recall": fall_r,
        "nfall_recall": nfall_r,
        "min_pr": min(fall_p, nfall_p, fall_r, nfall_r),
        "f1": float(metric.get("f1", 0.0)),
    }


def row_from_metrics(metrics_path: Path, source: str) -> Row | None:
    payload = load_json(metrics_path)
    if not payload:
        return None
    exp_id = str(payload.get("experiment_id") or metrics_path.parent.name)
    metrics = payload.get("metrics", {})
    test_video = metrics.get("test_video")
    if not isinstance(test_video, dict):
        return None
    values = metric_values(test_video)
    threshold_selection = payload.get("threshold_selection", {})
    stedgeai = payload.get("stedgeai", {}).get("analyze", {})
    stedgeai_host = payload.get("stedgeai_host_eval", {})
    stedgeai_ok = bool(stedgeai.get("analyze_ok") or stedgeai_host.get("eval_ok"))
    return Row(
        exp_id=exp_id,
        label=PHASE9_LABELS.get(exp_id, "-"),
        path=metrics_path.parent,
        source=source,
        min_pr=values["min_pr"],
        fall_precision=values["fall_precision"],
        nfall_precision=values["nfall_precision"],
        fall_recall=values["fall_recall"],
        nfall_recall=values["nfall_recall"],
        f1=values["f1"],
        threshold=float(threshold_selection["threshold"]) if "threshold" in threshold_selection else None,
        min_consecutive=int(threshold_selection["min_consecutive"]) if "min_consecutive" in threshold_selection else None,
        stedgeai_ok=stedgeai_ok,
    )


def collect_rows(output_root: Path, source: str) -> list[Row]:
    if not output_root.exists():
        return []
    rows = []
    for metrics_path in sorted(output_root.glob("*/metrics.json")):
        row = row_from_metrics(metrics_path, source)
        if row:
            rows.append(row)
    return rows


def collect_pending(output_root: Path, rows: list[Row]) -> list[Row]:
    known = {row.exp_id for row in rows}
    pending = []
    for exp_id, label in PHASE9_LABELS.items():
        if exp_id in known:
            continue
        if (exp_id.startswith("P9S") and "screen" in output_root.name) or (exp_id.startswith("P9F") and "full" in output_root.name):
            state = "RUNNING" if (output_root / f"{exp_id}.log").exists() else "PENDING"
            pending.append(
                Row(exp_id, label, output_root / exp_id, "phase9", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, False, state)
            )
    return pending


def collect_legacy(results_root: Path) -> list[Row]:
    rows = []
    if not results_root.exists():
        return rows
    for metrics_path in sorted(results_root.glob("*/*/metrics.json")):
        if "phase9_minpr_" in str(metrics_path):
            continue
        row = row_from_metrics(metrics_path, "legacy")
        if row:
            rows.append(row)
    return rows


def print_rows(rows: list[Row], target: float, limit: int | None = None) -> None:
    shown = rows[:limit] if limit else rows
    print(
        f"{'ID':8s} {'Config':36s} {'MinPR':>7s} {'FallP':>7s} {'NFallP':>7s} "
        f"{'FallR':>7s} {'NFallR':>7s} {'F1':>7s} {'Thr':>5s} {'MC':>3s} {'Pass':>4s} {'ST':>3s}"
    )
    print("-" * 117)
    for row in shown:
        if row.state != "DONE":
            print(f"{row.exp_id:8s} {row.label[:36]:36s} {row.state:>7s}")
            continue
        passed = "YES" if row.min_pr >= target else "NO"
        st = "YES" if row.stedgeai_ok else "-"
        thr = f"{row.threshold:.2f}" if row.threshold is not None else "-"
        mc = str(row.min_consecutive) if row.min_consecutive is not None else "-"
        print(
            f"{row.exp_id:8s} {row.label[:36]:36s} {row.min_pr:7.4f} "
            f"{row.fall_precision:7.4f} {row.nfall_precision:7.4f} "
            f"{row.fall_recall:7.4f} {row.nfall_recall:7.4f} {row.f1:7.4f} "
            f"{thr:>5s} {mc:>3s} {passed:>4s} {st:>3s}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Report Phase 9 MinPR experiment status without mutating results.")
    parser.add_argument("--output-root", type=Path, default=Path("results/phase9_minpr_screen"))
    parser.add_argument("--target-min-pr", type=float, default=0.92)
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--top", type=int, default=0, help="Limit printed rows, or ids-only count.")
    parser.add_argument("--ids-only", action="store_true", help="Print only ranked experiment IDs for automation.")
    args = parser.parse_args()

    rows = collect_rows(args.output_root, "phase9")
    ranked = sorted(rows, key=lambda row: row.sort_key, reverse=True)

    if args.ids_only:
        for row in ranked[: args.top or len(ranked)]:
            print(row.exp_id)
        return 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("")
    print(f"[{now}] Phase 9 MinPR status")
    print(f"output_root={args.output_root} target_min_pr={args.target_min_pr:.4f}")
    print(f"completed={len(rows)}")
    print_rows(ranked + collect_pending(args.output_root, rows), args.target_min_pr, args.top or None)

    if ranked:
        best = ranked[0]
        print(
            f"\nbest={best.exp_id} MinPR={best.min_pr:.4f} "
            f"FallR={best.fall_recall:.4f} FallP={best.fall_precision:.4f}"
        )

    if args.include_legacy:
        legacy = sorted(collect_legacy(args.results_root), key=lambda row: row.sort_key, reverse=True)
        if legacy:
            print("\nLegacy baseline recalculated from confusion matrices")
            print_rows(legacy, args.target_min_pr, 20)
        else:
            print("\nLegacy baseline recalculated from confusion matrices: no metrics found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
