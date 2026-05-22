#!/usr/bin/env python3
"""Fill TBD INT8 cells in all documentation files after eval_stedgeai_host_p37.py completes.

Run after: uv run python scripts/util/eval_stedgeai_host_p37.py --exp-dir ... completes.

Usage:
    python3 scripts/util/fill_int8_docs.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]

MODELS = [
    ("P42-kp17-w60",      PROJ / "results/phase42_cross/P42-kp17-w60"),
    ("P41-kp13-w60",      PROJ / "results/phase41_ablation/P41-kp13-w60"),
    ("P41-raw-kp7-w40",   PROJ / "results/phase41_ablation/P41-raw-kp7-w40"),
    ("P41-kp17-w40",      PROJ / "results/phase41_ablation/P41-kp17-w40"),
    ("P41-vel-kp13-w40",  PROJ / "results/phase41_ablation/P41-vel-kp13-w40"),
    ("P42-kp13-w40-h128", PROJ / "results/phase42_cross/P42-kp13-w40-h128"),
    ("P42-kp13-w60-h128", PROJ / "results/phase42_cross/P42-kp13-w60-h128"),
]

FLOAT_MINPR = {
    "P42-kp17-w60":      0.9211,
    "P41-kp13-w60":      0.9167,
    "P41-raw-kp7-w40":   0.9167,
    "P41-kp17-w40":      0.9123,
    "P41-vel-kp13-w40":  0.9123,
    "P42-kp13-w40-h128": 0.8991,
    "P42-kp13-w60-h128": 0.9035,
}


def load_int8_results() -> dict:
    results = {}
    missing = []
    for name, d in MODELS:
        mf = d / "metrics.json"
        if not mf.exists():
            missing.append(f"{name}: metrics.json missing")
            continue
        m = json.loads(mf.read_text())
        ev = m.get("stedgeai_host_eval", {})
        if not ev:
            missing.append(f"{name}: no stedgeai_host_eval")
            continue
        int8_minpr = ev.get("min_precision")
        float_minpr = FLOAT_MINPR[name]
        loss = round(int8_minpr - float_minpr, 4) if int8_minpr is not None else None
        goal = "✅" if (int8_minpr is not None and int8_minpr >= 0.90) else "❌"
        results[name] = {
            "float":    float_minpr,
            "int8":     int8_minpr,
            "loss":     loss,
            "fall_pr":  ev.get("fall_precision"),
            "fall_re":  ev.get("fall_recall"),
            "nfall_pr": ev.get("nfall_precision"),
            "nfall_re": ev.get("nfall_recall"),
            "vw":       ev.get("vote_window"),
            "vk":       ev.get("vote_k"),
            "goal":     goal,
            "stride":   ev.get("eval_stride"),
        }
    if missing:
        for m in missing:
            print(f"  MISSING: {m}")
    return results


def fmt(v, fmt_str=".4f"):
    if v is None:
        return "N/A"
    try:
        return format(v, fmt_str)
    except Exception:
        return str(v)


def update_top10_summary(results: dict, dry_run: bool):
    path = PROJ / "results/phase43_report/top10_summary.txt"
    content = path.read_text()
    for name, r in results.items():
        int8_str = fmt(r["int8"])
        content = content.replace(
            f"{name:<32}", f"{name:<32}"
        )
        # Replace the '-' in the int8 column for this model
        pattern = rf"({re.escape(name)}\s+\S+\s+\d+\s+\S+\s+\S+\s+\S+\s+[\d.]+\s+\S+)\s+-"
        replacement = rf"\1   {int8_str}"
        content = re.sub(pattern, replacement, content)
    if not dry_run:
        path.write_text(content)
    print(f"  top10_summary.txt {'(dry-run)' if dry_run else 'updated'}")


def update_doc_tbd_table(path: Path, results: dict, dry_run: bool):
    """Replace TBD cells in markdown tables for each model row."""
    content = path.read_text()
    original = content

    for name, r in results.items():
        int8_val = fmt(r["int8"])
        loss_val = fmt(r["loss"], "+.4f") if r["loss"] is not None else "—"
        goal_val = r["goal"]

        # Pattern: | ModelName ... | 0.XXXX | TBD | — | — |
        # Replace TBD with int8 value and fill loss/goal
        pattern = rf"(\|\s*{re.escape(name)}\s*\|[^|]*\|\s*{fmt(r['float'])}\s*\|\s*)TBD(\s*\|\s*)—(\s*\|\s*)—(\s*\|)"
        replacement = rf"\g<1>{int8_val}\g<2>{loss_val}\g<3>{goal_val}\g<4>"
        content = re.sub(pattern, replacement, content)

        # Simpler single-TBD pattern (07_model_evaluation.md style)
        pattern2 = rf"(\|\s*{re.escape(name)}\s*\|[^|]*\|\s*{fmt(r['float'])}\s*\|\s*)TBD(\s*\|)"
        replacement2 = rf"\g<1>{int8_val}\g<2>"
        content = re.sub(pattern2, replacement2, content)

    if content != original:
        if not dry_run:
            path.write_text(content)
        print(f"  {path.name} {'(dry-run)' if dry_run else 'updated'}")
    else:
        print(f"  {path.name} — no TBD patterns matched (check manually)")


def print_summary(results: dict):
    print("\n" + "="*72)
    print(f"{'Model':<24} {'Float':>6}  {'INT8':>6}  {'Loss':>7}  {'Goal'}")
    print("-"*72)
    for name, r in results.items():
        print(f"{name:<24} {fmt(r['float']):>6}  {fmt(r['int8']):>6}  {fmt(r['loss'],'+.4f'):>7}  {r['goal']}")
    print("="*72)
    passed = sum(1 for r in results.values() if r.get("int8") is not None and r["int8"] >= 0.90)
    print(f"INT8 ≥ 0.90: {passed}/{len(results)} models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("Loading INT8 eval results...")
    results = load_int8_results()

    if not results:
        print("No INT8 results found. Run eval_stedgeai_host_p37.py first.")
        return

    print_summary(results)

    docs = [
        PROJ / "docs/final-report/04_postprocessing_quantization.md",
        PROJ / "docs/final-report/06_ablation_study.md",
        PROJ / "docs/final-report/07_model_evaluation.md",
        PROJ / "results/phase43_report/experiment_report.md",
    ]

    print(f"\nUpdating {len(docs)} documentation files {'(DRY RUN)' if args.dry_run else ''}...")
    for doc in docs:
        if doc.exists():
            update_doc_tbd_table(doc, results, args.dry_run)
        else:
            print(f"  MISSING: {doc}")

    update_top10_summary(results, args.dry_run)
    print("\nDone.")


if __name__ == "__main__":
    main()
