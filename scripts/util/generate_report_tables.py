#!/usr/bin/env python3
"""Generate consolidated result tables for the final report.

Reads all metrics.json files from Phase 41/42 and additional experiments,
produces paper-ready CSV tables with the standard format:
  model, condition, event_minpr, fall_precision, fall_recall,
  nfall_precision, nfall_recall, tp, fp, fn, tn

Also generates supplementary tables for the additional experiments.

Usage:
    uv run python scripts/util/generate_report_tables.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJ_ROOT / "results/additional_report_experiments/report_tables"

# All Phase 41/42 models
ABLATION_MODELS = [
    # Feature set ablation (w=40, filtered, no vel)
    ("P41-kp5-w40",       "kp5, w=40, filtered"),
    ("P41-kp7-w40",       "kp7, w=40, filtered"),
    ("P41-kp9-w40",       "kp9, w=40, filtered"),
    ("P41-kp11-w40",      "kp11, w=40, filtered"),
    ("P41-kp13-w40",      "kp13, w=40, filtered"),
    ("P41-kp17-w40",      "kp17, w=40, filtered"),
    # Window size ablation (kp13, filtered, no vel)
    ("P41-kp13-w20",      "kp13, w=20, filtered"),
    ("P41-kp13-w30",      "kp13, w=30, filtered"),
    ("P41-kp13-w60",      "kp13, w=60, filtered"),
    # Architecture comparison (kp7, w=40, filtered)
    ("P41-tcn-kp7-w40",   "TCN, kp7, w=40, filtered"),
    ("P41-tcn-kp13-w40",  "TCN, kp13, w=40, filtered"),
    # Velocity features
    ("P41-vel-kp7-w40",   "kp7+vel, w=40, filtered"),
    ("P41-vel-kp13-w40",  "kp13+vel, w=40, filtered"),
    # Preprocessing: raw
    ("P41-raw-kp7-w40",   "kp7, w=40, raw"),
    ("P41-raw-kp13-w40",  "kp13, w=40, raw"),
    ("P41-raw-kp17-w40",  "kp17, w=40, raw"),
    # Phase 42 cross-validation
    ("P42-kp17-w60",      "kp17, w=60, filtered"),
    ("P42-raw-kp7-w60",   "kp7, w=60, raw"),
    ("P42-raw-kp17-w40",  "kp17, w=40, raw"),
    ("P42-vel-kp13-w60",  "kp13+vel, w=60, filtered"),
    ("P42-kp13-w40-h128", "kp13, w=40, h128, filtered"),
    ("P42-kp13-w60-h128", "kp13, w=60, h128, filtered"),
]

ABLATION_DIRS = [
    PROJ_ROOT / "results/phase41_ablation",
    PROJ_ROOT / "results/phase42_cross",
]


def find_exp_dir(exp_id: str) -> Path | None:
    for root in ABLATION_DIRS:
        p = root / exp_id
        if (p / "metrics.json").exists():
            return p
    return None


def extract_ablation_row(exp_id: str, condition: str) -> dict | None:
    exp_dir = find_exp_dir(exp_id)
    if exp_dir is None:
        return None
    mets = json.loads((exp_dir / "metrics.json").read_text())
    ue   = mets.get("unified_eval", {})
    ev   = ue.get("event", {})
    test = ev.get("test", {})

    # Hardware footprint from stedgeai analyze
    st   = mets.get("stedgeai", {}).get("analyze", {})

    return {
        "model": exp_id,
        "condition": condition,
        "event_minpr":      round(test.get("min_pr", ev.get("min_pr", 0)), 4),
        "fall_precision":   round(test.get("fall_precision", 0), 4),
        "fall_recall":      round(test.get("fall_recall", 0), 4),
        "nfall_precision":  round(test.get("nfall_precision", 0), 4),
        "nfall_recall":     round(test.get("nfall_recall", 0), 4),
        "tp": test.get("tp", ""),
        "fp": test.get("fp", ""),
        "fn": test.get("fn", ""),
        "tn": test.get("tn", ""),
        "threshold":        round(ue.get("threshold", 0), 3),
        "vote_window":      ev.get("vote_window", ""),
        "vote_k":           ev.get("vote_k", ""),
        "flash_kib":        st.get("weights_kib", ""),
        "activation_kib":   st.get("activations_kib", ""),
        "int8_minpr":       mets.get("stedgeai_host_eval", {}).get("min_precision", "TBD"),
    }


def generate_ablation_table():
    rows = []
    for exp_id, condition in ABLATION_MODELS:
        r = extract_ablation_row(exp_id, condition)
        if r is not None:
            rows.append(r)
        else:
            print(f"  MISSING: {exp_id}")

    if not rows:
        print("No ablation data found")
        return

    path = OUT_DIR / "ablation_full_results.csv"
    fieldnames = ["model", "condition", "event_minpr", "fall_precision", "fall_recall",
                  "nfall_precision", "nfall_recall", "tp", "fp", "fn", "tn",
                  "threshold", "vote_window", "vote_k", "flash_kib", "activation_kib", "int8_minpr"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path} ({len(rows)} rows)")

    # Print top 10 by event_minpr
    top10 = sorted(rows, key=lambda r: r["event_minpr"], reverse=True)[:10]
    print("\nTop 10 by Event MinP:")
    print(f"{'Model':<25} {'Condition':<30} {'ev_minpr':>8} {'INT8':>6} {'Flash':>6}")
    for r in top10:
        print(f"{r['model']:<25} {r['condition']:<30} {r['event_minpr']:>8.4f} {str(r['int8_minpr']):>6} {str(r['flash_kib']):>6}")


def generate_camera_table():
    """Aggregate camera breakdown for all 3 top models."""
    cam_path = PROJ_ROOT / "results/additional_report_experiments/exp1_camera_direction/camera_direction_metrics.csv"
    if not cam_path.exists():
        print("Camera breakdown CSV not found")
        return

    rows = []
    with open(cam_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    path = OUT_DIR / "camera_direction_results.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}")


def generate_latency_table():
    src = PROJ_ROOT / "results/additional_report_experiments/exp3_detection_latency/latency_stats.csv"
    if not src.exists():
        print("Latency stats CSV not found")
        return
    dst = OUT_DIR / "latency_stats.csv"
    dst.write_text(src.read_text())
    print(f"Saved: {dst}")


def generate_confidence_table():
    src = PROJ_ROOT / "results/additional_report_experiments/exp6_confidence_quartile/confidence_quartile_metrics.csv"
    if not src.exists():
        print("Confidence quartile CSV not found")
        return
    dst = OUT_DIR / "confidence_quartile_results.csv"
    dst.write_text(src.read_text())
    print(f"Saved: {dst}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Generating Report Tables ===\n")

    print("--- Ablation Table ---")
    generate_ablation_table()

    print("\n--- Camera/Direction Table ---")
    generate_camera_table()

    print("\n--- Latency Table ---")
    generate_latency_table()

    print("\n--- Confidence Quartile Table ---")
    generate_confidence_table()

    print(f"\nAll tables saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
