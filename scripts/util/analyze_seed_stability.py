#!/usr/bin/env python3
"""Analyze seed stability experiment results.

Reads metrics.json from exp4_seed_stability runs and computes
mean/std of ev_minpr across seeds for each model config.

Usage:
    uv run python scripts/util/analyze_seed_stability.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = PROJ_ROOT / "results/additional_report_experiments/exp4_seed_stability"
OUT_DIR = PROJ_ROOT / "results/additional_report_experiments/exp4_seed_stability"

MODEL_PREFIXES = [
    ("kp17-w60",       "kp17, w=60, filtered"),
    ("kp13-w60",       "kp13, w=60, filtered"),
    ("kp7-w40-raw",    "kp7, w=40, raw"),
]
SEEDS = [0, 7, 13, 21, 37, 99]


def main():
    rows = []
    summary_rows = []

    for prefix, label in MODEL_PREFIXES:
        model_minprs = []
        for seed in SEEDS:
            exp_id = f"seed-{prefix}-s{seed}"
            met_path = EXP_DIR / exp_id / "metrics.json"
            if not met_path.exists():
                print(f"  MISSING: {met_path}")
                continue
            mets = json.loads(met_path.read_text())
            ue = mets.get("unified_eval", {})
            ev = ue.get("event", {})
            minpr = ev.get("min_pr") or ev.get("test", {}).get("min_pr")
            if minpr is None:
                print(f"  No ev_minpr in {exp_id}")
                continue
            thr   = ue.get("threshold")
            vw    = ev.get("vote_window")
            vk    = ev.get("vote_k")
            rows.append({
                "model": prefix,
                "label": label,
                "seed": seed,
                "ev_minpr": minpr,
                "threshold": thr,
                "vote_window": vw,
                "vote_k": vk,
            })
            model_minprs.append(minpr)
            print(f"  {exp_id}: ev_minpr={minpr:.4f} thr={thr} v{vw}k{vk}")

        if model_minprs:
            a = np.array(model_minprs)
            summary_rows.append({
                "model": prefix,
                "label": label,
                "n_seeds": len(a),
                "mean_ev_minpr": round(float(a.mean()), 4),
                "std_ev_minpr":  round(float(a.std()),  4),
                "min_ev_minpr":  round(float(a.min()),  4),
                "max_ev_minpr":  round(float(a.max()),  4),
                "p25": round(float(np.percentile(a, 25)), 4),
                "p75": round(float(np.percentile(a, 75)), 4),
            })
            print(f"  → {prefix}: mean={a.mean():.4f} ±{a.std():.4f} "
                  f"[{a.min():.4f}, {a.max():.4f}]")

    # Save per-seed CSV
    if rows:
        path = OUT_DIR / "seed_stability_per_seed.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {path}")

    # Save summary CSV
    if summary_rows:
        path = OUT_DIR / "seed_stability_summary.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Saved: {path}")

    # Plot
    try:
        plot(rows, summary_rows)
    except Exception as e:
        print(f"  Plot error: {e}")


def plot(rows, summary_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [r["model"] for r in summary_rows]
    means  = [r["mean_ev_minpr"] for r in summary_rows]
    stds   = [r["std_ev_minpr"] for r in summary_rows]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart with error bars
    x = range(len(models))
    bars = ax1.bar(x, means, yerr=stds, capsize=8, color=colors[:len(models)],
                   alpha=0.85, width=0.5)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([r["label"] for r in summary_rows], fontsize=8)
    ax1.set_ylabel("Event MinP (mean ± std over 6 seeds)")
    ax1.set_title("Seed Stability: Event MinP Distribution")
    ax1.set_ylim(0.85, 0.98)
    ax1.axhline(0.90, ls="--", color="red", linewidth=1, label="INT8 target (0.90)")
    ax1.axhline(0.93, ls=":", color="orange", linewidth=1, label="Float target (0.93)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.001,
                 f"{m:.4f}", ha="center", va="bottom", fontsize=8)

    # Scatter: one point per seed
    model_to_idx = {m: i for i, m in enumerate(models)}
    for model_prefix, label in [(r["model"], r["label"]) for r in summary_rows]:
        idx = model_to_idx[model_prefix]
        pts = [r["ev_minpr"] for r in rows if r["model"] == model_prefix]
        seeds_used = [r["seed"] for r in rows if r["model"] == model_prefix]
        jitter = np.random.uniform(-0.08, 0.08, len(pts))
        ax2.scatter([idx + j for j in jitter], pts,
                    color=colors[idx], alpha=0.8, s=60, zorder=3)

    ax2.set_xticks(list(range(len(models))))
    ax2.set_xticklabels([r["label"] for r in summary_rows], fontsize=8)
    ax2.set_ylabel("Event MinP per seed")
    ax2.set_title("Seed Stability: Per-seed Values")
    ax2.set_ylim(0.85, 0.98)
    ax2.axhline(0.90, ls="--", color="red", linewidth=1)
    ax2.axhline(0.93, ls=":", color="orange", linewidth=1)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = OUT_DIR / "fig_seed_stability.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
