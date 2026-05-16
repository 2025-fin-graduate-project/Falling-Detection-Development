#!/usr/bin/env python3
"""INT8 batch eval 결과 시각화 — results/quantization/int8_eval_summary.tsv 기반."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_tsv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    # filter out entries without int8 result
    rows = [r for r in rows if r.get("status") == "ok"]
    rows.sort(key=lambda r: float(r.get("float_event_minpr") or 0), reverse=True)
    return rows


def _fval(r: dict, key: str) -> float:
    try:
        return float(r[key])
    except (KeyError, ValueError, TypeError):
        return 0.0


def plot_minpr_comparison(rows: list[dict], out_dir: Path):
    ids = [r["exp_id"] for r in rows]
    floats = [_fval(r, "float_event_minpr") for r in rows]
    int8s = [_fval(r, "int8_event_minpr") for r in rows]
    n = len(ids)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 6))
    x = np.arange(n)
    w = 0.35

    bar1 = ax.bar(x - w / 2, floats, w, label="Float32", color="#4C72B0", alpha=0.85)
    bar2 = ax.bar(x + w / 2, int8s, w, label="INT8", color="#DD8452", alpha=0.85)

    # highlight INT8 >= 0.90
    for i, v in enumerate(int8s):
        if v >= 0.90:
            ax.bar(x[i] + w / 2, v, w, color="#2ca02c", alpha=0.85)

    ax.axhline(0.90, color="red", linestyle="--", linewidth=1.2, label="INT8 target (0.90)")
    ax.axhline(0.93, color="purple", linestyle=":", linewidth=1.0, label="Float target (0.93)")

    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Event MinPR")
    ax.set_title("Float32 vs INT8 Event MinPR (22 models)")
    ax.legend(loc="lower left")

    green_patch = mpatches.Patch(color="#2ca02c", alpha=0.85, label="INT8 ≥ 0.90")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [green_patch], labels=labels + ["INT8 ≥ 0.90"], loc="lower left", fontsize=8)

    fig.tight_layout()
    out = out_dir / "int8_minpr_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_degradation(rows: list[dict], out_dir: Path):
    ids = [r["exp_id"] for r in rows]
    degradation = [_fval(r, "float_event_minpr") - _fval(r, "int8_event_minpr") for r in rows]
    n = len(ids)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))
    colors = ["#d62728" if d > 0.03 else "#2ca02c" for d in degradation]
    ax.bar(range(n), degradation, color=colors, alpha=0.85)
    ax.axhline(0.03, color="orange", linestyle="--", linewidth=1.2, label="허용 열화 (0.03)")
    ax.set_xticks(range(n))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Float → INT8 MinPR 열화")
    ax.set_title("양자화 열화량 (Float - INT8 MinPR)")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "int8_degradation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_fn_fp(rows: list[dict], out_dir: Path):
    ids = [r["exp_id"] for r in rows]
    fns = [int(r.get("int8_fn") or 0) for r in rows]
    fps = [int(r.get("int8_fp") or 0) for r in rows]
    n = len(ids)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w / 2, fns, w, label="FN (missed falls)", color="#d62728", alpha=0.85)
    ax.bar(x + w / 2, fps, w, label="FP (false alarms)", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("INT8 FN / FP per model (test set, event level)")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "int8_fn_fp.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_flash_macc(rows: list[dict], out_dir: Path):
    ids = [r["exp_id"] for r in rows]
    flash = [_fval(r, "weights_kib") for r in rows]
    macc = [_fval(r, "macc") / 1e6 for r in rows]  # → MACC (M)
    n = len(ids)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n * 0.6), 8), sharex=True)

    colors_f = ["#d62728" if v > 600 else "#4C72B0" for v in flash]
    ax1.bar(range(n), flash, color=colors_f, alpha=0.85)
    ax1.axhline(60000, color="red", linestyle="--", linewidth=1, label="Flash 한계 60MB")
    ax1.set_ylabel("Flash (KiB)")
    ax1.set_title("STedgeAI INT8 Flash 사용량")
    ax1.legend(fontsize=8)

    ax2.bar(range(n), macc, color="#4C72B0", alpha=0.85)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("MACC (M/frame)")
    ax2.set_title("MACC/frame")

    fig.tight_layout()
    out = out_dir / "int8_hw_resources.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def print_summary(rows: list[dict]):
    passed = [r for r in rows if _fval(r, "int8_event_minpr") >= 0.90]
    print(f"\n=== INT8 평가 요약 ===")
    print(f"총 평가: {len(rows)}개  |  INT8 ≥ 0.90: {len(passed)}개\n")
    print(f"{'ID':<25} {'Float':>7} {'INT8':>7} {'Δ':>6} {'FN':>4} {'FP':>4}  {'Flash(KiB)':>10}")
    print("-" * 75)
    for r in rows:
        f = _fval(r, "float_event_minpr")
        i = _fval(r, "int8_event_minpr")
        flag = " ✓" if i >= 0.90 else ""
        print(f"{r['exp_id']:<25} {f:>7.4f} {i:>7.4f} {f-i:>6.4f} "
              f"{r.get('int8_fn','-'):>4} {r.get('int8_fp','-'):>4}  "
              f"{r.get('weights_kib','-'):>10}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="results/quantization/int8_eval_summary.tsv")
    ap.add_argument("--out-dir", default="results/quantization")
    args = ap.parse_args()

    tsv = Path(args.tsv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tsv.exists():
        print(f"TSV not found: {tsv}")
        return

    rows = load_tsv(tsv)
    if not rows:
        print("평가 완료된 모델 없음 (status=ok 행 없음)")
        return

    print(f"평가 완료 모델: {len(rows)}개")
    print_summary(rows)

    plot_minpr_comparison(rows, out_dir)
    plot_degradation(rows, out_dir)
    plot_fn_fp(rows, out_dir)
    plot_flash_macc(rows, out_dir)
    print("\n완료.")


if __name__ == "__main__":
    main()
