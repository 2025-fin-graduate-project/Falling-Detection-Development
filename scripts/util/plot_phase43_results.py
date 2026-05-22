#!/usr/bin/env python3
"""Phase 43: P41+P42 결과 시각화 및 통합 테이블 생성.

논문용 차트 생성:
  1. feature_set vs ev_minpr (bar)
  2. window_size vs ev_minpr (line, filt vs raw)
  3. raw vs filtered (paired bar)
  4. velocity 효과 (paired bar)
  5. GRU 64,32 vs 128,64 (paired bar)
  6. TCN vs GRU (bar)
  7. 전체 통합 결과 테이블 CSV
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

PHASE_DIRS = [
    "results/phase41_ablation",
    "results/phase42_cross",
]


def load_all(repo: pathlib.Path) -> pd.DataFrame:
    rows = []
    for pd_ in PHASE_DIRS:
        for m in sorted((repo / pd_).rglob("metrics.json")):
            try:
                d = json.loads(m.read_text())
                ue = d.get("unified_eval", {})
                ev = ue.get("event", {}).get("min_pr", None)
                win = ue.get("window", {}).get("min_pr", None)
                cfg = d.get("config", {})
                data_dir = cfg.get("data_dir", "")
                prep = "raw" if ("class_balanced" in data_dir and "filtered" not in data_dir) else "filt"
                stedge = d.get("stedgeai", {}).get("analyze", {})
                int8 = d.get("stedgeai_host_eval", {})
                rows.append({
                    "id": m.parent.name,
                    "phase": int(pd_.split("phase")[1].split("_")[0]),
                    "kp": cfg.get("feature_set", "?"),
                    "window": cfg.get("window_size", 0),
                    "vel": cfg.get("use_velocity", False),
                    "prep": prep,
                    "model": cfg.get("model_type", "gru"),
                    "hidden": str(cfg.get("hidden_sizes", [64, 32])),
                    "win_minpr": win,
                    "ev_minpr": ev,
                    "flash_kib": stedge.get("weights_kib", None),
                    "act_kib": stedge.get("activations_kib", None),
                    "macc": stedge.get("macc", None),
                    "int8_minpr": int8.get("min_precision", None),
                    "exp_dir": str(m.parent),
                })
            except Exception:
                pass
    return pd.DataFrame(rows)


def bar_feature_set(df: pd.DataFrame, out: pathlib.Path):
    """feature_set vs ev_minpr (filt, no vel, GRU64, w40)"""
    sub = df[
        (df.prep == "filt") & (~df.vel) &
        (df.model == "gru") & (df.hidden == "[64, 32]") &
        (df.window == 40)
    ].dropna(subset=["ev_minpr"])
    sub = sub.sort_values("kp")

    kp_order = ["kp5", "kp7", "kp9", "kp11", "kp13", "kp17"]
    sub = sub[sub.kp.isin(kp_order)].copy()
    sub["kp"] = pd.Categorical(sub.kp, categories=kp_order, ordered=True)
    sub = sub.sort_values("kp")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(sub))
    bars = ax.bar(x, sub.ev_minpr, color="#4C72B0", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(sub.kp.tolist())
    ax.set_xlabel("Feature Set")
    ax.set_ylabel("Event-level MinPR")
    ax.set_title("Feature Set vs. Event MinPR\n(filtered, no vel, GRU 64/32, w=40)")
    ax.set_ylim(0.80, 0.95)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    for bar, v in zip(bars, sub.ev_minpr):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1, label="Target (0.90)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig1_feature_set.png")
    plt.close(fig)
    print("  [fig1] feature_set saved")


def line_window_size(df: pd.DataFrame, out: pathlib.Path):
    """window_size vs ev_minpr for filt vs raw (kp13, no vel, GRU64)"""
    fig, ax = plt.subplots(figsize=(6, 4))
    for prep, color, marker in [("filt", "#4C72B0", "o"), ("raw", "#DD8452", "s")]:
        sub = df[
            (df.prep == prep) & (~df.vel) &
            (df.model == "gru") & (df.hidden == "[64, 32]") &
            (df.kp == "kp13")
        ].dropna(subset=["ev_minpr"]).sort_values("window")
        if sub.empty:
            continue
        ax.plot(sub.window, sub.ev_minpr, marker=marker, color=color, label=prep, linewidth=1.5)
        for _, row in sub.iterrows():
            ax.annotate(f"{row.ev_minpr:.4f}", (row.window, row.ev_minpr),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("Window Size (frames)")
    ax.set_ylabel("Event-level MinPR")
    ax.set_title("Window Size vs. Event MinPR\n(kp13, no vel, GRU 64/32)")
    ax.set_xticks([20, 30, 40, 60])
    ax.set_ylim(0.84, 0.95)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1, label="Target (0.90)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig2_window_size.png")
    plt.close(fig)
    print("  [fig2] window_size saved")


def bar_raw_vs_filt(df: pd.DataFrame, out: pathlib.Path):
    """raw vs filt paired bar for matched experiments"""
    pairs = [
        ("kp7", 40, False, "kp7-w40"),
        ("kp13", 40, False, "kp13-w40"),
        ("kp13", 60, False, "kp13-w60"),
        ("kp17", 40, False, "kp17-w40"),
    ]
    labels, filt_vals, raw_vals = [], [], []
    for kp, w, vel, label in pairs:
        base = df[(df.kp == kp) & (df.window == w) & (df.vel == vel) &
                  (df.model == "gru") & (df.hidden == "[64, 32]")]
        fv = base[base.prep == "filt"].ev_minpr.values
        rv = base[base.prep == "raw"].ev_minpr.values
        if len(fv) and len(rv):
            labels.append(label)
            filt_vals.append(float(fv[0]))
            raw_vals.append(float(rv[0]))

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, filt_vals, w, label="Filtered", color="#4C72B0")
    b2 = ax.bar(x + w/2, raw_vals, w, label="Raw", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Config")
    ax.set_ylabel("Event-level MinPR")
    ax.set_title("Filtered vs. Raw Preprocessing")
    ax.set_ylim(0.88, 0.935)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    for bar, v in zip(list(b1) + list(b2), filt_vals + raw_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.0005, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1, label="Target (0.90)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig3_raw_vs_filt.png")
    plt.close(fig)
    print("  [fig3] raw_vs_filt saved")


def bar_velocity(df: pd.DataFrame, out: pathlib.Path):
    """velocity ON vs OFF"""
    pairs = [
        ("kp13", 40, "filt", "kp13-w40-filt"),
        ("kp7", 40, "filt", "kp7-w40-filt"),
        ("kp13", 60, "filt", "kp13-w60-filt"),
        ("kp13", 40, "raw", "kp13-w40-raw"),
    ]
    labels, no_vel, with_vel = [], [], []
    for kp, w, prep, label in pairs:
        base = df[(df.kp == kp) & (df.window == w) & (df.prep == prep) &
                  (df.model == "gru") & (df.hidden == "[64, 32]")]
        nv = base[~base.vel].ev_minpr.values
        wv = base[base.vel].ev_minpr.values
        if len(nv) and len(wv):
            labels.append(label)
            no_vel.append(float(nv[0]))
            with_vel.append(float(wv[0]))

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, no_vel, w, label="No Velocity", color="#4C72B0")
    b2 = ax.bar(x + w/2, with_vel, w, label="+ Velocity", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Config")
    ax.set_ylabel("Event-level MinPR")
    ax.set_title("Effect of Velocity Features")
    ymin = min(no_vel + with_vel) - 0.01
    ymax = max(no_vel + with_vel) + 0.01
    ax.set_ylim(max(0.84, ymin), min(0.95, ymax))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    for bar, v in zip(list(b1) + list(b2), no_vel + with_vel):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.0003, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig4_velocity.png")
    plt.close(fig)
    print("  [fig4] velocity saved")


def bar_hidden_size(df: pd.DataFrame, out: pathlib.Path):
    """GRU 64,32 vs 128,64"""
    pairs = [
        ("kp13", 40, "filt", False, "kp13-w40-filt"),
        ("kp13", 60, "filt", False, "kp13-w60-filt"),
    ]
    labels, h64_vals, h128_vals = [], [], []
    for kp, w, prep, vel, label in pairs:
        base = df[(df.kp == kp) & (df.window == w) & (df.prep == prep) &
                  (df.vel == vel) & (df.model == "gru")]
        h64 = base[base.hidden == "[64, 32]"].ev_minpr.values
        h128 = base[base.hidden == "[128, 64]"].ev_minpr.values
        if len(h64) and len(h128):
            labels.append(label)
            h64_vals.append(float(h64[0]))
            h128_vals.append(float(h128[0]))

    if not labels:
        print("  [fig5] hidden_size: 비교 데이터 부족, skip")
        return

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - w/2, h64_vals, w, label="GRU 64/32", color="#4C72B0")
    b2 = ax.bar(x + w/2, h128_vals, w, label="GRU 128/64", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Config")
    ax.set_ylabel("Event-level MinPR")
    ax.set_title("Hidden Size Effect\n(GRU 64/32 vs 128/64)")
    all_vals = h64_vals + h128_vals
    ax.set_ylim(min(all_vals) - 0.01, max(all_vals) + 0.01)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    for bar, v in zip(list(b1) + list(b2), h64_vals + h128_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.0003, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig5_hidden_size.png")
    plt.close(fig)
    print("  [fig5] hidden_size saved")


def bar_tcn_vs_gru(df: pd.DataFrame, out: pathlib.Path):
    """TCN vs GRU"""
    pairs = [
        ("kp13", 40, "filt", False, "kp13-w40-filt"),
        ("kp7",  40, "filt", False, "kp7-w40-filt"),
    ]
    labels, gru_vals, tcn_vals = [], [], []
    for kp, w, prep, vel, label in pairs:
        base = df[(df.kp == kp) & (df.window == w) & (df.prep == prep) &
                  (df.vel == vel) & (df.hidden == "[64, 32]")]
        gv = base[base.model == "gru"].ev_minpr.values
        tv = base[base.model == "tcn"].ev_minpr.values
        if len(gv) and len(tv):
            labels.append(label)
            gru_vals.append(float(gv[0]))
            tcn_vals.append(float(tv[0]))

    if not labels:
        print("  [fig6] tcn_vs_gru: 비교 데이터 부족, skip")
        return

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5, 4))
    b1 = ax.bar(x - w/2, gru_vals, w, label="GRU", color="#4C72B0")
    b2 = ax.bar(x + w/2, tcn_vals, w, label="TCN", color="#8172B2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Config")
    ax.set_ylabel("Event-level MinPR")
    ax.set_title("Model Architecture: GRU vs TCN")
    all_vals = gru_vals + tcn_vals
    ax.set_ylim(min(all_vals) - 0.01, max(all_vals) + 0.01)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    for bar, v in zip(list(b1) + list(b2), gru_vals + tcn_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.0003, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig6_tcn_vs_gru.png")
    plt.close(fig)
    print("  [fig6] tcn_vs_gru saved")


def save_table(df: pd.DataFrame, out: pathlib.Path):
    """통합 결과 테이블 CSV + 논문용 요약 텍스트"""
    cols = ["id", "phase", "kp", "window", "prep", "vel", "model", "hidden",
            "win_minpr", "ev_minpr", "flash_kib", "act_kib", "macc", "int8_minpr"]
    out_df = df[cols].sort_values("ev_minpr", ascending=False)
    csv_path = out / "results_table.csv"
    out_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  [table] saved {csv_path}")

    # 논문용 요약 (상위 10개)
    txt_path = out / "top10_summary.txt"
    with open(txt_path, "w") as f:
        f.write("P41+P42 Combined Results — Top 10 by Event MinPR\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'ID':30s}  {'kp':5s}  {'w':3s}  {'prep':4s}  {'vel':3s}  "
                f"{'hidden':12s}  {'ev_minpr':8s}  {'flash':6s}  {'int8':6s}\n")
        f.write("-" * 80 + "\n")
        for _, r in out_df.head(10).iterrows():
            flash = f"{r.flash_kib:.0f}K" if pd.notna(r.flash_kib) else "-"
            int8 = f"{r.int8_minpr:.4f}" if pd.notna(r.int8_minpr) else "-"
            f.write(f"{r.id:30s}  {r.kp:5s}  {r.window:<3}  {r.prep:4s}  "
                    f"{str(r.vel):3s}  {r.hidden:12s}  {r.ev_minpr:.4f}    "
                    f"{flash:6s}  {int8}\n")
    print(f"  [table] top10 saved {txt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/phase43_report")
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_all(repo)
    print(f"  Loaded {len(df)} experiments")

    if df.empty:
        print("No data found. Exiting.")
        sys.exit(1)

    print("Generating figures...")
    bar_feature_set(df, out)
    line_window_size(df, out)
    bar_raw_vs_filt(df, out)
    bar_velocity(df, out)
    bar_hidden_size(df, out)
    bar_tcn_vs_gru(df, out)

    print("Saving tables...")
    save_table(df, out)

    print(f"\nDone. Output: {out}/")
    print("  fig1_feature_set.png")
    print("  fig2_window_size.png")
    print("  fig3_raw_vs_filt.png")
    print("  fig4_velocity.png")
    print("  fig5_hidden_size.png")
    print("  fig6_tcn_vs_gru.png")
    print("  results_table.csv")
    print("  top10_summary.txt")


if __name__ == "__main__":
    main()
