#!/usr/bin/env python3
"""Apply Pipeline D (PP-D) to splits_v2 to produce splits_v2_filtered.

Loads ONLY raw kp + meta + label columns from splits_v2 (discards any
pre-computed features), then runs the same One-Euro + EMA filter pipeline
as build_filtered_dataset.py and computes all derived features fresh from
the filtered coordinates.

Output: dataset/splits_v2_filtered/{train,val,test}.csv
Both `label` (binary) and `label_3class` columns are included.

Usage:
    uv run python scripts/util/build_filtered_v2_splits.py
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Filter implementations (identical params to build_filtered_dataset.py) ───

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=0.5, beta=0.3, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev     = x0
        self.dx_prev    = 0.0
        self.t_prev     = t0

    @staticmethod
    def _alpha(t_e, cutoff):
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def __call__(self, t, x):
        if math.isnan(x):
            return self.x_prev
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return self.x_prev
        dx     = (x - self.x_prev) / t_e
        a_d    = self._alpha(t_e, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a      = self._alpha(t_e, cutoff)
        x_hat  = a * x + (1.0 - a) * self.x_prev
        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat


class EMAFilter:
    def __init__(self, x0, alpha=0.5):
        self.alpha  = alpha
        self.x_prev = x0 if not math.isnan(x0) else 0.0

    def __call__(self, x):
        if math.isnan(x):
            return self.x_prev
        x_hat       = self.alpha * x + (1.0 - self.alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat


N_KP        = 17
KP_Y_COLS   = [f"kp{i}_y" for i in range(N_KP)]
KP_X_COLS   = [f"kp{i}_x" for i in range(N_KP)]
KP_S_COLS   = [f"kp{i}_s" for i in range(N_KP)]
HSSC_KP_IDX = list(range(7))
HSSC_Y_COLS = [f"kp{i}_y" for i in HSSC_KP_IDX]
HSSC_X_COLS = [f"kp{i}_x" for i in HSSC_KP_IDX]

# Columns to keep from splits_v2 before running the pipeline
META_COLS  = ["video_id", "frame", "time_sec"]
LABEL_COLS = ["label"]          # label_3class is re-derived after
RAW_KP_COLS = [f"kp{i}_{ax}" for i in range(N_KP) for ax in ("y", "x", "s")]


def _process_video(group: pd.DataFrame,
                   min_cutoff=0.5, beta=0.3, d_cutoff=1.0,
                   conf_alpha=0.5, conf_thr=0.15, ema_deriv_alpha=0.4) -> pd.DataFrame:
    g     = group.sort_values("time_sec").copy()
    times = g["time_sec"].to_numpy()

    # 1. Low-confidence masking → linear interpolation → clip to [0,1]
    for i in range(N_KP):
        s_col = f"kp{i}_s"
        if s_col not in g.columns:
            continue
        low = g[s_col].to_numpy(dtype=float) < conf_thr
        for coord in (f"kp{i}_y", f"kp{i}_x"):
            if coord not in g.columns:
                continue
            v      = g[coord].to_numpy(dtype=float).copy()
            v[low] = np.nan
            nans   = np.isnan(v)
            if nans.any() and (~nans).any():
                idx    = np.arange(len(v))
                v[nans] = np.interp(idx[nans], idx[~nans], v[~nans])
            g[coord] = np.clip(v, 0.0, 1.0)

    # 2. One-Euro filter on kp coordinates
    for col in KP_Y_COLS + KP_X_COLS:
        if col not in g.columns:
            continue
        v     = g[col].to_numpy(dtype=float).copy()
        valid = ~np.isnan(v)
        if not valid.any():
            continue
        first = int(np.argmax(valid))
        f     = OneEuroFilter(times[first], v[first], min_cutoff, beta, d_cutoff)
        for idx in range(first, len(v)):
            v[idx] = f(times[idx], v[idx])
        g[col] = v

    # 3. EMA smooth confidence scores
    for col in KP_S_COLS:
        if col not in g.columns:
            continue
        v  = g[col].to_numpy(dtype=float).copy()
        x0 = 0.0 if np.isnan(v[0]) else v[0]
        f  = EMAFilter(x0, alpha=conf_alpha)
        for idx in range(len(v)):
            v[idx] = f(v[idx])
        g[col] = v

    # 4. Derive features from filtered kp (computed fresh, not adjusted)
    y_h = [c for c in HSSC_Y_COLS if c in g.columns]
    x_h = [c for c in HSSC_X_COLS if c in g.columns]
    if y_h:
        g["HSSC_y"] = g[y_h].mean(axis=1)
    if x_h:
        g["HSSC_x"] = g[x_h].mean(axis=1)

    y_all = [c for c in KP_Y_COLS if c in g.columns]
    x_all = [c for c in KP_X_COLS if c in g.columns]
    if y_all and x_all:
        width        = g[x_all].max(axis=1) - g[x_all].min(axis=1)
        height       = g[y_all].max(axis=1) - g[y_all].min(axis=1)
        g["RWHC"]    = width / height.replace(0.0, 1e-4)

    if "HSSC_y" in g.columns:
        g["VHSSC"] = np.gradient(g["HSSC_y"].values, times)

    # 5. EMA smooth VHSSC before taking second derivative
    if "VHSSC" in g.columns and ema_deriv_alpha > 0:
        g["VHSSC"] = g["VHSSC"].ewm(alpha=ema_deriv_alpha, adjust=False).mean()

    # 6. Acceleration features
    if "VHSSC" in g.columns:
        g["AHSSC"] = np.gradient(g["VHSSC"].values, times)
    if "HSSC_x" in g.columns:
        vhssc_x      = np.gradient(g["HSSC_x"].values, times)
        g["AHSSC_x"] = np.gradient(vhssc_x, times)

    return g


def add_label_3class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_3class"] = df["label"].astype("int8")
    for vid, grp in df.groupby("video_id"):
        fall_idx = grp.index[grp["label"] == 1]
        if len(fall_idx) > 0:
            after = grp.index[grp.index > fall_idx.max()]
            df.loc[after, "label_3class"] = 2
    return df


def process_split(src: Path, dst: Path) -> None:
    print(f"  Loading {src.name} ...", end=" ", flush=True)
    raw = pd.read_csv(src, low_memory=False)
    raw["video_id"] = raw["video_id"].astype(str)

    # Keep ONLY raw kp + meta + label — discard any pre-computed derived features
    keep = [c for c in META_COLS + LABEL_COLS + RAW_KP_COLS if c in raw.columns]
    df   = raw[keep].copy()
    print(f"{len(df):,} rows, {df['video_id'].nunique():,} videos")

    groups  = df.groupby("video_id", sort=False)
    results = []
    iterator = tqdm(groups, total=groups.ngroups, unit="video") if HAS_TQDM else groups
    for _, grp in iterator:
        results.append(_process_video(grp))

    out = pd.concat(results, ignore_index=True)
    out = out.sort_values(["video_id", "time_sec"]).reset_index(drop=True)
    out = add_label_3class(out)

    n0 = int((out["label"] == 0).sum())
    n1 = int((out["label"] == 1).sum())
    n2 = int((out["label_3class"] == 2).sum())
    print(f"    → {dst}  (normal={n0:,}  falling={n1:,}  fallen={n2:,})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Pipeline D filtering to split CSV files.")
    root = Path(__file__).resolve().parents[2] / "dataset"
    parser.add_argument("--src-dir", type=Path, default=root / "splits_v2")
    parser.add_argument("--dst-dir", type=Path, default=root / "splits_v2_filtered")
    args = parser.parse_args()

    src_dir = args.src_dir
    dst_dir = args.dst_dir

    print(f"Source : {src_dir}")
    print(f"Output : {dst_dir}")
    for split in ("train", "val", "test"):
        src = src_dir / f"{split}.csv"
        dst = dst_dir / f"{split}.csv"
        if not src.exists():
            print(f"  SKIP {split} — {src} not found")
            continue
        print(f"\n[{split}]")
        process_split(src, dst)

    print("\nDone.")


if __name__ == "__main__":
    main()
