#!/usr/bin/env python3
"""Build intermediate filter-stage datasets for Experiment 2.

Creates 4 datasets corresponding to filter pipeline stages A-D:
  A: Raw (no filter) — already exists as splits_v2_class_balanced
  B: One-Euro filter only (no EMA, no derived features)
  C: One-Euro + EMA confidence smoothing (no derived features)
  D: One-Euro + EMA + derived features — already exists as splits_v2_class_balanced_filtered

This script builds stages B and C, using splits_v2_class_balanced as source.
Output: dataset/splits_v2_class_balanced_filter_B/ and _filter_C/

Usage:
    uv run python scripts/util/build_filter_stage_datasets.py [--stage B|C|both]
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

REPO     = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO / "dataset/splits_v2_class_balanced"
DST_B    = REPO / "dataset/splits_v2_class_balanced_filter_B"
DST_C    = REPO / "dataset/splits_v2_class_balanced_filter_C"

N_KP = 17
KP_Y_COLS = [f"kp{i}_y" for i in range(N_KP)]
KP_X_COLS = [f"kp{i}_x" for i in range(N_KP)]
KP_S_COLS = [f"kp{i}_s" for i in range(N_KP)]


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
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / t_e)

    def __call__(self, t, x):
        t_e   = max(t - self.t_prev, 1e-6)
        alpha_d = self._alpha(t_e, self.d_cutoff)
        dx    = (x - self.x_prev) / t_e
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha  = self._alpha(t_e, cutoff)
        x_hat  = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat


class EMAFilter:
    def __init__(self, x0, alpha=0.5):
        self.alpha = alpha
        self.prev  = x0

    def __call__(self, x):
        out = self.alpha * x + (1 - self.alpha) * self.prev
        self.prev = out
        return out


def process_video_stage_b(group: pd.DataFrame,
                           min_cutoff=0.5, beta=0.3, d_cutoff=1.0,
                           conf_thr=0.15) -> pd.DataFrame:
    """Stage B: One-Euro filter on coordinates only (no EMA, no derived features)."""
    g     = group.sort_values("time_sec").copy()
    times = g["time_sec"].to_numpy()

    # Low-confidence masking → linear interpolation
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

    # One-Euro filter on kp coordinates
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

    return g


def process_video_stage_c(group: pd.DataFrame,
                           min_cutoff=0.5, beta=0.3, d_cutoff=1.0,
                           conf_alpha=0.5, conf_thr=0.15) -> pd.DataFrame:
    """Stage C: One-Euro + EMA confidence smoothing (no derived features)."""
    g = process_video_stage_b(group, min_cutoff, beta, d_cutoff, conf_thr)

    # EMA smooth confidence scores
    for col in KP_S_COLS:
        if col not in g.columns:
            continue
        v  = g[col].to_numpy(dtype=float).copy()
        x0 = 0.0 if np.isnan(v[0]) else v[0]
        f  = EMAFilter(x0, alpha=conf_alpha)
        for idx in range(len(v)):
            v[idx] = f(v[idx])
        g[col] = v

    return g


def process_split(src: Path, dst: Path, stage: str) -> None:
    print(f"  Processing {src.name} → {dst.name}/{src.name} (stage {stage})")
    df = pd.read_csv(src)
    vids = df["video_id"].unique()
    print(f"    {len(vids):,} videos, {len(df):,} rows")

    if HAS_TQDM:
        it = tqdm(df.groupby("video_id", sort=False), total=len(vids))
    else:
        it = df.groupby("video_id", sort=False)

    processed = []
    for vid, grp in it:
        if stage == "B":
            g = process_video_stage_b(grp)
        else:
            g = process_video_stage_c(grp)
        processed.append(g)

    out = pd.concat(processed, ignore_index=True)
    dst.mkdir(parents=True, exist_ok=True)
    out_path = dst / src.name
    out.to_csv(out_path, index=False)
    print(f"    Saved: {out_path} ({len(out):,} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["B", "C", "both"], default="both")
    args = ap.parse_args()

    stages = ["B", "C"] if args.stage == "both" else [args.stage]
    dst_map = {"B": DST_B, "C": DST_C}

    for split in ["train.csv", "val.csv", "test.csv"]:
        src = SRC_ROOT / split
        if not src.exists():
            print(f"  SKIP — {src} not found")
            continue
        for stage in stages:
            process_split(src, dst_map[stage], stage)

    print("\nDone.")
    print(f"Stage B (One-Euro only) → {DST_B}")
    print(f"Stage C (One-Euro+EMA)  → {DST_C}")
    print("Stage A (raw)           → dataset/splits_v2_class_balanced  (already exists)")
    print("Stage D (all features)  → dataset/splits_v2_class_balanced_filtered  (already exists)")


if __name__ == "__main__":
    main()
