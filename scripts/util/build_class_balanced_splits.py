#!/usr/bin/env python3
"""Build class-balanced stratified splits from final_dataset.csv.

Video-level stratified split (7:2:1) by direction × label.
Cameras C5-C8 only (matching canonical splits_v2 convention).

Outputs:
  dataset/splits_v2_class_balanced/{train,val,test}.csv   — raw kp + derived
  dataset/splits_v2_class_balanced_filtered/{train,val,test}.csv — One-Euro+EMA + AHSSC + label_3class

Usage:
    uv run python scripts/util/build_class_balanced_splits.py \
        [--src /home/min/Downloads/final_dataset.csv] \
        [--cameras C5,C6,C7,C8]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SRC = Path("/home/min/Downloads/final_dataset.csv")
DEFAULT_CAMERAS = {"C5", "C6", "C7", "C8"}

N_KP = 17
KP_Y_COLS = [f"kp{i}_y" for i in range(N_KP)]
KP_X_COLS = [f"kp{i}_x" for i in range(N_KP)]
KP_S_COLS = [f"kp{i}_s" for i in range(N_KP)]
HSSC_KP_IDX = list(range(7))
HSSC_Y_COLS = [f"kp{i}_y" for i in HSSC_KP_IDX]
HSSC_X_COLS = [f"kp{i}_x" for i in HSSC_KP_IDX]


# ── Filters ──────────────────────────────────────────────────────────────────

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=0.5, beta=0.3, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

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
        dx = (x - self.x_prev) / t_e
        a_d = self._alpha(t_e, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class EMAFilter:
    def __init__(self, x0, alpha=0.5):
        self.alpha = alpha
        self.x_prev = x0 if not math.isnan(x0) else 0.0

    def __call__(self, x):
        if math.isnan(x):
            return self.x_prev
        x_hat = self.alpha * x + (1.0 - self.alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat


# ── Feature computation ───────────────────────────────────────────────────────

def _process_video_filtered(group: pd.DataFrame,
                             min_cutoff=0.5, beta=0.3, d_cutoff=1.0,
                             conf_alpha=0.5, conf_thr=0.15,
                             ema_deriv_alpha=0.4) -> pd.DataFrame:
    g = group.sort_values("time_sec").copy()
    times = g["time_sec"].to_numpy()

    # 1. Low-confidence masking → linear interpolation → clip [0,1]
    for i in range(N_KP):
        s_col = f"kp{i}_s"
        if s_col not in g.columns:
            continue
        low = g[s_col].to_numpy(dtype=float) < conf_thr
        for coord in (f"kp{i}_y", f"kp{i}_x"):
            if coord not in g.columns:
                continue
            v = g[coord].to_numpy(dtype=float).copy()
            v[low] = np.nan
            nans = np.isnan(v)
            if nans.any() and (~nans).any():
                idx = np.arange(len(v))
                v[nans] = np.interp(idx[nans], idx[~nans], v[~nans])
            g[coord] = np.clip(v, 0.0, 1.0)

    # 2. One-Euro filter on kp coordinates
    for col in KP_Y_COLS + KP_X_COLS:
        if col not in g.columns:
            continue
        v = g[col].to_numpy(dtype=float).copy()
        valid = ~np.isnan(v)
        if not valid.any():
            continue
        first = int(np.argmax(valid))
        f = OneEuroFilter(times[first], v[first], min_cutoff, beta, d_cutoff)
        for idx in range(first, len(v)):
            v[idx] = f(times[idx], v[idx])
        g[col] = v

    # 3. EMA smooth confidence scores
    for col in KP_S_COLS:
        if col not in g.columns:
            continue
        v = g[col].to_numpy(dtype=float).copy()
        x0 = 0.0 if np.isnan(v[0]) else v[0]
        f = EMAFilter(x0, alpha=conf_alpha)
        for idx in range(len(v)):
            v[idx] = f(v[idx])
        g[col] = v

    # 4. Derive features from filtered kp
    y_h = [c for c in HSSC_Y_COLS if c in g.columns]
    x_h = [c for c in HSSC_X_COLS if c in g.columns]
    if y_h:
        g["HSSC_y"] = g[y_h].mean(axis=1)
    if x_h:
        g["HSSC_x"] = g[x_h].mean(axis=1)

    y_all = [c for c in KP_Y_COLS if c in g.columns]
    x_all = [c for c in KP_X_COLS if c in g.columns]
    if y_all and x_all:
        width = g[x_all].max(axis=1) - g[x_all].min(axis=1)
        height = g[y_all].max(axis=1) - g[y_all].min(axis=1)
        g["RWHC"] = width / height.replace(0.0, 1e-4)

    if "HSSC_y" in g.columns:
        g["VHSSC"] = np.gradient(g["HSSC_y"].values, times)

    # 5. EMA smooth VHSSC before second derivative
    if "VHSSC" in g.columns and ema_deriv_alpha > 0:
        g["VHSSC"] = g["VHSSC"].ewm(alpha=ema_deriv_alpha, adjust=False).mean()

    # 6. Acceleration features
    if "VHSSC" in g.columns:
        g["AHSSC"] = np.gradient(g["VHSSC"].values, times)
    if "HSSC_x" in g.columns:
        vhssc_x = np.gradient(g["HSSC_x"].values, times)
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


# ── Split logic ───────────────────────────────────────────────────────────────

def stratified_split(video_meta: pd.DataFrame,
                     train_ratio=0.7, val_ratio=0.2, seed=42):
    """Video-level stratified split by stratum (direction × label).
    Returns (train_vids, val_vids, test_vids).
    """
    train_vids, val_vids, test_vids = [], [], []
    for stratum, grp in video_meta.groupby("stratum"):
        vids = grp["video_id"].tolist()
        if len(vids) < 3:
            # too few — all go to train
            train_vids.extend(vids)
            continue
        n_test = max(1, round(len(vids) * (1 - train_ratio - val_ratio)))
        n_val = max(1, round(len(vids) * val_ratio))
        n_train = len(vids) - n_val - n_test
        if n_train < 1:
            n_train, n_val, n_test = len(vids), 0, 0
        tv, test = train_test_split(vids, test_size=n_test, random_state=seed)
        if n_val > 0 and len(tv) > 1:
            tr, val = train_test_split(tv, test_size=n_val, random_state=seed)
        else:
            tr, val = tv, []
        train_vids.extend(tr)
        val_vids.extend(val)
        test_vids.extend(test)
    return set(train_vids), set(val_vids), set(test_vids)


def save_split_summary(dst: Path, splits: dict[str, pd.DataFrame]) -> None:
    import json
    summary = {}
    for name, df in splits.items():
        vids = df["video_id"].unique()
        summary[name] = {
            "videos": int(len(vids)),
            "rows": int(len(df)),
        }
        if "direction" in df.columns:
            summary[name]["directions"] = df.groupby("video_id")["direction"].first().value_counts().to_dict()
        if "camera" in df.columns:
            summary[name]["cameras"] = df.groupby("video_id")["camera"].first().value_counts().to_dict()
    (dst / "split_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  Saved summary → {dst / 'split_summary.json'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--cameras", default="C5,C6,C7,C8")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-filtered", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    cameras = set(args.cameras.split(","))
    seed = args.seed

    raw_dst = REPO / "dataset" / "splits_v2_class_balanced"
    flt_dst = REPO / "dataset" / "splits_v2_class_balanced_filtered"

    print(f"Reading {src} …")
    df = pd.read_csv(src, low_memory=False)
    df["video_id"] = df["video_id"].astype(str)
    print(f"  {len(df):,} rows, {df['video_id'].nunique():,} videos")

    # Parse direction and camera from video_id
    parts = df["video_id"].str.split("_")
    df["direction"] = parts.str[3]
    df["camera"] = parts.str[4]

    # Filter cameras
    before = df["video_id"].nunique()
    df = df[df["camera"].isin(cameras)].copy()
    print(f"  After camera filter ({','.join(sorted(cameras))}): {df['video_id'].nunique():,} / {before:,} videos")

    # Build video-level metadata for stratification
    vid_label = df.groupby("video_id").agg(
        label=("label", "max"),
        direction=("direction", "first"),
    ).reset_index()
    vid_label["stratum"] = vid_label["direction"] + "_" + vid_label["label"].astype(str)

    print("\nStratified split …")
    train_vids, val_vids, test_vids = stratified_split(vid_label, seed=seed)
    print(f"  train={len(train_vids)}, val={len(val_vids)}, test={len(test_vids)}")

    def assign_split(vid):
        if vid in train_vids:
            return "train"
        if vid in val_vids:
            return "val"
        return "test"

    df["_split"] = df["video_id"].map(assign_split)

    # ── Raw output ────────────────────────────────────────────────────────────
    print(f"\nSaving raw splits → {raw_dst}")
    raw_dst.mkdir(parents=True, exist_ok=True)

    col_order = (["video_id", "frame", "time_sec"]
                 + [f"kp{i}_{ax}" for i in range(N_KP) for ax in ("y", "x", "s")]
                 + ["HSSC_y", "HSSC_x", "RWHC", "VHSSC", "label", "direction", "camera"])
    col_order = [c for c in col_order if c in df.columns]

    raw_splits = {}
    for split in ("train", "val", "test"):
        sub = df[df["_split"] == split][col_order].copy()
        sub = sub.sort_values(["video_id", "time_sec"]).reset_index(drop=True)
        out_path = raw_dst / f"{split}.csv"
        sub.to_csv(out_path, index=False)
        raw_splits[split] = sub
        print(f"  {split}: {len(sub):,} rows, {sub['video_id'].nunique():,} videos → {out_path}")
    save_split_summary(raw_dst, raw_splits)

    if args.skip_filtered:
        print("\nSkipped filtered build (--skip-filtered).")
        return

    # ── Filtered output ────────────────────────────────────────────────────────
    print(f"\nBuilding filtered splits (One-Euro+EMA) → {flt_dst}")
    flt_dst.mkdir(parents=True, exist_ok=True)

    # Keep only raw kp + meta + label for filtering
    raw_cols = (["video_id", "frame", "time_sec", "label", "direction", "camera"]
                + [f"kp{i}_{ax}" for i in range(N_KP) for ax in ("y", "x", "s")])
    raw_cols = [c for c in raw_cols if c in df.columns]

    flt_col_order = (["video_id", "frame", "time_sec", "label"]
                     + [f"kp{i}_{ax}" for i in range(N_KP) for ax in ("y", "x", "s")]
                     + ["HSSC_y", "HSSC_x", "RWHC", "VHSSC", "AHSSC", "AHSSC_x", "label_3class"])

    flt_splits = {}
    for split in ("train", "val", "test"):
        sub = df[df["_split"] == split][raw_cols].copy()
        sub = sub.sort_values(["video_id", "time_sec"]).reset_index(drop=True)

        groups = sub.groupby("video_id", sort=False)
        results = []
        iterator = tqdm(groups, total=groups.ngroups, unit="vid", desc=split) if HAS_TQDM else groups
        for _, grp in iterator:
            results.append(_process_video_filtered(grp))

        out = pd.concat(results, ignore_index=True)
        out = add_label_3class(out)
        out = out.sort_values(["video_id", "time_sec"]).reset_index(drop=True)

        keep = [c for c in flt_col_order if c in out.columns]
        out = out[keep]
        out_path = flt_dst / f"{split}.csv"
        out.to_csv(out_path, index=False)
        flt_splits[split] = out
        print(f"  {split}: {len(out):,} rows, {out['video_id'].nunique():,} videos → {out_path}")

    save_split_summary(flt_dst, flt_splits)
    print("\nDone.")


if __name__ == "__main__":
    main()
