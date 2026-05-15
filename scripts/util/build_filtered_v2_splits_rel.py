#!/usr/bin/env python3
"""Build splits_v2_filtered_rel with pose-relative kp7 features.

This reuses the PP-D filtered coordinate pipeline from build_filtered_v2_splits.py,
then adds torso/hip-centered relative features intended to reduce dependence on
absolute image position and center-of-mass motion.

Output: dataset/splits_v2_filtered_rel/{train,val,test}.csv

Usage:
    uv run python scripts/util/build_filtered_v2_splits_rel.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_filtered_v2_splits import (
    HAS_TQDM,
    LABEL_COLS,
    META_COLS,
    N_KP,
    RAW_KP_COLS,
    _process_video,
    add_label_3class,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


EPS = 1e-4
KP7_REL_IDX = [0, 5, 6, 7, 8, 11, 12]


def _mean_pair(g: pd.DataFrame, a: str, b: str) -> pd.Series:
    return (g[a].astype(float) + g[b].astype(float)) * 0.5


def add_pose_relative_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()

    required = [
        "kp0_y", "kp0_x",
        "kp5_y", "kp5_x", "kp6_y", "kp6_x",
        "kp11_y", "kp11_x", "kp12_y", "kp12_x",
    ]
    missing = [col for col in required if col not in g.columns]
    if missing:
        raise ValueError(f"Cannot build pose-relative features; missing columns: {missing}")

    shoulder_y = _mean_pair(g, "kp5_y", "kp6_y")
    shoulder_x = _mean_pair(g, "kp5_x", "kp6_x")
    hip_y = _mean_pair(g, "kp11_y", "kp12_y")
    hip_x = _mean_pair(g, "kp11_x", "kp12_x")

    torso_dy = hip_y - shoulder_y
    torso_dx = hip_x - shoulder_x
    torso_len = np.sqrt(np.square(torso_dy) + np.square(torso_dx)).clip(lower=EPS)

    g["torso_dx"] = torso_dx
    g["torso_dy"] = torso_dy
    g["torso_len"] = torso_len
    g["torso_angle_sin"] = torso_dy / torso_len
    g["torso_angle_cos"] = torso_dx / torso_len
    g["head_hip_dy"] = (g["kp0_y"].astype(float) - hip_y) / torso_len
    g["head_hip_dx"] = (g["kp0_x"].astype(float) - hip_x) / torso_len

    shoulder_width = (g["kp6_x"].astype(float) - g["kp5_x"].astype(float)).abs()
    hip_width = (g["kp12_x"].astype(float) - g["kp11_x"].astype(float)).abs()
    g["shoulder_width_norm"] = shoulder_width / torso_len
    g["hip_width_norm"] = hip_width / torso_len
    g["shoulder_hip_width_ratio"] = shoulder_width / hip_width.clip(lower=EPS)

    y_cols = [f"kp{i}_y" for i in range(N_KP) if f"kp{i}_y" in g.columns]
    x_cols = [f"kp{i}_x" for i in range(N_KP) if f"kp{i}_x" in g.columns]
    body_height = (g[y_cols].max(axis=1) - g[y_cols].min(axis=1)).astype(float).clip(lower=EPS)
    body_width = (g[x_cols].max(axis=1) - g[x_cols].min(axis=1)).astype(float)
    g["body_height_norm"] = body_height / torso_len
    g["body_width_norm"] = body_width / torso_len
    g["body_aspect_ratio"] = body_width / body_height

    times = g["time_sec"].to_numpy(dtype=float)
    if len(g) > 1:
        g["body_aspect_velocity"] = np.gradient(g["body_aspect_ratio"].to_numpy(dtype=float), times)
    else:
        g["body_aspect_velocity"] = 0.0

    for idx in KP7_REL_IDX:
        g[f"kp{idx}_rel_y"] = (g[f"kp{idx}_y"].astype(float) - hip_y) / torso_len
        g[f"kp{idx}_rel_x"] = (g[f"kp{idx}_x"].astype(float) - hip_x) / torso_len

    rel_cols = [col for col in g.columns if "_rel_" in col or col in {
        "torso_dx",
        "torso_dy",
        "torso_len",
        "torso_angle_sin",
        "torso_angle_cos",
        "head_hip_dx",
        "head_hip_dy",
        "shoulder_width_norm",
        "hip_width_norm",
        "shoulder_hip_width_ratio",
        "body_height_norm",
        "body_width_norm",
        "body_aspect_ratio",
        "body_aspect_velocity",
    }]
    g[rel_cols] = g[rel_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return g


def process_split(src: Path, dst: Path) -> None:
    print(f"  Loading {src.name} ...", end=" ", flush=True)
    raw = pd.read_csv(src, low_memory=False)
    raw["video_id"] = raw["video_id"].astype(str)

    keep = [c for c in META_COLS + LABEL_COLS + RAW_KP_COLS if c in raw.columns]
    df = raw[keep].copy()
    print(f"{len(df):,} rows, {df['video_id'].nunique():,} videos")

    groups = df.groupby("video_id", sort=False)
    results = []
    iterator = tqdm(groups, total=groups.ngroups, unit="video") if HAS_TQDM and tqdm else groups
    for _, grp in iterator:
        filtered = _process_video(grp)
        results.append(add_pose_relative_features(filtered))

    out = pd.concat(results, ignore_index=True)
    out = out.sort_values(["video_id", "time_sec"]).reset_index(drop=True)
    out = add_label_3class(out)

    n0 = int((out["label"] == 0).sum())
    n1 = int((out["label"] == 1).sum())
    n2 = int((out["label_3class"] == 2).sum())
    rel_count = len([c for c in out.columns if "_rel_" in c])
    print(f"    -> {dst}  (normal={n0:,}  falling={n1:,}  fallen={n2:,}  rel_cols={rel_count})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "dataset"
    src_dir = root / "splits_v2"
    dst_dir = root / "splits_v2_filtered_rel"

    print(f"Source : {src_dir}")
    print(f"Output : {dst_dir}")
    for split in ("train", "val", "test"):
        src = src_dir / f"{split}.csv"
        dst = dst_dir / f"{split}.csv"
        if not src.exists():
            print(f"  SKIP {split} - {src} not found")
            continue
        print(f"\n[{split}]")
        process_split(src, dst)

    print("\nDone.")


if __name__ == "__main__":
    main()
