#!/usr/bin/env python3
"""
build_v2_dataset.py

Dataset V2 Pipeline:
  1. Filter out low-confidence cameras (C1, C2, C3, C4)
  2. Remove video-level outliers based on confidence and stability
  3. Stratified split (7:2:1) by direction (BY, FY, SY, N)
  4. Save to dataset/splits_v2/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────────
KP_S_COLS = [f"kp{i}_s" for i in range(17)]
DIRECTIONS = ["BY", "FY", "SY", "N"]
REMOVED_CAMERAS = ["C1", "C2", "C3", "C4"]

def get_direction(vid: str) -> str:
    parts = vid.split("_")
    # Format: {ID}_{Subject}_{Action}_{Direction}_{Camera}
    # Example: 00050_H_A_BY_C2 -> index 3 is BY
    return parts[3] if len(parts) >= 4 else "UNKNOWN"

def get_camera(vid: str) -> str:
    return vid.split("_")[-1]

def load_and_filter_cameras(path: Path) -> pd.DataFrame:
    print(f"[1/4] Loading and filtering cameras from {path}...")
    df = pd.read_csv(path, low_memory=False)
    
    # Extract meta
    df["direction"] = df["video_id"].apply(get_direction)
    df["camera"] = df["video_id"].apply(get_camera)
    
    initial_count = len(df)
    initial_vids = df["video_id"].nunique()
    
    # Filter C1~C4
    df = df[~df["camera"].isin(REMOVED_CAMERAS)].copy()
    
    filtered_count = len(df)
    filtered_vids = df["video_id"].nunique()
    
    print(f"      Removed cameras: {REMOVED_CAMERAS}")
    print(f"      Rows: {initial_count:,} -> {filtered_count:,} ({(filtered_count/initial_count)*100:.1f}%)")
    print(f"      Videos: {initial_vids:,} -> {filtered_vids:,} ({(filtered_vids/initial_vids)*100:.1f}%)")
    
    return df

def remove_outliers(df: pd.DataFrame, z_thresh: float = 2.5, min_conf: float = 0.30) -> pd.DataFrame:
    """
    Remove video-level outliers. 
    z_thresh is adjusted higher (default 2.5) to be more conservative (lower outlier ratio).
    """
    print(f"[2/4] Removing outliers (z_thresh={z_thresh}, min_conf={min_conf})...")
    
    grp = df.groupby("video_id")
    video_stats = pd.DataFrame({
        "direction": df.groupby("video_id")["direction"].first(),
        "conf_mean": grp[KP_S_COLS].mean().mean(axis=1),
        "rwhc_std":  grp["RWHC"].std() if "RWHC" in df.columns else 0,
        "vhssc_std": grp["VHSSC"].std() if "VHSSC" in df.columns else 0,
    })
    
    outlier_flags = pd.Series(False, index=video_stats.index)
    
    # 1. Absolute confidence filter
    low_conf = video_stats["conf_mean"] < min_conf
    outlier_flags |= low_conf
    
    # 2. Z-score filter (within each direction)
    for d in DIRECTIONS:
        sub = video_stats[video_stats["direction"] == d]
        if len(sub) < 5: continue
        
        for col in ["conf_mean", "rwhc_std", "vhssc_std"]:
            vals = sub[col].fillna(sub[col].median())
            z = np.abs(stats.zscore(vals))
            outlier_flags.loc[sub.index[z > z_thresh]] = True
            
    bad_vids = set(outlier_flags[outlier_flags].index)
    df_clean = df[~df["video_id"].isin(bad_vids)].copy()
    
    n_removed = len(bad_vids)
    total_vids = len(video_stats)
    print(f"      Removed {n_removed}/{total_vids} videos ({ (n_removed/total_vids)*100:.1f}% outlier ratio)")
    print(f"      Remaining rows: {len(df_clean):,}")
    
    return df_clean

def split_and_save(df: pd.DataFrame, out_dir: Path, ratios: tuple = (0.7, 0.2, 0.1), seed: int = 42):
    print(f"[3/4] Splitting dataset (ratios={ratios}, seed={seed})...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    vid_dir = df.drop_duplicates("video_id").set_index("video_id")["direction"]
    
    train_vids, val_vids, test_vids = [], [], []
    
    for d in DIRECTIONS:
        vids = sorted(vid_dir[vid_dir == d].index.tolist())
        rng.shuffle(vids)
        
        n = len(vids)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        
        train_vids.extend(vids[:n_tr])
        val_vids.extend(vids[n_tr:n_tr + n_va])
        test_vids.extend(vids[n_tr + n_va:])
        
    splits = {
        "train": train_vids,
        "val": val_vids,
        "test": test_vids
    }
    
    stats_summary = {}
    for name, vlist in splits.items():
        sub = df[df["video_id"].isin(vlist)].copy()
        sub.to_csv(out_dir / f"{name}.csv", index=False)
        
        # Stats
        v_count = len(vlist)
        r_count = len(sub)
        d_dist = sub.drop_duplicates("video_id")["direction"].value_counts().to_dict()
        c_dist = sub.drop_duplicates("video_id")["camera"].value_counts().to_dict()
        
        stats_summary[name] = {
            "videos": v_count,
            "rows": r_count,
            "directions": d_dist,
            "cameras": c_dist
        }
        print(f"      {name:5s}: {v_count:4d} videos, {r_count:9,} rows")
        
    return stats_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="dataset/final_dataset.csv")
    parser.add_argument("--out-dir", default="dataset/splits_v2")
    parser.add_argument("--z-thresh", type=float, default=2.5, help="Higher = fewer outliers removed")
    parser.add_argument("--min-conf", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    
    # 1. Load and filter cameras
    df = load_and_filter_cameras(Path(args.input))
    
    # 2. Remove outliers
    df = remove_outliers(df, z_thresh=args.z_thresh, min_conf=args.min_conf)
    
    # 3. Split and Save
    stats_summary = split_and_save(df, out_dir, seed=args.seed)
    
    # 4. Save summary
    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n[4/4] Done. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
