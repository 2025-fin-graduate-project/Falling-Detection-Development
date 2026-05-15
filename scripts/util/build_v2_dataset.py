#!/usr/bin/env python3
"""
build_v2_dataset.py

Dataset V2 Pipeline:
  1. Filter out low-confidence cameras (C1, C2, C3, C4)
  2. Remove video-level outliers based on confidence and stability
  3. Stratified split (7:2:1) by configurable video metadata
  4. Save to dataset/splits_v2/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────────
KP_S_COLS = [f"kp{i}_s" for i in range(17)]
DIRECTIONS = ["BY", "FY", "SY", "N"]
REMOVED_CAMERAS = ["C1", "C2", "C3", "C4"]
SPLIT_NAMES = ("train", "val", "test")

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

def parse_ratios(value: str) -> tuple[float, float, float]:
    ratios = tuple(float(v.strip()) for v in value.split(","))
    if len(ratios) != 3:
        raise ValueError("--ratios must contain exactly three comma-separated values")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("--ratios must sum to a positive value")
    return tuple(v / total for v in ratios)


def parse_positive_labels(value: str) -> set[int | float | str]:
    labels: set[int | float | str] = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            number = float(item)
            labels.add(int(number) if number.is_integer() else number)
        except ValueError:
            labels.add(item)
    if not labels:
        raise ValueError("--positive-labels must contain at least one label")
    return labels


def split_key_columns(value: str) -> list[str]:
    keys = [v.strip() for v in value.split(",") if v.strip()]
    if not keys:
        raise ValueError("--stratify-keys must contain at least one key")
    return keys


def build_video_stats(
    df: pd.DataFrame,
    label_col: str,
    positive_labels: set[int | float | str],
) -> pd.DataFrame:
    if label_col not in df.columns:
        raise KeyError(f"Label column not found: {label_col}")

    is_positive = df[label_col].isin(positive_labels)
    meta = df.groupby("video_id").agg(
        direction=("direction", "first"),
        camera=("camera", "first"),
        rows=("video_id", "size"),
    )
    label_counts = (
        df.assign(_is_positive=is_positive)
        .groupby("video_id")["_is_positive"]
        .agg(pos_rows="sum", neg_rows=lambda s: int((~s).sum()))
    )
    video_stats = meta.join(label_counts)
    video_stats["pos_rows"] = video_stats["pos_rows"].astype(int)
    video_stats["neg_rows"] = video_stats["neg_rows"].astype(int)
    video_stats["video_label"] = np.where(video_stats["pos_rows"] > 0, "fall", "nonfall")
    video_stats["positive_frame_ratio"] = video_stats["pos_rows"] / video_stats["rows"].clip(lower=1)
    return video_stats


def round_robin_split(
    video_stats: pd.DataFrame,
    ratios: tuple[float, float, float],
    seed: int,
    stratify_keys: list[str],
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    splits = {name: [] for name in SPLIT_NAMES}

    grouped = video_stats.groupby(stratify_keys, dropna=False, sort=True)
    for _, group in grouped:
        vids = group.index.to_numpy(copy=True)
        rng.shuffle(vids)

        n = len(vids)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])

        splits["train"].extend(vids[:n_tr].tolist())
        splits["val"].extend(vids[n_tr:n_tr + n_va].tolist())
        splits["test"].extend(vids[n_tr + n_va:].tolist())

    return splits


def greedy_frame_balanced_split(
    video_stats: pd.DataFrame,
    ratios: tuple[float, float, float],
    seed: int,
    stratify_keys: list[str],
    video_weight: float,
    class_weight: float,
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    splits = {name: [] for name in SPLIT_NAMES}

    grouped = video_stats.groupby(stratify_keys, dropna=False, sort=True)
    groups = [(key, group.copy()) for key, group in grouped]
    rng.shuffle(groups)

    for _, group in groups:
        group = group.assign(_shuffle=rng.random(len(group)))
        group = group.sort_values(["rows", "pos_rows", "_shuffle"], ascending=[False, False, True])

        group_total = {
            "videos": len(group),
            "pos_rows": int(group["pos_rows"].sum()),
            "neg_rows": int(group["neg_rows"].sum()),
        }
        group_target = {
            split: {metric: group_total[metric] * ratio for metric in group_total}
            for split, ratio in zip(SPLIT_NAMES, ratios)
        }
        group_state = {
            name: {"videos": 0, "pos_rows": 0, "neg_rows": 0}
            for name in SPLIT_NAMES
        }

        def score(candidate_split: str, row: pd.Series) -> float:
            score_value = 0.0
            for split in SPLIT_NAMES:
                for metric, weight in (
                    ("videos", video_weight),
                    ("pos_rows", class_weight),
                    ("neg_rows", class_weight),
                ):
                    current = group_state[split][metric]
                    if split == candidate_split:
                        current += 1 if metric == "videos" else int(row[metric])
                    target = group_target[split][metric]
                    denom = max(target, 1.0)
                    score_value += weight * ((current - target) / denom) ** 2
            return score_value

        for vid, row in group.iterrows():
            target_split = min(SPLIT_NAMES, key=lambda name: score(name, row))
            splits[target_split].append(vid)
            group_state[target_split]["videos"] += 1
            group_state[target_split]["pos_rows"] += int(row["pos_rows"])
            group_state[target_split]["neg_rows"] += int(row["neg_rows"])

    return splits


def split_and_save(
    df: pd.DataFrame,
    out_dir: Path,
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
    stratify_keys: list[str] | None = None,
    label_col: str = "label",
    positive_labels: set[int | float | str] | None = None,
    balance_frame_labels: bool = False,
    video_weight: float = 1.0,
    class_weight: float = 3.0,
):
    stratify_keys = stratify_keys or ["direction"]
    positive_labels = positive_labels or {1}
    print(f"[3/4] Splitting dataset (ratios={ratios}, seed={seed})...")
    print(f"      Stratify keys: {stratify_keys}")
    print(f"      Frame-label balance: {'on' if balance_frame_labels else 'off'}")
    out_dir.mkdir(parents=True, exist_ok=True)

    video_stats = build_video_stats(df, label_col, positive_labels)
    missing_keys = sorted(set(stratify_keys) - set(video_stats.columns))
    if missing_keys:
        raise KeyError(f"Unknown stratify keys: {missing_keys}. Available: {sorted(video_stats.columns)}")

    if balance_frame_labels:
        splits = greedy_frame_balanced_split(
            video_stats,
            ratios=ratios,
            seed=seed,
            stratify_keys=stratify_keys,
            video_weight=video_weight,
            class_weight=class_weight,
        )
    else:
        splits = round_robin_split(
            video_stats,
            ratios=ratios,
            seed=seed,
            stratify_keys=stratify_keys,
        )
    
    stats_summary = {}
    for name, vlist in splits.items():
        sub = df[df["video_id"].isin(vlist)].copy()
        sub.to_csv(out_dir / f"{name}.csv", index=False)
        
        # Stats
        v_count = len(vlist)
        r_count = len(sub)
        d_dist = sub.drop_duplicates("video_id")["direction"].value_counts().to_dict()
        c_dist = sub.drop_duplicates("video_id")["camera"].value_counts().to_dict()
        split_video_stats = video_stats.loc[vlist]
        pos_rows = int(split_video_stats["pos_rows"].sum())
        neg_rows = int(split_video_stats["neg_rows"].sum())
        video_label_dist = split_video_stats["video_label"].value_counts().to_dict()
        
        stats_summary[name] = {
            "videos": v_count,
            "rows": r_count,
            "directions": d_dist,
            "cameras": c_dist,
            "video_labels": video_label_dist,
            "frame_labels": {
                "positive": pos_rows,
                "negative": neg_rows,
                "positive_ratio": pos_rows / max(pos_rows + neg_rows, 1),
            },
        }
        print(
            f"      {name:5s}: {v_count:4d} videos, {r_count:9,} rows, "
            f"pos_frame_ratio={stats_summary[name]['frame_labels']['positive_ratio']:.4f}"
        )
        
    return stats_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="dataset/final_dataset.csv")
    parser.add_argument("--out-dir", default="dataset/splits_v2")
    parser.add_argument("--ratios", default="0.7,0.2,0.1", help="Comma-separated train,val,test ratios")
    parser.add_argument("--z-thresh", type=float, default=2.5, help="Higher = fewer outliers removed")
    parser.add_argument("--min-conf", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-keys",
        default="direction",
        help="Comma-separated video stat keys. Available after loading: direction,camera,video_label",
    )
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--positive-labels", default="1", help="Comma-separated labels treated as fall/positive")
    parser.add_argument(
        "--balance-frame-labels",
        action="store_true",
        help="Greedily assign whole videos so positive/negative frame counts follow split ratios",
    )
    parser.add_argument("--video-weight", type=float, default=1.0)
    parser.add_argument("--class-weight", type=float, default=3.0)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    ratios = parse_ratios(args.ratios)
    stratify_keys = split_key_columns(args.stratify_keys)
    positive_labels = parse_positive_labels(args.positive_labels)
    
    # 1. Load and filter cameras
    df = load_and_filter_cameras(Path(args.input))
    
    # 2. Remove outliers
    df = remove_outliers(df, z_thresh=args.z_thresh, min_conf=args.min_conf)
    
    # 3. Split and Save
    stats_summary = split_and_save(
        df,
        out_dir,
        ratios=ratios,
        seed=args.seed,
        stratify_keys=stratify_keys,
        label_col=args.label_column,
        positive_labels=positive_labels,
        balance_frame_labels=args.balance_frame_labels,
        video_weight=args.video_weight,
        class_weight=args.class_weight,
    )
    
    # 4. Save summary
    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n[4/4] Done. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
