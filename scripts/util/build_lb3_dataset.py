#!/usr/bin/env python3
"""Build LB-3 datasets by adding label_3class column to existing split CSVs.

label_3class:
  0 = 비낙상 normal
  1 = 낙상 중 falling   (original label == 1)
  2 = 낙상 후 fallen    (all frames AFTER the last label==1 frame in each fall video)

Usage:
    uv run python scripts/util/build_lb3_dataset.py                    # default splits
    uv run python scripts/util/build_lb3_dataset.py --source splits_v2 # outlier-removed splits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_label_3class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_3class"] = df["label"].astype("int8")

    fall_videos = df.loc[df["label"] == 1, "video_id"].unique()
    for vid in fall_videos:
        mask = df["video_id"] == vid
        vid_idx = df.index[mask]
        fall_idx = df.index[mask & (df["label"] == 1)]
        last_fall = fall_idx[-1]
        # Frames strictly after the last falling frame → fallen (2)
        after = vid_idx[vid_idx > last_fall]
        df.loc[after, "label_3class"] = 2

    return df


def process_splits(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        src = src_dir / f"{split}.csv"
        if not src.exists():
            print(f"  SKIP {src} (not found)")
            continue
        print(f"  Processing {src} ...", end=" ", flush=True)
        df = pd.read_csv(src, low_memory=False)
        df = add_label_3class(df)

        n_fall   = int((df["label_3class"] == 1).sum())
        n_fallen = int((df["label_3class"] == 2).sum())
        n_normal = int((df["label_3class"] == 0).sum())
        print(f"normal={n_normal:,}  falling={n_fall:,}  fallen={n_fallen:,}")

        dst = dst_dir / f"{split}.csv"
        df.to_csv(dst, index=False)
        print(f"    → {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="splits",
        choices=["splits", "splits_v2"],
        help="Source split directory name under dataset/ (default: splits)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2] / "dataset"
    src_dir = root / args.source
    dst_name = "lb3" if args.source == "splits" else "lb3_v2"
    dst_dir = root / dst_name

    print(f"Source : {src_dir}")
    print(f"Output : {dst_dir}")
    process_splits(src_dir, dst_dir)
    print("Done.")


if __name__ == "__main__":
    main()
