#!/usr/bin/env python3
"""CSV → numpy window arrays for Keras 3.7 training (STedgeAI Python has no pandas).

Usage:
    uv run python scripts/util/build_train_npy.py \
        --train-csv dataset/splits_v2_filtered/train.csv \
        --val-csv   dataset/splits_v2_filtered/val.csv \
        --test-csv  dataset/splits_v2_filtered/test.csv \
        --feature-set kp12 --preprocessing filtered \
        --target-steps 60 --output-dir /tmp/p44_kp13_w60_npy

Output dir contains:
    X_train.npy, y_train.npy, groups_train.npy
    X_val.npy,   y_val.npy,   groups_val.npy
    X_test.npy,  y_test.npy,  groups_test.npy
    norm_params.npz  (norm_min, norm_scale)
    data_config.json (feature_cols, num_features, target_steps, …)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── feature set definitions (mirrors train_baseline.py) ─────────────────────

_FEATURE_SETS: dict[str, list[int]] = {
    "kp7":  [0, 5, 6, 7, 8, 11, 12],
    "kp8":  [0, 5, 6, 7, 8, 9, 11, 12],
    "kp12": list(range(13)),
    "all":  list(range(17)),
}

_ENGINEERED_RAW      = ["HSSC_y", "HSSC_x", "RWHC", "VHSSC"]
_ENGINEERED_FILTERED = ["AHSSC", "AHSSC_x"]

# velocity columns from build_filtered_v2_splits_kv.py
_VEL_KP_IDX = [0, 5, 6, 11, 12]
_VEL_COLS   = [f"kp{i}_{ax}" for i in _VEL_KP_IDX for ax in ("vy", "vx")]


def build_feature_cols(df_cols: list[str], feature_set: str, preprocessing: str, add_velocity: bool = False) -> list[str]:
    kp_idx = _FEATURE_SETS.get(feature_set, _FEATURE_SETS["kp12"])
    kp_cols = [f"kp{i}_{ax}" for i in kp_idx for ax in ("x", "y", "conf")]
    eng = list(_ENGINEERED_RAW)
    if preprocessing == "filtered":
        eng += _ENGINEERED_FILTERED
    vel = _VEL_COLS if add_velocity else []
    all_cols = kp_cols + eng + vel
    return [c for c in all_cols if c in df_cols]


# ── window extraction ────────────────────────────────────────────────────────

def extract_windows(
    csv_path: Path,
    feat_cols: list[str],
    target_steps: int,
    window_start_sec: float,
    window_end_sec: float,
    label_column: str,
    stride: int,
    train_negative_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, groups) — unnormalized float32 windows."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df[feat_cols] = df[feat_cols].fillna(0.0)
    df["video_id"] = df["video_id"].astype(str)
    df["_lbl"] = (df[label_column] > 0).astype(np.int32)

    windows, labels, groups = [], [], []

    for video_id, grp in df.groupby("video_id", sort=False):
        seg = grp[(grp["time_sec"] >= window_start_sec) & (grp["time_sec"] < window_end_sec)].reset_index(drop=True)
        if len(seg) < target_steps:
            continue
        vals = seg[feat_cols].to_numpy(dtype=np.float32)
        ylabs = seg["_lbl"].to_numpy(dtype=np.int32)

        neg_count = 0
        for start in range(0, len(seg) - target_steps + 1, stride):
            chunk = vals[start: start + target_steps]
            lbl = int(ylabs[start: start + target_steps].max())
            if lbl == 0 and train_negative_stride > 1:
                if neg_count % train_negative_stride != 0:
                    neg_count += 1
                    continue
                neg_count += 1
            windows.append(chunk)
            labels.append(lbl)
            groups.append(str(video_id))

    if not windows:
        raise ValueError(f"No windows extracted from {csv_path}. Check time range / target_steps / CSV columns.")

    X = np.stack(windows, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    g = np.array(groups, dtype=str)
    return X, y, g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv",   required=True)
    ap.add_argument("--test-csv",  required=True)
    ap.add_argument("--feature-set", default="kp12", choices=list(_FEATURE_SETS))
    ap.add_argument("--preprocessing", default="filtered", choices=["raw", "filtered"])
    ap.add_argument("--add-velocity", action="store_true", help="Add per-joint velocity cols")
    ap.add_argument("--target-steps", type=int, default=60)
    ap.add_argument("--window-start-sec", type=float, default=3.0)
    ap.add_argument("--window-end-sec",   type=float, default=9.0)
    ap.add_argument("--label-column",     default="label")
    ap.add_argument("--eval-stride",      type=int,   default=1)
    ap.add_argument("--train-negative-stride", type=int, default=2)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # determine feature columns from train CSV schema
    probe_df = pd.read_csv(args.train_csv, nrows=1)
    feat_cols = build_feature_cols(
        probe_df.columns.tolist(), args.feature_set, args.preprocessing, args.add_velocity
    )
    if not feat_cols:
        raise ValueError("No feature columns found — check feature_set and CSV columns.")
    print(f"Feature columns: {len(feat_cols)}  (e.g. {feat_cols[:3]} … {feat_cols[-2:]})")

    # extract
    print("Extracting train windows …")
    X_tr, y_tr, g_tr = extract_windows(
        Path(args.train_csv), feat_cols, args.target_steps,
        args.window_start_sec, args.window_end_sec,
        args.label_column, stride=1,
        train_negative_stride=args.train_negative_stride,
    )
    print(f"  train: {X_tr.shape}  pos={y_tr.sum()}  neg={(y_tr==0).sum()}")

    print("Extracting val windows …")
    X_va, y_va, g_va = extract_windows(
        Path(args.val_csv), feat_cols, args.target_steps,
        args.window_start_sec, args.window_end_sec,
        args.label_column, stride=args.eval_stride, train_negative_stride=1,
    )
    print(f"  val:   {X_va.shape}  pos={y_va.sum()}  neg={(y_va==0).sum()}")

    print("Extracting test windows …")
    X_te, y_te, g_te = extract_windows(
        Path(args.test_csv), feat_cols, args.target_steps,
        args.window_start_sec, args.window_end_sec,
        args.label_column, stride=args.eval_stride, train_negative_stride=1,
    )
    print(f"  test:  {X_te.shape}  pos={y_te.sum()}  neg={(y_te==0).sum()}")

    # normalize (min-max fitted on train)
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    norm_min   = flat.min(axis=0)
    norm_scale = np.maximum(flat.max(axis=0) - norm_min, 1e-6)

    def normalize(x: np.ndarray) -> np.ndarray:
        return np.clip((x - norm_min) / norm_scale, 0.0, 1.0).astype(np.float32)

    X_tr = normalize(X_tr)
    X_va = normalize(X_va)
    X_te = normalize(X_te)

    # save
    np.save(out / "X_train.npy",  X_tr)
    np.save(out / "y_train.npy",  y_tr)
    np.save(out / "X_val.npy",    X_va)
    np.save(out / "y_val.npy",    y_va)
    np.save(out / "X_test.npy",   X_te)
    np.save(out / "y_test.npy",   y_te)
    np.savez(out / "norm_params.npz", norm_min=norm_min, norm_scale=norm_scale)

    # groups as object arrays
    np.save(out / "groups_train.npy", g_tr)
    np.save(out / "groups_val.npy",   g_va)
    np.save(out / "groups_test.npy",  g_te)

    cfg = {
        "feature_set": args.feature_set,
        "preprocessing": args.preprocessing,
        "add_velocity": args.add_velocity,
        "target_steps": args.target_steps,
        "num_features": len(feat_cols),
        "feature_cols": feat_cols,
        "window_start_sec": args.window_start_sec,
        "window_end_sec": args.window_end_sec,
        "label_column": args.label_column,
        "eval_stride": args.eval_stride,
        "train_negative_stride": args.train_negative_stride,
    }
    (out / "data_config.json").write_text(json.dumps(cfg, indent=2))

    print(f"\nSaved to {out}")
    print(f"  input shape: ({args.target_steps}, {len(feat_cols)})")
    print(f"  norm_min[:4]={norm_min[:4].round(4)}  norm_scale[:4]={norm_scale[:4].round(4)}")


if __name__ == "__main__":
    main()
