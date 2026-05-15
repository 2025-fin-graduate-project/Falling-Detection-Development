#!/usr/bin/env python3
"""Phase 10 FP/FN error timeline report and hard-negative mining for P9O-v01.

Outputs (all in --output-dir):
  false_positive_videos.csv
  false_negative_videos.csv
  video_score_summary.csv
  fp_score_timelines.csv
  fn_score_timelines.csv
  hard_negative_windows.csv   (val FP windows with score >= 0.40 or top-10 per video)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


ENGINEERED_FILTERED = ["HSSC_y", "HSSC_x", "RWHC", "VHSSC", "AHSSC", "AHSSC_x"]
FEATURE_SETS = {
    "minimal": [0, 5, 6, 11, 12],
    "kp7":     [0, 5, 6, 7, 8, 11, 12],
    "kp8":     [0, 5, 6, 7, 8, 9, 11, 12],
    "kp12":    list(range(13)),
    "all":     list(range(17)),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True, type=Path,
                   help="Path to the experiment dir (e.g. results/phase9_minpr_open/P9O-v01)")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=None,
                   help="Override threshold (default: read from metrics.json)")
    p.add_argument("--min-consecutive", type=int, default=None,
                   help="Override min_consecutive (default: read from metrics.json)")
    p.add_argument("--hard-negative-score-threshold", type=float, default=0.40)
    p.add_argument("--hard-negative-top-k", type=int, default=10)
    return p.parse_args()


def feature_columns(columns: list[str], feature_set: str) -> list[str]:
    kp_indexes = FEATURE_SETS[feature_set]
    kp_cols = [
        f"kp{idx}_{axis}"
        for idx in kp_indexes
        for axis in ("y", "x", "s")
        if f"kp{idx}_{axis}" in columns
    ]
    return kp_cols + [c for c in ENGINEERED_FILTERED if c in columns]


def apply_consecutive_rule(binary_pred: np.ndarray, min_consecutive: int) -> np.ndarray:
    if min_consecutive <= 1:
        return binary_pred.astype(np.int32)
    filtered = np.zeros_like(binary_pred, dtype=np.int32)
    start = None
    for idx, value in enumerate(binary_pred):
        if value == 1 and start is None:
            start = idx
        elif value == 0 and start is not None:
            if idx - start >= min_consecutive:
                filtered[start:idx] = 1
            start = None
    if start is not None and len(binary_pred) - start >= min_consecutive:
        filtered[start:] = 1
    return filtered


def max_consecutive_ones(arr: np.ndarray) -> int:
    max_run = cur = 0
    for v in arr:
        if v == 1:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def build_windows(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_steps: int,
    window_start_sec: float,
    window_end_sec: float,
    eval_stride: int,
) -> tuple[list[str], list[int], list[float], list[int], list[str], list[np.ndarray]]:
    video_ids, window_idxs, window_start_secs, window_start_frames, directions, chunks = [], [], [], [], [], []
    df = df.sort_values(["video_id", "time_sec", "frame"])
    for video_id, grp in df.groupby("video_id", sort=False):
        seg = grp[(grp["time_sec"] >= window_start_sec) & (grp["time_sec"] < window_end_sec)]
        if len(seg) < target_steps:
            continue
        values = seg[feat_cols].to_numpy(dtype=np.float32)
        frames = seg["frame"].to_numpy(dtype=np.int32)
        times = seg["time_sec"].to_numpy(dtype=np.float32)
        dir_ = str(seg["direction"].iloc[0]) if "direction" in seg.columns else "UNKNOWN"
        for wi, start in enumerate(range(0, len(seg) - target_steps + 1)):
            if start % eval_stride != 0:
                continue
            video_ids.append(str(video_id))
            window_idxs.append(wi)
            window_start_secs.append(float(times[start]))
            window_start_frames.append(int(frames[start]))
            directions.append(dir_)
            chunks.append(values[start : start + target_steps])
    return video_ids, window_idxs, window_start_secs, window_start_frames, directions, chunks


def infer_direction(video_id: str) -> str:
    for part in str(video_id).split("_"):
        if part in {"BY", "FY", "SY", "N"}:
            return part
    return "UNKNOWN"


def load_split(csv_path: Path, feat_cols_ordered: list[str], feature_set: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["video_id"] = df["video_id"].astype(str)
    if "direction" not in df.columns:
        df["direction"] = df["video_id"].map(infer_direction)
    df["direction"] = df["direction"].astype(str)
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    needed = feature_columns(df.columns.tolist(), feature_set)
    for c in needed:
        if c not in df.columns:
            df[c] = 0.0
    df[needed] = df[needed].fillna(0.0)
    return df


def normalize(chunks: list[np.ndarray], min_v: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = np.stack(chunks).astype(np.float32)
    return ((x - min_v) / scale).astype(np.float32)


def per_video_summary(
    video_ids: list[str],
    scores: np.ndarray,
    binary_windows: np.ndarray,
    labels: np.ndarray,
    directions: list[str],
    window_start_secs: list[float],
    conf_means: list[float],
    threshold: float,
    min_consecutive: int,
) -> pd.DataFrame:
    rows = []
    uid_arr = np.array(video_ids)
    for vid in np.unique(uid_arr):
        mask = uid_arr == vid
        sc = scores[mask]
        bw = binary_windows[mask]
        lb = int(labels[mask].max())
        dir_ = directions[np.where(mask)[0][0]]
        secs = np.array(window_start_secs)[mask]
        conf = np.array(conf_means)[mask]

        filtered = apply_consecutive_rule(bw, min_consecutive)
        pred = int(filtered.max()) if len(filtered) > 0 else 0

        pos_indices = np.where(bw == 1)[0]
        first_pos = int(pos_indices[0]) if len(pos_indices) > 0 else -1
        last_pos  = int(pos_indices[-1]) if len(pos_indices) > 0 else -1

        top3 = float(np.mean(np.sort(sc)[-3:])) if len(sc) >= 3 else float(np.mean(sc))
        top5 = float(np.mean(np.sort(sc)[-5:])) if len(sc) >= 5 else float(np.mean(sc))

        rows.append({
            "video_id":             vid,
            "true_label":           lb,
            "pred_label":           pred,
            "direction":            dir_,
            "max_score":            float(sc.max()),
            "top3_mean":            top3,
            "top5_mean":            top5,
            "positive_count":       int(bw.sum()),
            "positive_ratio":       float(bw.mean()),
            "max_consecutive":      max_consecutive_ones(bw),
            "first_positive_window": first_pos,
            "last_positive_window":  last_pos,
            "mean_keypoint_confidence": float(np.mean(conf)),
            "min_keypoint_confidence":  float(np.min(conf)),
        })
    return pd.DataFrame(rows)


def per_video_timelines(
    video_ids: list[str],
    window_idxs: list[int],
    window_start_frames: list[int],
    window_start_secs: list[float],
    scores: np.ndarray,
    labels: np.ndarray,
    directions: list[str],
    conf_means: list[float],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for i in range(len(video_ids)):
        rows.append({
            "video_id":                 video_ids[i],
            "window_index":             window_idxs[i],
            "window_start_frame":       window_start_frames[i],
            "window_start_sec":         window_start_secs[i],
            "score":                    float(scores[i]),
            "binary_at_selected_threshold": int(scores[i] >= threshold),
            "label":                    int(labels[i]),
            "direction":                directions[i],
            "confidence_mean":          float(conf_means[i]),
        })
    return pd.DataFrame(rows)


def mine_hard_negatives(
    timelines: pd.DataFrame,
    fp_video_ids: set[str],
    score_threshold: float,
    top_k: int,
    feature_set: str,
    target_steps: int,
    source_split: str,
) -> pd.DataFrame:
    """Mine high-score windows from val FP non-fall videos."""
    fp_tl = timelines[(timelines["video_id"].isin(fp_video_ids)) & (timelines["label"] == 0)].copy()
    if fp_tl.empty:
        return pd.DataFrame()

    rows = []
    for vid, grp in fp_tl.groupby("video_id"):
        grp_sorted = grp.sort_values("score", ascending=False).reset_index(drop=True)
        mask = (grp_sorted["score"] >= score_threshold) | (grp_sorted.index < top_k)
        selected = grp_sorted[mask].copy()
        selected["rank_in_video"] = range(1, len(selected) + 1)
        for _, row in selected.iterrows():
            rows.append({
                "video_id":         str(vid),
                "window_start_frame": int(row["window_start_frame"]),
                "window_start_sec": float(row["window_start_sec"]),
                "score":            float(row["score"]),
                "rank_in_video":    int(row["rank_in_video"]),
                "feature_set":      feature_set,
                "target_steps":     target_steps,
                "source_split":     source_split,
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = json.loads((exp_dir / "metrics.json").read_text())
    run_config   = json.loads((exp_dir / "run_config.resolved.json").read_text())
    norm_json    = json.loads((exp_dir / "normalization.json").read_text())
    feat_cols    = json.loads((exp_dir / "feature_columns.json").read_text())

    threshold      = args.threshold      if args.threshold      is not None else float(metrics_json["threshold_selection"]["threshold"])
    min_consecutive = args.min_consecutive if args.min_consecutive is not None else int(metrics_json["threshold_selection"]["min_consecutive"])
    feature_set    = run_config["feature_set"]
    target_steps   = run_config["target_steps"]
    window_start   = run_config["window_start_sec"]
    window_end     = run_config["window_end_sec"]
    eval_stride    = run_config.get("eval_stride", 2)

    print(f"threshold={threshold}  min_consecutive={min_consecutive}  feature_set={feature_set}  target_steps={target_steps}", flush=True)

    min_v  = np.array(norm_json["min"],   dtype=np.float32).reshape(1, 1, -1)
    scale_ = np.array(norm_json["scale"], dtype=np.float32).reshape(1, 1, -1)

    print("loading model ...", flush=True)

    class _SparseFocalLoss(tf.keras.losses.Loss):
        def __init__(self, alpha=0.25, gamma=2.0, **kw):
            super().__init__(**kw)
            self.alpha = alpha; self.gamma = gamma
        def call(self, y_true, y_pred):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            idx = tf.stack([tf.range(tf.shape(y_true)[0]), y_true], axis=1)
            p_t = tf.gather_nd(y_pred, idx)
            alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1.0 - self.alpha)
            return alpha_t * tf.pow(1.0 - p_t, self.gamma) * (-tf.math.log(p_t))
        def get_config(self):
            return {**super().get_config(), "alpha": self.alpha, "gamma": self.gamma}

    model = tf.keras.models.load_model(
        str(exp_dir / "model.keras"),
        custom_objects={"SparseFocalLoss": _SparseFocalLoss},
    )

    results = {}
    for split_name in ["val", "test"]:
        csv_path = Path(run_config[f"{split_name}_csv"])
        if not csv_path.is_absolute():
            csv_path = Path(".") / csv_path
        print(f"loading {split_name} split: {csv_path}", flush=True)
        df = load_split(csv_path, feat_cols, feature_set)

        # binary video label per video_id
        label_col = run_config.get("label_column", "label")
        positive_labels = run_config.get("positive_labels", [1])
        vid_labels: dict[str, int] = {}
        for vid, grp in df.groupby("video_id"):
            vid_labels[str(vid)] = int(grp[label_col].isin(positive_labels).any())

        video_ids, window_idxs, window_start_secs, window_start_frames, directions, chunks = build_windows(
            df, feat_cols, target_steps, window_start, window_end, eval_stride
        )
        if not chunks:
            print(f"  WARNING: no windows built for {split_name}", flush=True)
            continue

        x = normalize(chunks, min_v, scale_)

        # keypoint confidence: every 3rd feature column (index 2, 5, 8, ...)
        conf_means = [float(np.nanmean(c[:, 2::3])) if c.shape[1] >= 3 else 0.0 for c in chunks]

        print(f"  predicting {len(x)} windows ...", flush=True)
        raw_preds = model.predict(x, batch_size=512, verbose=0)
        pos_cols = np.array(positive_labels)
        scores = raw_preds[:, pos_cols].sum(axis=1)

        # window-level labels from vid_labels
        win_labels = np.array([vid_labels.get(v, 0) for v in video_ids], dtype=np.int32)
        binary_windows = (scores >= threshold).astype(np.int32)

        timelines_df = per_video_timelines(
            video_ids, window_idxs, window_start_frames, window_start_secs,
            scores, win_labels, directions, conf_means, threshold,
        )
        timelines_df.to_csv(out_dir / f"{split_name}_score_timelines.csv", index=False)

        summary = per_video_summary(
            video_ids, scores, binary_windows, win_labels, directions,
            window_start_secs, conf_means, threshold, min_consecutive,
        )
        summary.to_csv(out_dir / f"{split_name}_video_score_summary.csv", index=False)

        fp_df = summary[(summary["true_label"] == 0) & (summary["pred_label"] == 1)]
        fn_df = summary[(summary["true_label"] == 1) & (summary["pred_label"] == 0)]
        fp_df.to_csv(out_dir / f"{split_name}_false_positive_videos.csv", index=False)
        fn_df.to_csv(out_dir / f"{split_name}_false_negative_videos.csv", index=False)

        n_fp = len(fp_df); n_fn = len(fn_df)
        n_total = len(summary)
        min_pr = metrics_json["metrics"].get(f"{split_name}_video", {}).get("min_pr", float("nan"))
        print(
            f"  {split_name}: total_videos={n_total}  FP={n_fp}  FN={n_fn}  "
            f"[from metrics.json min_pr={min_pr:.4f}]",
            flush=True,
        )

        results[split_name] = {
            "timelines": timelines_df,
            "summary": summary,
            "fp_df": fp_df,
            "fn_df": fn_df,
        }

    # Canonical split files use fp/fn naming for consistency with plan
    if "val" in results:
        results["val"]["fp_df"].to_csv(out_dir / "false_positive_videos.csv", index=False)
        results["val"]["fn_df"].to_csv(out_dir / "false_negative_videos.csv", index=False)
        results["val"]["summary"].to_csv(out_dir / "video_score_summary.csv", index=False)
        # Timelines for val FPs and FNs
        fp_ids = set(results["val"]["fp_df"]["video_id"].tolist())
        fn_ids = set(results["val"]["fn_df"]["video_id"].tolist())
        tl = results["val"]["timelines"]
        tl[tl["video_id"].isin(fp_ids)].to_csv(out_dir / "fp_score_timelines.csv", index=False)
        tl[tl["video_id"].isin(fn_ids)].to_csv(out_dir / "fn_score_timelines.csv", index=False)

        # Hard-negative mining from val FPs only (never test)
        hard_negs = mine_hard_negatives(
            timelines=tl,
            fp_video_ids=fp_ids,
            score_threshold=args.hard_negative_score_threshold,
            top_k=args.hard_negative_top_k,
            feature_set=feature_set,
            target_steps=target_steps,
            source_split="val",
        )
        hard_negs.to_csv(out_dir / "hard_negative_windows.csv", index=False)
        print(f"hard_negative_windows: {len(hard_negs)} windows from {hard_negs['video_id'].nunique() if not hard_negs.empty else 0} val FP videos", flush=True)

    print(f"done. outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
