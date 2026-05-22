#!/usr/bin/env python3
"""Experiment 3: Post-processing detection latency analysis.

For each fall video, measure the frame at which the K-of-N vote triggers.
Compute the distribution of latency (frames / seconds from fall start).

Outputs:
  results/additional_report_experiments/exp3_detection_latency/
    latency_stats.csv        — per-model latency statistics
    latency_per_video.csv    — raw per-video latency values
    fig_latency_cdf.png      — CDF of detection latency

Usage:
    uv run python scripts/util/exp3_detection_latency.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

PROJ_ROOT = Path(__file__).resolve().parents[2]

MODEL_CONFIGS = [
    {
        "name": "P42-kp17-w60",
        "exp_dir": PROJ_ROOT / "results/phase42_cross/P42-kp17-w60",
        "label": "kp17, w=60, filtered",
    },
    {
        "name": "P41-kp13-w60",
        "exp_dir": PROJ_ROOT / "results/phase41_ablation/P41-kp13-w60",
        "label": "kp13, w=60, filtered",
    },
    {
        "name": "P41-raw-kp7-w40",
        "exp_dir": PROJ_ROOT / "results/phase41_ablation/P41-raw-kp7-w40",
        "label": "kp7, w=40, raw",
    },
]

FPS = 15.0
OUT_DIR = PROJ_ROOT / "results/additional_report_experiments/exp3_detection_latency"


def _kp_cols(indices):
    return [f"kp{i}_{ax}" for i in indices for ax in ("y","x","s")]

FEATURE_SETS = {
    "kp5":  _kp_cols([0,5,6,11,12]),
    "kp7":  _kp_cols([0,5,6,11,12,15,16]),
    "kp9":  _kp_cols([0,5,6,7,8,11,12,15,16]),
    "kp11": _kp_cols([0,5,6,7,8,9,10,11,12,15,16]),
    "kp13": _kp_cols([0,5,6,7,8,9,10,11,12,13,14,15,16]),
    "kp17": _kp_cols(range(17)),
}
DERIVED_FILT = ["HSSC_y","HSSC_x","RWHC","VHSSC","AHSSC","AHSSC_x"]
VEL_BASE_MAP = {
    k: [c for c in v if c.endswith("_y") or c.endswith("_x")]
    for k, v in FEATURE_SETS.items()
}


def add_velocity(frames: np.ndarray, vel_idx: list[int]) -> np.ndarray:
    vel = np.zeros_like(frames[:, vel_idx])
    vel[1:] = frames[1:, vel_idx] - frames[:-1, vel_idx]
    return np.concatenate([frames, vel], axis=1)


def load_video_data(csv_path: Path, feat_cols: list[str]) -> dict:
    data = defaultdict(lambda: {"feat": [], "label": []})
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            try:
                feat = [float(row[c]) if c in row and row[c] != "" else 0.0 for c in feat_cols]
            except (KeyError, ValueError):
                continue
            data[vid]["feat"].append(feat)
            data[vid]["label"].append(int(row["label"]))
    result = {}
    for vid, d in data.items():
        result[vid] = (
            np.array(d["feat"], dtype=np.float32),
            np.array(d["label"], dtype=np.int32),
        )
    return result


def build_windows(video_data, window_size, vel_idx, mn, sc, stride=1):
    result = {}
    for vid, (frames, labels) in video_data.items():
        n = len(frames)
        if n < window_size:
            continue
        f = frames.copy()
        if vel_idx:
            f = add_velocity(f, vel_idx)
        f = np.clip((f - mn) / sc, 0.0, 1.0)
        indices = list(range(0, n - window_size + 1, stride))
        wins = np.stack([f[t:t + window_size] for t in indices], 0).astype(np.float32)
        has_fall = bool(np.any(labels == 1))
        fall_start = int(np.argmax(labels == 1)) if has_fall else -1
        result[vid] = (has_fall, wins, np.array(indices, dtype=np.int32), fall_start, labels)
    return result


def predict_float(model, video_windows: dict, batch_size: int = 512) -> dict:
    vid_order = list(video_windows.keys())
    all_wins = np.concatenate([video_windows[v][1] for v in vid_order], axis=0)
    print(f"    Float inference on {len(all_wins):,} windows ...")
    preds = model.predict(all_wins, batch_size=batch_size, verbose=0)
    fall_scores = preds[:, 1] if preds.shape[1] == 2 else preds[:, 0]

    result = {}
    offset = 0
    for vid in vid_order:
        n = len(video_windows[vid][1])
        result[vid] = fall_scores[offset:offset+n]
        offset += n
    return result


def compute_latency(video_windows, vid_scores, threshold, vote_window, vote_k):
    """Return list of (vid, fall_start_frame, detect_frame, latency_frames) for fall videos."""
    latencies = []
    missed = 0
    for vid, (has_fall, wins, indices, fall_start, labels) in video_windows.items():
        if not has_fall:
            continue
        scores = vid_scores[vid]
        buf = np.zeros(vote_window, dtype=np.int32)
        vote_sum = head = 0
        detected_frame = None
        for i, (t_start, score) in enumerate(zip(indices, scores)):
            v = 1 if score >= threshold else 0
            vote_sum += v - int(buf[head])
            buf[head] = v
            head = (head + 1) % vote_window
            if vote_sum >= vote_k:
                # Detection triggered at end of window at t_start
                detected_frame = t_start + len(wins[i]) - 1
                break
        if detected_frame is None:
            missed += 1
            continue
        latency = detected_frame - fall_start  # frames from first fall label to detection
        latencies.append({
            "vid": vid,
            "fall_start_frame": fall_start,
            "detect_frame": detected_frame,
            "latency_frames": latency,
            "latency_sec": round(latency / FPS, 3),
        })
    return latencies, missed


def run_model(mcfg: dict) -> tuple[dict, list[dict]]:
    import tensorflow as tf

    exp_dir = mcfg["exp_dir"]
    model_name = mcfg["name"]
    print(f"\n{'='*60}")
    print(f"[{model_name}] {mcfg['label']}")

    mets      = json.loads((exp_dir / "metrics.json").read_text())
    norm      = json.loads((exp_dir / "normalization.json").read_text())
    feat_cols = json.loads((exp_dir / "feature_columns.json").read_text())
    cfg       = mets.get("config", {})

    feature_set  = cfg.get("feature_set", "kp13")
    window_size  = int(cfg.get("window_size", 40))
    use_velocity = cfg.get("use_velocity", False)
    data_dir     = Path(cfg.get("data_dir", ""))

    mn_flat = np.array(norm["min"],   dtype=np.float32)
    sc_flat = np.array(norm["scale"], dtype=np.float32)

    vel_idx = None
    if use_velocity:
        base_cols = VEL_BASE_MAP.get(feature_set, [])
        vel_idx = [feat_cols.index(c) for c in base_cols if c in feat_cols]

    ue          = mets.get("unified_eval", {})
    threshold   = ue.get("threshold", 0.5)
    ev_params   = ue.get("event", {})
    vote_window = ev_params.get("vote_window", 10)
    vote_k      = ev_params.get("vote_k", 6)
    print(f"  thr={threshold}, v{vote_window}k{vote_k}, window_size={window_size}")

    test_csv = data_dir / "test.csv"
    print(f"  Loading {test_csv} ...")
    video_data = load_video_data(test_csv, feat_cols)
    video_windows = build_windows(video_data, window_size, vel_idx, mn_flat, sc_flat, stride=1)

    print(f"  Loading model ...")
    model = tf.keras.models.load_model(str(exp_dir / "model.keras"), compile=False)

    vid_scores = predict_float(model, video_windows)

    latencies, missed = compute_latency(video_windows, vid_scores, threshold, vote_window, vote_k)

    fall_vids = sum(1 for v in video_windows.values() if v[0])
    print(f"  Fall videos: {fall_vids}, detected: {len(latencies)}, missed (FN): {missed}")

    if latencies:
        lats = [r["latency_frames"] for r in latencies]
        secs = [r["latency_sec"] for r in latencies]
        stats = {
            "model": model_name,
            "label": mcfg["label"],
            "n_fall_videos": fall_vids,
            "n_detected": len(latencies),
            "n_missed": missed,
            "fall_recall": round(len(latencies)/fall_vids, 4),
            "mean_latency_frames": round(float(np.mean(lats)), 1),
            "median_latency_frames": round(float(np.median(lats)), 1),
            "p25_latency_frames": round(float(np.percentile(lats, 25)), 1),
            "p75_latency_frames": round(float(np.percentile(lats, 75)), 1),
            "min_latency_frames": int(np.min(lats)),
            "max_latency_frames": int(np.max(lats)),
            "mean_latency_sec": round(float(np.mean(secs)), 3),
            "median_latency_sec": round(float(np.median(secs)), 3),
            "window_size": window_size,
        }
        print(f"  Latency: mean={stats['mean_latency_sec']:.2f}s, median={stats['median_latency_sec']:.2f}s, "
              f"p25={stats['p25_latency_frames']/FPS:.2f}s, p75={stats['p75_latency_frames']/FPS:.2f}s")
        for r in latencies:
            r["model"] = model_name
        return stats, latencies
    return {"model": model_name, "label": mcfg["label"], "n_missed": missed}, []


def plot_latency_cdf(all_per_video: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    models = [c["name"] for c in MODEL_CONFIGS]

    for model_name, color in zip(models, colors):
        rows = [r for r in all_per_video if r["model"] == model_name]
        if not rows:
            continue
        lats = sorted(r["latency_sec"] for r in rows)
        cdf = np.arange(1, len(lats)+1) / len(lats)
        ax1.plot(lats, cdf, label=model_name, color=color, linewidth=2)
        ax2.hist(lats, bins=30, label=model_name, color=color, alpha=0.6, density=True)

    ax1.set_xlabel("Detection Latency (seconds from fall start)")
    ax1.set_ylabel("CDF")
    ax1.set_title("Detection Latency CDF — Fall Videos Only")
    ax1.axvline(0, ls="--", color="gray", linewidth=1)
    ax1.axvline(1.0, ls=":", color="orange", linewidth=1, label="1 s reference")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Detection Latency (seconds)")
    ax2.set_ylabel("Density")
    ax2.set_title("Detection Latency Distribution")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUT_DIR / "fig_latency_cdf.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = []
    all_per_video = []

    for mcfg in MODEL_CONFIGS:
        stats, per_video = run_model(mcfg)
        all_stats.append(stats)
        all_per_video.extend(per_video)

    # Save stats CSV
    stats_path = OUT_DIR / "latency_stats.csv"
    if all_stats:
        fieldnames = list(all_stats[0].keys())
        with open(stats_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_stats)
        print(f"\nSaved: {stats_path}")

    # Save per-video CSV
    vid_path = OUT_DIR / "latency_per_video.csv"
    if all_per_video:
        fieldnames2 = ["model", "vid", "fall_start_frame", "detect_frame",
                       "latency_frames", "latency_sec"]
        with open(vid_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames2, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_per_video)
        print(f"Saved: {vid_path}")

    plot_latency_cdf(all_per_video)
    print("\nExperiment 3 complete.")


if __name__ == "__main__":
    main()
