#!/usr/bin/env python3
"""Experiment 1: Camera/direction breakdown of float model performance.

For the top 3 ablation models, compute event-level MinP broken down by:
  - Camera: C5, C6, C7, C8
  - Fall direction: BY (back), FY (front), SY (side), N (non-fall)

Output:
  results/additional_report_experiments/exp1_camera_direction/
    camera_direction_metrics.csv
    fig_camera_minpr.png
    fig_direction_minpr.png

Usage:
    uv run python scripts/util/exp1_camera_direction_eval.py
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

# Top 3 models from Phase 41/42 ablation
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

VOTE_COMBOS = [(1,1),(3,2),(5,3),(5,4),(7,4),(7,5),(10,6)]

OUT_DIR = PROJ_ROOT / "results/additional_report_experiments/exp1_camera_direction"


# ── Feature column definitions ─────────────────────────────────────────────────

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


def get_feature_cols(feature_set: str, data_dir: str) -> list[str]:
    base = FEATURE_SETS[feature_set]
    if "filtered" in data_dir:
        return base + DERIVED_FILT
    return base


def add_velocity(frames: np.ndarray, vel_idx: list[int]) -> np.ndarray:
    vel = np.zeros_like(frames[:, vel_idx])
    vel[1:] = frames[1:, vel_idx] - frames[:-1, vel_idx]
    return np.concatenate([frames, vel], axis=1)


# ── CSV loading ────────────────────────────────────────────────────────────────

def load_video_data(csv_path: Path, feat_cols: list[str]) -> dict:
    """Returns {vid: (frames np.float32, labels np.int32)}."""
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
        indices = range(0, n - window_size + 1, stride)
        wins = np.stack([f[t:t + window_size] for t in indices], 0).astype(np.float32)
        has_fall = bool(np.any(labels == 1))
        result[vid] = (has_fall, wins)
    return result


def parse_camera(video_id: str) -> str:
    return video_id.split("_")[-1]  # C5, C6, C7, C8


def parse_direction(video_id: str) -> str:
    parts = video_id.split("_")
    d = parts[-2] if len(parts) >= 2 else "?"
    return d if d in ("BY", "FY", "SY") else "N"


# ── Inference ────────────────────────────────────────────────────────────────

def predict_float(model, video_windows: dict, batch_size: int = 512) -> dict:
    """Returns {vid: np.ndarray of fall_scores}."""
    # Collect all windows with vid bookkeeping
    vid_order = list(video_windows.keys())
    all_wins = np.concatenate([video_windows[v][1] for v in vid_order], axis=0)
    print(f"    Running float inference on {len(all_wins):,} windows ...")
    preds = model.predict(all_wins, batch_size=batch_size, verbose=0)
    fall_scores = preds[:, 1] if preds.shape[1] == 2 else preds[:, 0]

    result = {}
    offset = 0
    for vid in vid_order:
        n = len(video_windows[vid][1])
        result[vid] = (video_windows[vid][0], fall_scores[offset:offset+n])
        offset += n
    return result


# ── Vote post-processing ───────────────────────────────────────────────────────

def apply_vote(vid_scores: dict, threshold: float, vote_window: int, vote_k: int) -> dict:
    """Returns {vid: (has_fall, detected)}."""
    decisions = {}
    for vid, (has_fall, scores) in vid_scores.items():
        buf = np.zeros(vote_window, dtype=np.int32)
        vote_sum = head = 0
        detected = False
        for score in scores:
            v = 1 if score >= threshold else 0
            vote_sum += v - int(buf[head])
            buf[head] = v
            head = (head + 1) % vote_window
            if vote_sum >= vote_k:
                detected = True
                break
        decisions[vid] = (has_fall, detected)
    return decisions


def compute_metrics(decisions: dict) -> dict:
    tp = fp = fn = tn = 0
    for has_fall, detected in decisions.values():
        if has_fall:
            tp += detected; fn += not detected
        else:
            fp += detected; tn += not detected
    e = 1e-9
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fall_precision":  round(tp/(tp+fp+e), 4),
        "fall_recall":     round(tp/(tp+fn+e), 4),
        "nfall_precision": round(tn/(tn+fn+e), 4),
        "nfall_recall":    round(tn/(tn+fp+e), 4),
        "min_pr":          round(min(tp/(tp+fp+e), tp/(tp+fn+e), tn/(tn+fn+e), tn/(tn+fp+e)), 4),
        "n_videos": tp+fp+fn+tn,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_model(mcfg: dict) -> list[dict]:
    import tensorflow as tf

    exp_dir = mcfg["exp_dir"]
    model_name = mcfg["name"]
    print(f"\n{'='*60}")
    print(f"[{model_name}] {mcfg['label']}")

    met_path   = exp_dir / "metrics.json"
    norm_path  = exp_dir / "normalization.json"
    fc_path    = exp_dir / "feature_columns.json"
    keras_path = exp_dir / "model.keras"

    mets      = json.loads(met_path.read_text())
    norm      = json.loads(norm_path.read_text())
    feat_cols = json.loads(fc_path.read_text())
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

    # Load threshold and vote params from float eval
    ue = mets.get("unified_eval", {})
    threshold   = ue.get("threshold", 0.5)
    ev_params   = ue.get("event", {})
    vote_window = ev_params.get("vote_window", 10)
    vote_k      = ev_params.get("vote_k", 6)
    print(f"  thr={threshold}, v{vote_window}k{vote_k}")

    test_csv = data_dir / "test.csv"
    print(f"  Loading {test_csv} ...")
    video_data = load_video_data(test_csv, feat_cols)
    video_windows = build_windows(video_data, window_size, vel_idx, mn_flat, sc_flat, stride=1)

    print(f"  Loading model {keras_path.name} ...")
    model = tf.keras.models.load_model(str(keras_path), compile=False)

    vid_scores = predict_float(model, video_windows)

    decisions = apply_vote(vid_scores, threshold, vote_window, vote_k)

    # Overall metrics
    overall = compute_metrics(decisions)
    print(f"  Overall: MinP={overall['min_pr']:.4f} (tp={overall['tp']}, fp={overall['fp']}, fn={overall['fn']}, tn={overall['tn']})")

    rows = []

    # Camera breakdown
    for cam in ["C5", "C6", "C7", "C8"]:
        subset = {v: d for v, d in decisions.items() if parse_camera(v) == cam}
        if not subset:
            continue
        m = compute_metrics(subset)
        rows.append({
            "model": model_name,
            "group_type": "camera",
            "group": cam,
            **m,
        })
        print(f"  {cam}: MinP={m['min_pr']:.4f} n={m['n_videos']} tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")

    # Direction breakdown
    for direction in ["BY", "FY", "SY", "N"]:
        subset = {v: d for v, d in decisions.items() if parse_direction(v) == direction}
        if not subset:
            continue
        m = compute_metrics(subset)
        rows.append({
            "model": model_name,
            "group_type": "direction",
            "group": direction,
            **m,
        })
        print(f"  {direction}: MinP={m['min_pr']:.4f} n={m['n_videos']} tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")

    # Overall
    rows.append({
        "model": model_name,
        "group_type": "overall",
        "group": "overall",
        **overall,
    })

    return rows


def plot_results(all_rows: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    models = [c["name"] for c in MODEL_CONFIGS]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    # Camera breakdown plot
    fig, ax = plt.subplots(figsize=(9, 5))
    cameras = ["C5", "C6", "C7", "C8"]
    x = np.arange(len(cameras))
    width = 0.25
    for i, (model_name, color) in enumerate(zip(models, colors)):
        vals = []
        for cam in cameras:
            r = next((r for r in all_rows if r["model"] == model_name and r["group"] == cam), None)
            vals.append(r["min_pr"] if r else 0.0)
        ax.bar(x + i*width, vals, width, label=model_name, color=color, alpha=0.85)
    ax.set_xlabel("Camera")
    ax.set_ylabel("Event MinP")
    ax.set_title("Camera-wise Event MinP — Top 3 Models")
    ax.set_xticks(x + width)
    ax.set_xticklabels(cameras)
    ax.set_ylim(0.75, 1.0)
    ax.axhline(0.90, ls="--", color="red", linewidth=1, label="INT8 target (0.90)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "fig_camera_minpr.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path.name}")

    # Direction breakdown plot — only fall directions + N
    fig, ax = plt.subplots(figsize=(9, 5))
    direction_labels = {"BY": "Back (BY)", "FY": "Front (FY)", "SY": "Side (SY)", "N": "Non-fall (N)"}
    dirs = list(direction_labels.keys())
    x = np.arange(len(dirs))
    for i, (model_name, color) in enumerate(zip(models, colors)):
        vals = []
        for d in dirs:
            r = next((r for r in all_rows if r["model"] == model_name and r["group"] == d), None)
            vals.append(r["min_pr"] if r else 0.0)
        ax.bar(x + i*width, vals, width, label=model_name, color=color, alpha=0.85)
    ax.set_xlabel("Fall Direction / Category")
    ax.set_ylabel("Event MinP (or Recall/Precision where applicable)")
    ax.set_title("Direction-wise Event MinP — Top 3 Models")
    ax.set_xticks(x + width)
    ax.set_xticklabels([direction_labels[d] for d in dirs])
    ax.set_ylim(0.75, 1.0)
    ax.axhline(0.90, ls="--", color="red", linewidth=1, label="INT8 target (0.90)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "fig_direction_minpr.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for mcfg in MODEL_CONFIGS:
        rows = run_model(mcfg)
        all_rows.extend(rows)

    # Save CSV
    csv_path = OUT_DIR / "camera_direction_metrics.csv"
    fieldnames = ["model", "group_type", "group", "n_videos",
                  "min_pr", "fall_precision", "fall_recall",
                  "nfall_precision", "nfall_recall", "tp", "fp", "fn", "tn"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved: {csv_path}")

    plot_results(all_rows)
    print("\nExperiment 1 complete.")


if __name__ == "__main__":
    main()
