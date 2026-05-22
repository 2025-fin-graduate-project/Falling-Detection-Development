#!/usr/bin/env python3
"""Experiment 7: Error timeline case studies.

Generates fall-score timelines for representative TP, FP, FN, TN videos.
For each error category, picks 3 representative examples and plots the
continuous fall score per window, with the fall event annotated.

Outputs:
  results/additional_report_experiments/exp7_score_timelines/
    fig_timelines_tp.png
    fig_timelines_fp.png
    fig_timelines_fn.png
    fig_timelines_tn.png
    timeline_examples.csv   — list of chosen videos per category

Usage:
    uv run python scripts/util/exp7_score_timelines.py [--model P42-kp17-w60]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

PROJ_ROOT = Path(__file__).resolve().parents[2]

MODEL_CONFIGS = {
    "P42-kp17-w60": {
        "exp_dir": PROJ_ROOT / "results/phase42_cross/P42-kp17-w60",
        "label": "kp17, w=60, filtered",
    },
    "P41-kp13-w60": {
        "exp_dir": PROJ_ROOT / "results/phase41_ablation/P41-kp13-w60",
        "label": "kp13, w=60, filtered",
    },
    "P41-raw-kp7-w40": {
        "exp_dir": PROJ_ROOT / "results/phase41_ablation/P41-raw-kp7-w40",
        "label": "kp7, w=40, raw",
    },
}

OUT_DIR = PROJ_ROOT / "results/additional_report_experiments/exp7_score_timelines"
FPS = 15.0
N_EXAMPLES = 3  # examples per category


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


def build_windows(video_data, window_size, vel_idx, mn, sc):
    result = {}
    for vid, (frames, labels) in video_data.items():
        n = len(frames)
        if n < window_size:
            continue
        f = frames.copy()
        if vel_idx:
            f = add_velocity(f, vel_idx)
        f = np.clip((f - mn) / sc, 0.0, 1.0)
        # stride=1 for full timeline
        indices = list(range(0, n - window_size + 1, 1))
        wins = np.stack([f[t:t + window_size] for t in indices], 0).astype(np.float32)
        has_fall = bool(np.any(labels == 1))
        fall_start = int(np.argmax(labels == 1)) if has_fall else -1
        result[vid] = (has_fall, wins, np.array(indices, dtype=np.int32), fall_start, labels)
    return result


def predict_float(model, video_windows: dict, batch_size: int = 512) -> dict:
    vid_order = list(video_windows.keys())
    all_wins = np.concatenate([v[1] for v in video_windows.values()], axis=0)
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


def classify_video(has_fall, scores, indices, threshold, vote_window, vote_k):
    buf = np.zeros(vote_window, dtype=np.int32)
    vote_sum = head = 0
    detected = False
    detect_frame = None
    for i, (t_start, score) in enumerate(zip(indices, scores)):
        v = 1 if score >= threshold else 0
        vote_sum += v - int(buf[head])
        buf[head] = v
        head = (head + 1) % vote_window
        if vote_sum >= vote_k:
            detected = True
            detect_frame = t_start
            break
    if has_fall and detected:
        return "TP", detect_frame
    elif has_fall and not detected:
        return "FN", None
    elif not has_fall and detected:
        return "FP", detect_frame
    else:
        return "TN", None


def plot_timelines(category, examples, threshold, model_label, out_dir: Path | None = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    if out_dir is None:
        out_dir = OUT_DIR

    n = len(examples)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    cat_colors = {"TP": "#4CAF50", "FP": "#FF5722", "FN": "#9C27B0", "TN": "#2196F3"}
    color = cat_colors.get(category, "gray")

    for ax, ex in zip(axes, examples):
        vid = ex["vid"]
        scores = ex["scores"]
        indices = ex["indices"]
        fall_start = ex["fall_start"]
        detect_frame = ex["detect_frame"]
        n_frames = ex["n_frames"]

        # x-axis: center frame of each window, in seconds
        center_frames = indices + ex["window_size"] // 2
        t = center_frames / FPS

        ax.plot(t, scores, color=color, linewidth=1.5, alpha=0.9)
        ax.axhline(threshold, ls="--", color="gray", linewidth=1, label=f"thr={threshold}")
        ax.fill_between(t, 0, scores, alpha=0.15, color=color)

        if fall_start >= 0:
            ax.axvline(fall_start / FPS, ls="-", color="red", linewidth=1.5, label="fall start")
        if detect_frame is not None:
            ax.axvline(detect_frame / FPS, ls=":", color="orange", linewidth=1.5, label="detected")

        ax.set_xlim(0, n_frames / FPS)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time (s)")
        if ax == axes[0]:
            ax.set_ylabel("Fall Score")
        ax.set_title(f"{vid[:20]}", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    cat_names = {"TP": "True Positives (detected fall)", "FP": "False Positives (false alarm)",
                 "FN": "False Negatives (missed fall)", "TN": "True Negatives (correct non-fall)"}
    fig.suptitle(f"{category}: {cat_names.get(category, '')} — {model_label}", fontsize=10, y=1.02)
    fig.tight_layout()

    out_path = out_dir / f"fig_timelines_{category.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


def run_model(model_name: str):
    import tensorflow as tf

    mcfg    = MODEL_CONFIGS[model_name]
    exp_dir = mcfg["exp_dir"]
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
    print(f"  Building stride-1 windows (slow — for full timeline) ...")
    video_windows = build_windows(video_data, window_size, vel_idx, mn_flat, sc_flat)

    print(f"  Loading model ...")
    model = tf.keras.models.load_model(str(exp_dir / "model.keras"), compile=False)
    vid_scores = predict_float(model, video_windows)

    # Classify each video
    categories = defaultdict(list)
    for vid, (has_fall, wins, indices, fall_start, labels) in video_windows.items():
        scores = vid_scores[vid]
        cat, detect_frame = classify_video(has_fall, scores, indices, threshold, vote_window, vote_k)
        categories[cat].append({
            "vid": vid,
            "scores": scores,
            "indices": indices,
            "fall_start": fall_start,
            "detect_frame": detect_frame,
            "n_frames": len(labels),
            "window_size": window_size,
        })

    print(f"  TP={len(categories['TP'])} FP={len(categories['FP'])} "
          f"FN={len(categories['FN'])} TN={len(categories['TN'])}")

    model_out_dir = OUT_DIR / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    for cat in ["TP", "FP", "FN", "TN"]:
        examples = categories[cat]
        # Sort for reproducibility: pick videos with interesting variation
        # TP: pick ones with shortest latency (early detection)
        # FN: pick ones with highest max score (close misses)
        # FP: pick ones with highest max score
        # TN: pick ones with highest max score (close calls)
        if cat == "TP":
            examples = sorted(examples, key=lambda e: (e["detect_frame"] or 9999) - e["fall_start"])
        elif cat in ("FN", "FP", "TN"):
            examples = sorted(examples, key=lambda e: -float(np.max(e["scores"])))
        chosen = examples[:N_EXAMPLES]
        plot_timelines(cat, chosen, threshold, mcfg["label"], out_dir=model_out_dir)
        for ex in chosen:
            csv_rows.append({
                "model": model_name,
                "category": cat,
                "vid": ex["vid"],
                "fall_start_frame": ex["fall_start"],
                "detect_frame": ex["detect_frame"],
                "max_score": round(float(np.max(ex["scores"])), 4),
                "mean_score": round(float(np.mean(ex["scores"])), 4),
            })

    return csv_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="P42-kp17-w60", choices=list(MODEL_CONFIGS.keys()))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_rows = run_model(args.model)

    fieldnames = ["model", "category", "vid", "fall_start_frame", "detect_frame",
                  "max_score", "mean_score"]

    # Per-model CSV
    model_out_dir = OUT_DIR / args.model
    model_out_dir.mkdir(parents=True, exist_ok=True)
    model_csv = model_out_dir / "timeline_examples.csv"
    with open(model_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nSaved: {model_csv}")

    # Combined CSV (append-mode so multiple models accumulate)
    combined_path = OUT_DIR / "timeline_examples.csv"
    write_header = not combined_path.exists()
    with open(combined_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(csv_rows)
    print(f"Appended: {combined_path}")
    print("Experiment 7 complete.")


if __name__ == "__main__":
    main()
