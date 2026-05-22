#!/usr/bin/env python3
"""Experiment 6: Keypoint confidence quartile performance analysis.

Groups test videos by their median keypoint confidence score (averaged over all
keypoints and frames), then measures event-level MinP per quartile for the top model.

Lower confidence → noisier keypoints → expect lower MinP. Quantifies how much
pose estimation quality affects fall detection accuracy.

Outputs:
  results/additional_report_experiments/exp6_confidence_quartile/
    confidence_quartile_metrics.csv
    confidence_per_video.csv
    fig_confidence_quartile.png

Usage:
    uv run python scripts/util/exp6_confidence_quartile.py
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

# Use all three top models; primary analysis on P42-kp17-w60
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

OUT_DIR = PROJ_ROOT / "results/additional_report_experiments/exp6_confidence_quartile"

# All 17 keypoint score columns
SCORE_COLS = [f"kp{i}_s" for i in range(17)]


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


def load_video_data_with_conf(csv_path: Path, feat_cols: list[str]) -> dict:
    """Returns {vid: (frames np.float32, labels np.int32, mean_conf float)}."""
    data = defaultdict(lambda: {"feat": [], "label": [], "conf": []})
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            try:
                feat = [float(row[c]) if c in row and row[c] != "" else 0.0 for c in feat_cols]
            except (KeyError, ValueError):
                continue
            conf_vals = []
            for sc in SCORE_COLS:
                if sc in row and row[sc] != "":
                    try:
                        conf_vals.append(float(row[sc]))
                    except ValueError:
                        pass
            data[vid]["feat"].append(feat)
            data[vid]["label"].append(int(row["label"]))
            if conf_vals:
                data[vid]["conf"].append(float(np.mean(conf_vals)))

    result = {}
    for vid, d in data.items():
        frames = np.array(d["feat"], dtype=np.float32)
        labels = np.array(d["label"], dtype=np.int32)
        mean_conf = float(np.mean(d["conf"])) if d["conf"] else 0.0
        result[vid] = (frames, labels, mean_conf)
    return result


def build_windows(video_data, window_size, vel_idx, mn, sc, stride=5):
    result = {}
    for vid, (frames, labels, mean_conf) in video_data.items():
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
        result[vid] = (has_fall, wins, mean_conf)
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


def apply_vote(vid, scores, threshold, vote_window, vote_k) -> bool:
    buf = np.zeros(vote_window, dtype=np.int32)
    vote_sum = head = 0
    for score in scores:
        v = 1 if score >= threshold else 0
        vote_sum += v - int(buf[head])
        buf[head] = v
        head = (head + 1) % vote_window
        if vote_sum >= vote_k:
            return True
    return False


def compute_metrics(decisions) -> dict:
    tp = fp = fn = tn = 0
    for has_fall, detected in decisions:
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


def run_model(mcfg: dict) -> tuple[list[dict], list[dict]]:
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

    # Use raw data dir for confidence scores — both have same video structure
    # For raw model, data_dir already points to splits_v2_class_balanced (raw)
    # For filtered model, data_dir points to splits_v2_class_balanced_filtered
    # Both have kp score columns
    test_csv = data_dir / "test.csv"
    print(f"  Loading {test_csv} ...")
    video_data = load_video_data_with_conf(test_csv, feat_cols)
    video_windows = build_windows(video_data, window_size, vel_idx, mn_flat, sc_flat, stride=5)

    # Confidence distribution
    confs = {vid: vd[2] for vid, vd in video_data.items()}

    print(f"  Loading model ...")
    model = tf.keras.models.load_model(str(exp_dir / "model.keras"), compile=False)

    vid_scores = predict_float(model, video_windows)

    # Decisions per video
    decisions_map = {}
    for vid, (has_fall, wins, mean_conf) in video_windows.items():
        detected = apply_vote(vid, vid_scores[vid], threshold, vote_window, vote_k)
        decisions_map[vid] = (has_fall, detected, confs.get(vid, 0.0))

    # Quartile boundaries (by mean confidence)
    all_confs = np.array([v[2] for v in decisions_map.values()])
    q25, q50, q75 = np.percentile(all_confs, [25, 50, 75])
    print(f"  Confidence quartiles: Q25={q25:.3f}, Q50={q50:.3f}, Q75={q75:.3f}")

    quartile_labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    quartile_rows = []
    per_video_rows = []

    for vid, (has_fall, detected, conf) in decisions_map.items():
        if conf <= q25:
            q = "Q1 (low)"
        elif conf <= q50:
            q = "Q2"
        elif conf <= q75:
            q = "Q3"
        else:
            q = "Q4 (high)"
        per_video_rows.append({
            "model": model_name,
            "vid": vid,
            "mean_conf": round(conf, 4),
            "quartile": q,
            "has_fall": int(has_fall),
            "detected": int(detected),
        })

    for ql in quartile_labels:
        subset = [(hf, det) for _, (hf, det, conf) in decisions_map.items()
                  if (conf <= q25 and ql == "Q1 (low)") or
                     (q25 < conf <= q50 and ql == "Q2") or
                     (q50 < conf <= q75 and ql == "Q3") or
                     (conf > q75 and ql == "Q4 (high)")]
        m = compute_metrics(subset)
        quartile_rows.append({
            "model": model_name,
            "quartile": ql,
            "conf_range": f"≤{q25:.3f}" if ql=="Q1 (low)" else
                          f"{q25:.3f}–{q50:.3f}" if ql=="Q2" else
                          f"{q50:.3f}–{q75:.3f}" if ql=="Q3" else
                          f">{q75:.3f}",
            **m,
        })
        print(f"  {ql}: MinP={m['min_pr']:.4f} n={m['n_videos']} tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")

    return quartile_rows, per_video_rows


def plot_results(all_quartile_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    models = [c["name"] for c in MODEL_CONFIGS]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    quartiles = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    x = np.arange(len(quartiles))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, color) in enumerate(zip(models, colors)):
        vals = []
        for q in quartiles:
            r = next((r for r in all_quartile_rows if r["model"] == model_name and r["quartile"] == q), None)
            vals.append(r["min_pr"] if r else 0.0)
        ax.bar(x + i*width, vals, width, label=model_name, color=color, alpha=0.85)

    ax.set_xlabel("Keypoint Confidence Quartile")
    ax.set_ylabel("Event MinP")
    ax.set_title("Event MinP by Keypoint Confidence Quartile — Top 3 Models")
    ax.set_xticks(x + width)
    ax.set_xticklabels(quartiles)
    ax.set_ylim(0.6, 1.0)
    ax.axhline(0.90, ls="--", color="red", linewidth=1, label="INT8 target (0.90)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    out_path = OUT_DIR / "fig_confidence_quartile.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_quartile_rows = []
    all_per_video = []

    for mcfg in MODEL_CONFIGS:
        q_rows, pv_rows = run_model(mcfg)
        all_quartile_rows.extend(q_rows)
        all_per_video.extend(pv_rows)

    csv_path = OUT_DIR / "confidence_quartile_metrics.csv"
    if all_quartile_rows:
        fieldnames = list(all_quartile_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_quartile_rows)
        print(f"\nSaved: {csv_path}")

    vid_path = OUT_DIR / "confidence_per_video.csv"
    if all_per_video:
        fieldnames2 = ["model", "vid", "mean_conf", "quartile", "has_fall", "detected"]
        with open(vid_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames2, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_per_video)
        print(f"Saved: {vid_path}")

    plot_results(all_quartile_rows)
    print("\nExperiment 6 complete.")


if __name__ == "__main__":
    main()
