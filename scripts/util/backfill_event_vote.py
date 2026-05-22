#!/usr/bin/env python3
"""Backfill event_vote metrics for all experiment phases.

Scans all results/ directories, detects format (new Phase36+ / old Phase1-35),
runs event_vote evaluation, and writes results under metrics["event_vote_backfill"].

Usage:
  CUDA_VISIBLE_DEVICES=-1 uv run python scripts/eval_all_event_vote.py
  CUDA_VISIBLE_DEVICES=-1 uv run python scripts/eval_all_event_vote.py --exp-dirs results/phase27_seed_sweep results/phase35_ncw
  CUDA_VISIBLE_DEVICES=-1 uv run python scripts/eval_all_event_vote.py --skip-done  # skip already backfilled
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO     = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "dataset/splits_v2_class_balanced_filtered"

KP7_COLS = [
    "kp0_y","kp0_x","kp0_s",
    "kp5_y","kp5_x","kp5_s","kp6_y","kp6_x","kp6_s",
    "kp7_y","kp7_x","kp7_s","kp8_y","kp8_x","kp8_s",
    "kp11_y","kp11_x","kp11_s","kp12_y","kp12_x","kp12_s",
    "HSSC_y","HSSC_x","RWHC","VHSSC","AHSSC","AHSSC_x",
]
KP13_COLS = [
    "kp0_y","kp0_x","kp0_s",
    "kp5_y","kp5_x","kp5_s","kp6_y","kp6_x","kp6_s",
    "kp7_y","kp7_x","kp7_s","kp8_y","kp8_x","kp8_s",
    "kp9_y","kp9_x","kp9_s","kp10_y","kp10_x","kp10_s",
    "kp11_y","kp11_x","kp11_s","kp12_y","kp12_x","kp12_s",
    "kp13_y","kp13_x","kp13_s","kp14_y","kp14_x","kp14_s",
    "kp15_y","kp15_x","kp15_s","kp16_y","kp16_x","kp16_s",
    "HSSC_y","HSSC_x","RWHC","VHSSC","AHSSC","AHSSC_x",
]
KP7_VEL_BASE  = [c for c in KP7_COLS  if c.endswith("_y") or c.endswith("_x")]
KP13_VEL_BASE = [c for c in KP13_COLS if c.endswith("_y") or c.endswith("_x")]
FEATURE_SETS  = {"kp7": KP7_COLS, "kp13": KP13_COLS, "kp12": KP13_COLS}


def load_video_frames(csv_path, feat_cols):
    data = defaultdict(lambda: {"feat": [], "label": []})
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            lbl = int(row["label"])
            try:
                feat = [float(row[c]) for c in feat_cols]
            except (KeyError, ValueError):
                continue
            data[vid]["feat"].append(feat)
            data[vid]["label"].append(lbl)
    return dict(data)


def normalize(X, mn, scale):
    return (X - mn) / np.where(scale == 0, 1.0, scale)


def event_vote_eval(model, video_data, window_size, threshold,
                    vote_window=5, vote_k=3, vel_idx=None, mn=None, sc=None):
    tp = fp = fn = tn = 0
    for vid, d in video_data.items():
        frames = np.array(d["feat"], dtype=np.float32)
        labels = np.array(d["label"], dtype=np.int32)
        if mn is not None:
            frames = normalize(frames, mn, sc)
        if vel_idx is not None:
            n = len(frames)
            vel = np.zeros((n, len(vel_idx)), np.float32)
            vel[1:] = frames[1:, vel_idx] - frames[:-1, vel_idx]
            frames = np.concatenate([frames, vel], axis=1)

        n = len(frames)
        if n < window_size:
            continue
        n_windows = n - window_size + 1
        windows = np.stack([frames[i:i+window_size] for i in range(n_windows)])
        scores = model.predict(windows, batch_size=256, verbose=0)[:, 1]

        # vote logic
        buf = []
        voted_fall = False
        for score in scores:
            buf.append(int(score >= threshold))
            if len(buf) > vote_window:
                buf.pop(0)
            if sum(buf) >= vote_k:
                voted_fall = True
                break

        vid_label = int(labels[-1])
        if vid_label == 1:
            if voted_fall: tp += 1
            else:          fn += 1
        else:
            if voted_fall: fp += 1
            else:          tn += 1

    fall_pr   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    nfall_pr  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fall_rc   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    nfall_rc  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    min_pr    = min(fall_pr, nfall_pr)
    return {
        "min_pr": round(min_pr, 4),
        "fall_precision": round(fall_pr, 4),
        "nfall_precision": round(nfall_pr, 4),
        "fall_recall": round(fall_rc, 4),
        "nfall_recall": round(nfall_rc, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "vote_window": vote_window, "vote_k": vote_k,
    }


def parse_exp_config(exp_dir: Path):
    """Extract (feature_set, window_size, threshold, model_path, use_velocity) from exp_dir.
    Returns None if the directory cannot be evaluated."""
    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    m = json.loads(metrics_path.read_text())

    # new format (Phase 36+): m["config"] exists
    if "config" in m:
        cfg = m["config"]
        feat_set    = cfg.get("feature_set", "kp13")
        window_size = cfg.get("window_size", 40)
        threshold   = m.get("threshold", 0.5)
        use_vel     = cfg.get("use_velocity", False)
        model_path  = exp_dir / "model_best.keras"
        if not model_path.exists():
            model_path = exp_dir / "model.keras"
    else:
        # old format (Phase 1-35): top-level fields
        feat_set    = m.get("feature_set", "kp7")
        threshold   = None  # will sweep on val set
        use_vel     = False
        model_path  = exp_dir / "model.keras"
        # window_size from run_config
        rc_path = exp_dir / "run_config.resolved.json"
        if rc_path.exists():
            rc = json.loads(rc_path.read_text())
            window_size = rc.get("target_steps", 40)
        else:
            window_size = 40

    if not model_path.exists():
        return None
    norm_path = exp_dir / "normalization.json"
    if not norm_path.exists():
        return None
    if feat_set not in FEATURE_SETS:
        return None

    return {
        "feat_set": feat_set,
        "window_size": window_size,
        "threshold": threshold,   # None = sweep on val
        "model_path": model_path,
        "use_vel": use_vel,
        "metrics_path": metrics_path,
        "norm_path": norm_path,
        "m": m,
    }


def run_backfill(exp_dir: Path, skip_done: bool, video_cache: dict):
    cfg = parse_exp_config(exp_dir)
    if cfg is None:
        return

    m = cfg["m"]
    if skip_done and "event_vote_backfill" in m:
        print(f"  SKIP {exp_dir.name} — already backfilled")
        return
    # Also skip Phase 36+ that already has event_vote in metrics
    if "event_vote_backfill" in m:
        pass  # re-run anyway unless skip_done
    if skip_done and m.get("metrics", {}).get("test_event_vote"):
        print(f"  SKIP {exp_dir.name} — native event_vote exists")
        return

    feat_cols = FEATURE_SETS[cfg["feat_set"]]
    vel_idx   = None
    if cfg["use_vel"]:
        vel_base = KP7_VEL_BASE if cfg["feat_set"] == "kp7" else KP13_VEL_BASE
        vel_idx  = [feat_cols.index(c) for c in vel_base]

    norm = json.loads(cfg["norm_path"].read_text())
    mn   = np.array(norm["min"], np.float32)
    sc   = np.array(norm["scale"], np.float32)

    model = tf.keras.models.load_model(str(cfg["model_path"]), compile=False)

    # Load val set first (needed for threshold sweep if threshold is None)
    cache_key_val = ("val", cfg["feat_set"], cfg["use_vel"])
    if cache_key_val not in video_cache:
        csv_path = DATA_DIR / "val.csv"
        print(f"    Loading val videos ({cfg['feat_set']})...")
        video_cache[cache_key_val] = load_video_frames(csv_path, feat_cols)
    vdata_val = video_cache[cache_key_val]

    # Threshold sweep on val if not given (old-format models)
    threshold = cfg["threshold"]
    if threshold is None:
        thresholds = np.arange(0.10, 0.96, 0.05)
        best_thr, best_minp = 0.5, -1.0
        for thr in thresholds:
            res = event_vote_eval(model, vdata_val, cfg["window_size"], float(thr),
                                  vote_window=5, vote_k=3, vel_idx=vel_idx, mn=mn, sc=sc)
            if res["min_pr"] > best_minp:
                best_minp, best_thr = res["min_pr"], float(thr)
        threshold = best_thr
        print(f"  EVAL {exp_dir.name}  feat={cfg['feat_set']}  ws={cfg['window_size']}  thr=sweep→{threshold:.2f}  val_best={best_minp:.4f}")
    else:
        print(f"  EVAL {exp_dir.name}  feat={cfg['feat_set']}  ws={cfg['window_size']}  thr={threshold:.3f}")

    results = {}
    val_vote = event_vote_eval(model, vdata_val, cfg["window_size"], threshold,
                               vote_window=5, vote_k=3, vel_idx=vel_idx, mn=mn, sc=sc)
    results["val"] = val_vote
    print(f"    val: event_vote MinP={val_vote['min_pr']:.4f}  FP={val_vote['fp']}  FN={val_vote['fn']}")

    cache_key_test = ("test", cfg["feat_set"], cfg["use_vel"])
    if cache_key_test not in video_cache:
        csv_path = DATA_DIR / "test.csv"
        print(f"    Loading test videos ({cfg['feat_set']})...")
        video_cache[cache_key_test] = load_video_frames(csv_path, feat_cols)
    vdata_test = video_cache[cache_key_test]
    test_vote = event_vote_eval(model, vdata_test, cfg["window_size"], threshold,
                                vote_window=5, vote_k=3, vel_idx=vel_idx, mn=mn, sc=sc)
    results["test"] = test_vote
    print(f"    test: event_vote MinP={test_vote['min_pr']:.4f}  FP={test_vote['fp']}  FN={test_vote['fn']}")

    m["event_vote_backfill"] = {
        "val":  results["val"],
        "test": results["test"],
        "vote_window": 5, "vote_k": 3,
        "threshold": threshold,
    }
    cfg["metrics_path"].write_text(json.dumps(m, indent=4))
    print(f"    → metrics.json updated")

    del model
    tf.keras.backend.clear_session()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dirs", nargs="*",
                        help="Specific result dirs to scan (default: all results/ subdirs)")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip experiments already backfilled or with native event_vote")
    args = parser.parse_args()

    if args.exp_dirs:
        scan_roots = [Path(d) for d in args.exp_dirs]
    else:
        scan_roots = sorted((REPO / "results").iterdir())

    # Collect all experiment directories
    exp_dirs = []
    for root in scan_roots:
        if not root.is_dir():
            continue
        # Check if root itself is an experiment dir (has metrics.json)
        if (root / "metrics.json").exists():
            exp_dirs.append(root)
        else:
            # Scan one level deeper
            for sub in sorted(root.iterdir()):
                if sub.is_dir() and (sub / "metrics.json").exists():
                    exp_dirs.append(sub)

    print(f"Found {len(exp_dirs)} experiment directories")
    video_cache: dict = {}

    for exp_dir in exp_dirs:
        try:
            run_backfill(exp_dir, args.skip_done, video_cache)
        except Exception as e:
            print(f"  ERROR {exp_dir.name}: {e}")

    print("\n=== Done ===")
    # Print summary table
    print(f"\n{'ID':<40} {'feat':<6} {'ws':>4} {'test_vote':>10} {'FP':>5} {'FN':>5}")
    print("-" * 70)
    for exp_dir in exp_dirs:
        try:
            m = json.loads((exp_dir / "metrics.json").read_text())
            bf = m.get("event_vote_backfill", {}).get("test", {})
            native = m.get("metrics", {}).get("test_event_vote", {})
            result = bf or native
            if result:
                feat = m.get("feature_set") or m.get("config", {}).get("feature_set", "?")
                ws   = m.get("config", {}).get("window_size") or 40
                print(f"  {exp_dir.name:<38} {feat:<6} {ws:>4} {result['min_pr']:>10.4f} {result.get('fp','?'):>5} {result.get('fn','?'):>5}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
