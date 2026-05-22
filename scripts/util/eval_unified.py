#!/usr/bin/env python3
"""Unified evaluation across all phases.

For every model found (Phase 1-40), computes:
  - window MinP : per-window threshold sweep on val, evaluate on test
  - event MinP  : threshold × min_consecutive=3 sweep on val, evaluate on test

Results saved as metrics["unified_eval"] in each metrics.json.

Usage:
  CUDA_VISIBLE_DEVICES=-1 uv run python -u scripts/eval_unified.py
  CUDA_VISIBLE_DEVICES=-1 uv run python -u scripts/eval_unified.py --skip-done
  CUDA_VISIBLE_DEVICES=-1 uv run python -u scripts/eval_unified.py \
      --exp-dirs results/phase27_seed_sweep results/phase36_window_ablation
"""
from __future__ import annotations

import argparse
import csv
import json
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

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_videos(csv_path: Path, feat_cols: list[str]) -> dict:
    """Returns {video_id: {"feat": [T,F], "label": int}} with video-level label."""
    raw: dict = defaultdict(lambda: {"rows": []})
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            lbl = int(row["label"])
            try:
                feat = [float(row[c]) for c in feat_cols]
            except (KeyError, ValueError):
                continue
            raw[vid]["rows"].append((feat, lbl))

    result = {}
    for vid, d in raw.items():
        feats  = np.array([r[0] for r in d["rows"]], dtype=np.float32)
        labels = [r[1] for r in d["rows"]]
        vid_label = int(any(l == 1 for l in labels))
        result[vid] = {"feat": feats, "vid_label": vid_label,
                       "frame_labels": labels}
    return result


def normalize(X: np.ndarray, mn: np.ndarray, sc: np.ndarray) -> np.ndarray:
    return (X - mn) / np.where(sc == 0, 1.0, sc)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_all_windows(model, video_data: dict, window_size: int,
                        mn: np.ndarray, sc: np.ndarray,
                        vel_idx=None) -> tuple[np.ndarray, np.ndarray, list]:
    """Returns (all_scores, all_win_labels, per_video_groups).

    per_video_groups: list of (scores_array, vid_label) per video
    all_win_labels: window label = last frame label of the window
    """
    all_scores, all_win_labels = [], []
    groups = []
    for vid, d in video_data.items():
        frames = d["feat"]  # (n, F) raw
        if vel_idx is not None:
            # velocity from raw features, then normalize jointly (norm has F+V dims)
            n = len(frames)
            vel = np.zeros((n, len(vel_idx)), np.float32)
            vel[1:] = frames[1:, vel_idx] - frames[:-1, vel_idx]
            frames = np.concatenate([frames, vel], axis=1)
        frames = normalize(frames, mn, sc)

        n = len(frames)
        if n < window_size:
            continue
        n_win = n - window_size + 1
        windows = np.stack([frames[i:i+window_size] for i in range(n_win)])
        scores  = model.predict(windows, batch_size=256, verbose=0)[:, 1]

        # window label = last frame label of the window
        flabels = d["frame_labels"]
        win_labels = np.array([flabels[i + window_size - 1] for i in range(n_win)],
                               dtype=np.int32)

        all_scores.extend(scores.tolist())
        all_win_labels.extend(win_labels.tolist())
        groups.append((scores, d["vid_label"]))

    return (np.array(all_scores, np.float32),
            np.array(all_win_labels, np.int32),
            groups)


# ── Window-level MinP ─────────────────────────────────────────────────────────

def window_minp(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fall_pr  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    nfall_pr = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {"min_pr": round(min(fall_pr, nfall_pr), 4),
            "fall_pr": round(fall_pr, 4), "nfall_pr": round(nfall_pr, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_window_threshold(scores_val, labels_val):
    best_thr, best_minp = THRESHOLDS[0], -1.0
    for thr in THRESHOLDS:
        m = window_minp(scores_val, labels_val, float(thr))
        if m["min_pr"] > best_minp:
            best_minp, best_thr = m["min_pr"], float(thr)
    return best_thr, best_minp


# ── Event-level MinP (threshold × min_consecutive=3) ─────────────────────────

def event_minp(groups: list, threshold: float, min_consec: int = 3) -> dict:
    tp = fp = fn = tn = 0
    for scores, vid_label in groups:
        preds = (scores >= threshold).astype(int)
        # consecutive rule
        count = 0
        detected = False
        for p in preds:
            count = count + 1 if p == 1 else 0
            if count >= min_consec:
                detected = True
                break
        if vid_label == 1:
            if detected: tp += 1
            else:        fn += 1
        else:
            if detected: fp += 1
            else:        tn += 1
    fall_pr  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    nfall_pr = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {"min_pr": round(min(fall_pr, nfall_pr), 4),
            "fall_pr": round(fall_pr, 4), "nfall_pr": round(nfall_pr, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_event_threshold(groups_val, min_consec=3):
    best_thr, best_minp = THRESHOLDS[0], -1.0
    for thr in THRESHOLDS:
        m = event_minp(groups_val, float(thr), min_consec)
        if m["min_pr"] > best_minp:
            best_minp, best_thr = m["min_pr"], float(thr)
    return best_thr, best_minp


# ── Config parsing ────────────────────────────────────────────────────────────

def parse_exp(exp_dir: Path):
    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    m = json.loads(metrics_path.read_text())

    if "config" in m:                          # Phase 36+ new format
        cfg = m["config"]
        feat_set    = cfg.get("feature_set", "kp13")
        window_size = cfg.get("window_size", 40)
        use_vel     = cfg.get("use_velocity", False)
        model_path  = exp_dir / "model_best.keras"
        if not model_path.exists():
            model_path = exp_dir / "model_proxy.keras"   # P40 stateful dirs
        if not model_path.exists():
            model_path = exp_dir / "model.keras"
    else:                                       # Phase 1-35 old format (+ P40 stateful)
        feat_set    = m.get("feature_set", "kp7")
        use_vel     = False
        model_path  = exp_dir / "model_proxy.keras"  # P40 stateful dirs
        if not model_path.exists():
            model_path = exp_dir / "model.keras"
        rc = exp_dir / "run_config.resolved.json"
        window_size = json.loads(rc.read_text()).get("target_steps", 40) \
                      if rc.exists() else 40

    if not model_path.exists():
        return None
    norm_path = exp_dir / "normalization.json"
    if not norm_path.exists():
        return None
    if feat_set not in FEATURE_SETS:
        return None

    return dict(feat_set=feat_set, window_size=window_size, use_vel=use_vel,
                model_path=model_path, norm_path=norm_path,
                metrics_path=metrics_path, m=m)


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_unified(exp_dir: Path, skip_done: bool, video_cache: dict):
    cfg = parse_exp(exp_dir)
    if cfg is None:
        return

    if skip_done and "unified_eval" in cfg["m"]:
        print(f"  SKIP {exp_dir.name}")
        return

    feat_cols = FEATURE_SETS[cfg["feat_set"]]
    vel_idx   = None
    if cfg["use_vel"]:
        vel_base = KP7_VEL_BASE if cfg["feat_set"] == "kp7" else KP13_VEL_BASE
        vel_idx  = [feat_cols.index(c) for c in vel_base]

    norm = json.loads(cfg["norm_path"].read_text())
    mn   = np.array(norm["min"],   np.float32)
    sc   = np.array(norm["scale"], np.float32)

    print(f"  EVAL {exp_dir.name}  feat={cfg['feat_set']}  ws={cfg['window_size']}")
    model = tf.keras.models.load_model(str(cfg["model_path"]), compile=False)

    # Predict on val (for threshold selection)
    cache_key = ("val", cfg["feat_set"], cfg["use_vel"])
    if cache_key not in video_cache:
        print(f"    Loading val ({cfg['feat_set']})...")
        video_cache[cache_key] = load_videos(DATA_DIR / "val.csv", feat_cols)
    scores_val, labels_val, groups_val = predict_all_windows(
        model, video_cache[cache_key], cfg["window_size"], mn, sc, vel_idx)

    win_thr, win_val  = best_window_threshold(scores_val, labels_val)
    ev_thr,  ev_val   = best_event_threshold(groups_val)

    print(f"    val sweep → win_thr={win_thr:.2f} val_win={win_val:.4f} | ev_thr={ev_thr:.2f} val_ev={ev_val:.4f}")

    # Evaluate on test
    cache_key_t = ("test", cfg["feat_set"], cfg["use_vel"])
    if cache_key_t not in video_cache:
        print(f"    Loading test ({cfg['feat_set']})...")
        video_cache[cache_key_t] = load_videos(DATA_DIR / "test.csv", feat_cols)
    scores_test, labels_test, groups_test = predict_all_windows(
        model, video_cache[cache_key_t], cfg["window_size"], mn, sc, vel_idx)

    win_test = window_minp(scores_test, labels_test, win_thr)
    ev_test  = event_minp(groups_test, ev_thr)

    print(f"    test  → win={win_test['min_pr']:.4f} FP={win_test['fp']} FN={win_test['fn']} | "
          f"ev={ev_test['min_pr']:.4f} FP={ev_test['fp']} FN={ev_test['fn']}")

    cfg["m"]["unified_eval"] = {
        "window": {"threshold": win_thr, "val_min_pr": round(win_val,4), **win_test},
        "event":  {"threshold": ev_thr,  "val_min_pr": round(ev_val,4),  **ev_test,
                   "min_consecutive": 3},
    }
    cfg["metrics_path"].write_text(json.dumps(cfg["m"], indent=4))
    print(f"    → saved")

    del model
    tf.keras.backend.clear_session()


def collect_exp_dirs(scan_roots: list[Path]) -> list[Path]:
    exp_dirs = []
    for root in scan_roots:
        if not root.is_dir():
            continue
        if (root / "metrics.json").exists():
            exp_dirs.append(root)
        else:
            for sub in sorted(root.iterdir()):
                if sub.is_dir() and (sub / "metrics.json").exists():
                    exp_dirs.append(sub)
    return exp_dirs


def print_summary(exp_dirs):
    print(f"\n{'ID':<40} {'feat':^6} {'ws':>4}  {'win_test':>9}  {'ev_test':>9}  {'FP_ev':>6} {'FN_ev':>6}")
    print("─" * 82)
    for exp_dir in exp_dirs:
        try:
            m = json.loads((exp_dir / "metrics.json").read_text())
            ue = m.get("unified_eval", {})
            if not ue:
                continue
            win = ue["window"]; ev = ue["event"]
            feat = m.get("feature_set") or m.get("config",{}).get("feature_set","?")
            ws   = m.get("config",{}).get("window_size", 40)
            print(f"  {exp_dir.name:<38} {feat:^6} {ws:>4}  "
                  f"{win['min_pr']:>9.4f}  {ev['min_pr']:>9.4f}  "
                  f"{ev['fp']:>6} {ev['fn']:>6}")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dirs", nargs="*")
    parser.add_argument("--skip-done", action="store_true")
    args = parser.parse_args()

    if args.exp_dirs:
        scan_roots = [Path(d) for d in args.exp_dirs]
    else:
        scan_roots = sorted((REPO / "results").iterdir())

    exp_dirs = collect_exp_dirs(scan_roots)
    print(f"Found {len(exp_dirs)} experiment directories")

    video_cache: dict = {}
    for exp_dir in exp_dirs:
        try:
            run_unified(exp_dir, args.skip_done, video_cache)
        except Exception as e:
            print(f"  ERROR {exp_dir.name}: {e}")

    print("\n=== Summary ===")
    print_summary(exp_dirs)


if __name__ == "__main__":
    main()
