#!/usr/bin/env python3
"""Re-evaluate LB-3 models with three strategies:
  1. current   : max(P1, P2) > threshold  (binary, original)
  2. class1    : P(class_1) > threshold   (falling-only detection)
  3. statemachine: P(class_1) > t1 within window, then P(class_2) > t2 → alarm

Reads existing model.keras + val/test CSVs.
Runs with CUDA_VISIBLE_DEVICES="" (CPU-only, no CudnnRNNV3).

Usage:
    CUDA_VISIBLE_DEVICES="" uv run python scripts/util/eval_lb3_statemachine.py \
        --exp-dirs results/gru_baseline_phase1/P1-v05 \
                   results/gru_baseline_phase1/P1-v06 \
                   results/gru_baseline_phase1/P1-v07
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


# ── windowing ────────────────────────────────────────────────────────────────

def build_windows(df: pd.DataFrame, feature_cols: list[str],
                  win: int, start_sec: float, end_sec: float, stride: int = 1):
    """Returns (X, y_eval, y_3class, groups).

    y_eval:  binary — 1 if any frame in window has label_3class in {1,2}
             (mirrors train_baseline.py's eval_label for LB-3 with positive_labels=[1,2])
    y_3cls:  max label_3class in window (0/1/2)
    """
    windows, y_eval, y_3cls, groups = [], [], [], []
    for vid, grp in df.groupby("video_id", sort=False):
        seg = grp[(grp["time_sec"] >= start_sec) & (grp["time_sec"] < end_sec)].sort_values("frame")
        if len(seg) < win:
            continue
        vals  = seg[feature_cols].to_numpy(dtype=np.float32)
        l3    = seg["label_3class"].to_numpy(dtype=np.int32)
        leval = (l3 >= 1).astype(np.int32)  # 1 if class 1 or 2
        for i in range(0, len(seg) - win + 1, stride):
            windows.append(vals[i:i+win])
            y_eval.append(int(leval[i:i+win].max()))
            y_3cls.append(int(l3[i:i+win].max()))
            groups.append(str(vid))
    return (np.stack(windows),
            np.array(y_eval), np.array(y_3cls),
            np.array(groups))


# ── state machine ────────────────────────────────────────────────────────────

def state_machine_predict(p1: np.ndarray, p2: np.ndarray,
                          thresh1: float, thresh2: float,
                          lookforward: int) -> int:
    """Returns 1 if state machine fires (falling→fallen) for a video's window sequence."""
    state = 0  # 0=normal, 1=falling_detected
    since = 0
    for i, (q1, q2) in enumerate(zip(p1, p2)):
        if state == 0:
            if q1 >= thresh1:
                state = 1
                since = i
        elif state == 1:
            if q2 >= thresh2:
                return 1
            if i - since >= lookforward:
                state = 0  # timeout — reset
    return 0


def video_level_eval(y_true_win: np.ndarray, scores: np.ndarray,
                     groups: np.ndarray, threshold: float,
                     min_consec: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Standard binary consecutive-rule evaluation, returns (v_true, v_pred)."""
    vids = list(dict.fromkeys(groups))
    v_true, v_pred = [], []
    for vid in vids:
        mask = groups == vid
        gt = int(y_true_win[mask].max())
        scores_v = scores[mask]
        preds_v  = (scores_v >= threshold).astype(int)
        # consecutive rule
        alarm = 0
        consec = 0
        for p in preds_v:
            if p == 1:
                consec += 1
                if consec >= min_consec:
                    alarm = 1
                    break
            else:
                consec = 0
        v_true.append(gt)
        v_pred.append(alarm)
    return np.array(v_true), np.array(v_pred)


def video_level_statemachine(y_true_win: np.ndarray,
                              p1_all: np.ndarray, p2_all: np.ndarray,
                              groups: np.ndarray,
                              thresh1: float, thresh2: float,
                              lookforward: int) -> tuple[np.ndarray, np.ndarray]:
    vids = list(dict.fromkeys(groups))
    v_true, v_pred = [], []
    for vid in vids:
        mask = groups == vid
        gt = int(y_true_win[mask].max())
        pred = state_machine_predict(p1_all[mask], p2_all[mask],
                                     thresh1, thresh2, lookforward)
        v_true.append(gt)
        v_pred.append(pred)
    return np.array(v_true), np.array(v_pred)


def metrics(v_true, v_pred, label=""):
    f1  = f1_score(v_true, v_pred, zero_division=0)
    rec = recall_score(v_true, v_pred, zero_division=0)
    pre = precision_score(v_true, v_pred, zero_division=0)
    cm  = confusion_matrix(v_true, v_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    nfall_pre = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    min_pre   = min(pre, nfall_pre)
    return dict(label=label, f1=f1, recall=rec, fall_prec=pre,
                nfall_prec=nfall_pre, min_prec=min_pre,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


# ── main ─────────────────────────────────────────────────────────────────────

def evaluate_exp(exp_dir: Path, val_df: pd.DataFrame, test_df: pd.DataFrame):
    import sys, tensorflow as tf
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from train_baseline import TemporalAttention

    cfg_path    = exp_dir / "run_config.resolved.json"
    feat_path   = exp_dir / "feature_columns.json"
    norm_path   = exp_dir / "normalization.json"
    model_path  = exp_dir / "model.keras"
    orig_metrics = exp_dir / "metrics.json"

    cfg    = json.loads(cfg_path.read_text())
    feats  = json.loads(feat_path.read_text())
    norm   = json.loads(norm_path.read_text())

    WIN        = int(cfg.get("target_steps", 60))
    START_SEC  = float(cfg.get("window_start_sec", 5.0))
    END_SEC    = float(cfg.get("window_end_sec", 9.0))

    # normalise: stored as (min, scale=max-min), transform = (x - min) / scale
    norm_min   = np.array(norm["min"])
    norm_scale = np.array(norm["scale"])
    norm_scale = np.where(norm_scale < 1e-8, 1.0, norm_scale)

    def apply_norm(df_in):
        df2 = df_in.copy()
        df2[feats] = (df2[feats].to_numpy(dtype=np.float32) - norm_min) / norm_scale
        return df2

    val_n  = apply_norm(val_df)
    test_n = apply_norm(test_df)

    print(f"\n  Loading {model_path.name} ...", flush=True)
    model = tf.keras.models.load_model(str(model_path), custom_objects={"TemporalAttention": TemporalAttention})

    results = {}
    for split_name, df_n in [("val", val_n), ("test", test_n)]:
        X, y_bin, y_3cls, grps = build_windows(df_n, feats, WIN, START_SEC, END_SEC, stride=1)
        preds = model.predict(X, batch_size=256, verbose=0)  # (N, 3)

        p0, p1, p2 = preds[:, 0], preds[:, 1], preds[:, 2]
        score_binary = p1 + p2  # original LB-3 score

        # ── 1. Original approach (best threshold from saved metrics) ────────
        orig = json.loads(orig_metrics.read_text())
        orig_thresh  = float(orig["threshold_selection"]["threshold"])
        orig_mincn   = int(orig["threshold_selection"].get("min_consecutive", 1))
        v_true, v_pred = video_level_eval(y_bin, score_binary, grps, orig_thresh, orig_mincn)
        results[f"{split_name}_original"] = metrics(v_true, v_pred, f"original(t={orig_thresh:.2f}, cn={orig_mincn})")

        # ── 2. Class-1 only (P(falling) > threshold sweep) ─────────────────
        best_c1 = dict(f1=-1.0)
        for t in np.arange(0.05, 0.96, 0.05):
            for cn in [1, 3, 5]:
                vt, vp = video_level_eval(y_bin, p1, grps, t, cn)
                m = metrics(vt, vp)
                if m["f1"] > best_c1["f1"]:
                    best_c1 = m | {"thresh": round(float(t), 2), "mincn": cn}
        results[f"{split_name}_class1only"] = metrics(
            *video_level_eval(y_bin, p1, grps, best_c1["thresh"], best_c1["mincn"]),
            f"class1_only(t={best_c1['thresh']:.2f}, cn={best_c1['mincn']})"
        ) | {"thresh": best_c1["thresh"], "mincn": best_c1["mincn"]}

        # ── 3. State machine sweep ──────────────────────────────────────────
        best_sm = dict(f1=-1.0)
        for t1 in np.arange(0.05, 0.91, 0.10):
            for t2 in np.arange(0.20, 0.96, 0.10):
                for lf in [3, 5, 10]:
                    vt, vp = video_level_statemachine(y_bin, p1, p2, grps, t1, t2, lf)
                    m = metrics(vt, vp)
                    if m["f1"] > best_sm["f1"]:
                        best_sm = m | {"t1": round(float(t1),2), "t2": round(float(t2),2), "lf": lf}
        results[f"{split_name}_statemachine"] = metrics(
            *video_level_statemachine(y_bin, p1, p2, grps,
                                      best_sm["t1"], best_sm["t2"], best_sm["lf"]),
            f"statemachine(t1={best_sm['t1']:.2f},t2={best_sm['t2']:.2f},lf={best_sm['lf']})"
        ) | {"t1": best_sm["t1"], "t2": best_sm["t2"], "lf": best_sm["lf"]}

        # ── P(class_1) stats ────────────────────────────────────────────────
        results[f"{split_name}_p1_stats"] = {
            "max": float(p1.max()), "mean": float(p1.mean()),
            "p99": float(np.percentile(p1, 99)),
            "gt01": int((p1 > 0.01).sum()), "gt05": int((p1 > 0.05).sum()),
            "total_windows": len(p1)
        }

    return results


def print_results(exp_id: str, results: dict):
    print(f"\n{'='*65}")
    print(f"  {exp_id}")
    print(f"{'='*65}")
    for split in ["val", "test"]:
        p1s = results[f"{split}_p1_stats"]
        print(f"\n  [{split}] P(class_1) stats: max={p1s['max']:.5f}, "
              f"mean={p1s['mean']:.5f}, >0.01={p1s['gt01']}/{p1s['total_windows']}")
        for mode in ["original", "class1only", "statemachine"]:
            r = results[f"{split}_{mode}"]
            print(f"  [{split}] {mode:12s}: F1={r['f1']:.4f}  Rec={r['recall']:.4f}  "
                  f"fall_prec={r['fall_prec']:.4f}  nfall_prec={r['nfall_prec']:.4f}  "
                  f"min_prec={r['min_prec']:.4f}   ({r['label']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dirs", nargs="+", required=True)
    args = parser.parse_args()

    # Load datasets once
    all_results = {}
    for exp_dir_str in args.exp_dirs:
        exp_dir = Path(exp_dir_str)
        cfg = json.loads((exp_dir / "run_config.resolved.json").read_text())
        val_path  = cfg.get("val_csv")
        test_path = cfg.get("test_csv")
        if not val_path or not test_path:
            print(f"  {exp_dir.name}: missing CSV paths in config, skip")
            continue
        val_df  = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        res = evaluate_exp(exp_dir, val_df, test_df)
        print_results(exp_dir.name, res)
        all_results[exp_dir.name] = res

    # Summary table
    print(f"\n\n{'='*65}")
    print("SUMMARY — Test set")
    print(f"{'='*65}")
    print(f"{'ID':8s} {'mode':14s} {'F1':>7} {'Recall':>7} {'min_prec':>9}")
    print("-"*50)
    for exp_id, res in all_results.items():
        for mode in ["original", "class1only", "statemachine"]:
            r = res[f"test_{mode}"]
            print(f"{exp_id:8s} {mode:14s} {r['f1']:7.4f} {r['recall']:7.4f} {r['min_prec']:9.4f}")
        print()

    out_path = Path("results/gru_baseline_phase1/lb3_statemachine_eval.json")
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
