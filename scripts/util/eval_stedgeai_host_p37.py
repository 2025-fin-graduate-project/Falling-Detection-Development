#!/usr/bin/env python3
"""STedgeAI --mode host INT8 evaluation for train_window_phase37.py experiments.

train_baseline.py 포맷(run_config.resolved.json)이 아닌
train_window_phase37.py 포맷(metrics.json["config"])용.

Usage:
    uv run python scripts/util/eval_stedgeai_host_p37.py \
        --exp-dir results/phase42_cross/P42-kp17-w60 \
        [--eval-stride 1]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

PROJ_ROOT = Path(__file__).resolve().parents[2]
STEDGEAI  = Path("/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/stedgeai")

VOTE_COMBOS = [(1,1),(3,2),(5,3),(5,4),(7,4),(7,5),(10,6)]


# ── 특징 컬럼 정의 (train_window_phase37.py 동일) ──────────────────────────────

def _kp_cols(indices):
    return [f"kp{i}_{ax}" for i in indices for ax in ("y","x","s")]

KP5_COLS  = _kp_cols([0,5,6,11,12])
KP7_COLS  = _kp_cols([0,5,6,11,12,15,16])
KP9_COLS  = _kp_cols([0,5,6,7,8,11,12,15,16])
KP11_COLS = _kp_cols([0,5,6,7,8,9,10,11,12,15,16])
KP13_COLS = _kp_cols([0,5,6,7,8,9,10,11,12,13,14,15,16])
KP17_COLS = _kp_cols(range(17))

# derived feature columns (appended after kp coords in filtered dataset)
DERIVED_FILT = ["HSSC_y","HSSC_x","RWHC","VHSSC","AHSSC","AHSSC_x"]

FEATURE_SETS = {
    "kp5":  KP5_COLS,
    "kp7":  KP7_COLS,
    "kp9":  KP9_COLS,
    "kp11": KP11_COLS,
    "kp13": KP13_COLS,
    "kp17": KP17_COLS,
}
VEL_BASE_MAP = {
    k: [c for c in v if c.endswith("_y") or c.endswith("_x")]
    for k, v in FEATURE_SETS.items()
}


def get_feature_cols(feature_set: str, data_dir: str) -> list[str]:
    """Return ordered feature column list matching training."""
    base = FEATURE_SETS[feature_set]
    # filtered dataset includes derived features
    if "filtered" in data_dir:
        return base + DERIVED_FILT
    return base


def add_velocity(frames: np.ndarray, vel_idx: list[int]) -> np.ndarray:
    vel = np.zeros_like(frames[:, vel_idx])
    vel[1:] = frames[1:, vel_idx] - frames[:-1, vel_idx]
    return np.concatenate([frames, vel], axis=1)


# ── CSV 로드 ───────────────────────────────────────────────────────────────────

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


# ── 窓 구축 + 정규화 ───────────────────────────────────────────────────────────

def build_video_windows(
    video_data: dict,
    window_size: int,
    vel_idx: list[int] | None,
    mn: np.ndarray,
    sc: np.ndarray,
    stride: int = 1,
) -> dict:
    """Returns {vid: (scores_placeholder, has_fall, windows np.float32)}."""
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
        result[vid] = (has_fall, wins, np.array(list(indices), dtype=np.int32), labels)
    return result


# ── STedgeAI validate ──────────────────────────────────────────────────────────

def run_validate_host(
    model_path: Path,
    name: str,
    windows_npy: Path,
    out_dir: Path,
    work_dir: Path,
) -> Path | None:
    cmd = [
        str(STEDGEAI), "validate",
        "--target", "stm32h7",
        "--model", str(model_path),
        "--type", "keras",
        "--name", name,
        "--output", str(out_dir),
        "--workspace", str(work_dir),
        "--mode", "host",
        "--classifier",
        "--save-csv",
        "--valinput", str(windows_npy),
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[stedgeai] validate failed:\n{result.stderr[-2000:]}", file=sys.stderr)
        return None
    csvs = list(out_dir.glob(f"{name}_val_c_outputs_*.csv"))
    if not csvs:
        csvs = list(out_dir.glob(f"{name}_val_output*.csv"))
    return csvs[0] if csvs else None


def load_predictions(csv_path: Path) -> np.ndarray:
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                vals = [float(v) for v in row if v.strip()]
            except ValueError:
                continue
            if len(vals) >= 2:
                rows.append(vals[:2])
    return np.array(rows, dtype=np.float32)


# ── 평가 지표 ─────────────────────────────────────────────────────────────────

def apply_vote_metrics(video_scores: dict, threshold: float, vote_window: int, vote_k: int) -> dict:
    tp = fp = fn = tn = 0
    for has_fall, scores in video_scores.values():
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
        if has_fall:
            tp += detected; fn += not detected
        else:
            fp += detected; tn += not detected
    e = 1e-9
    return {
        "min_pr":          round(min(tp/(tp+fp+e), tp/(tp+fn+e), tn/(tn+fn+e), tn/(tn+fp+e)), 4),
        "fall_precision":  round(tp/(tp+fp+e), 4),
        "fall_recall":     round(tp/(tp+fn+e), 4),
        "nfall_precision": round(tn/(tn+fn+e), 4),
        "nfall_recall":    round(tn/(tn+fp+e), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ── 메인 ───────────────────────────────────────────────────────────────────────

def evaluate_one(exp_dir: Path, eval_stride: int = 5, test_only: bool = False) -> dict | None:
    exp_dir = exp_dir.resolve()
    print(f"\n{'='*60}")
    print(f"[{exp_dir.name}]")

    met_path  = exp_dir / "metrics.json"
    norm_path = exp_dir / "normalization.json"
    fc_path   = exp_dir / "feature_columns.json"
    keras_path = exp_dir / "model_stedgeai_compat.keras"

    for p in [met_path, norm_path, fc_path]:
        if not p.exists():
            print(f"  SKIP — {p.name} not found"); return None
    if not keras_path.exists():
        print(f"  SKIP — model_stedgeai_compat.keras not found (run export_stedgeai.py first)")
        return None

    mets = json.loads(met_path.read_text())
    if "stedgeai_host_eval" in mets:
        print(f"  SKIP — already evaluated"); return mets["stedgeai_host_eval"]

    cfg  = mets.get("config", {})
    norm = json.loads(norm_path.read_text())
    feat_cols = json.loads(fc_path.read_text())

    feature_set  = cfg.get("feature_set", "kp13")
    window_size  = int(cfg.get("window_size", 40))
    use_velocity = cfg.get("use_velocity", False)
    data_dir     = Path(cfg.get("data_dir", ""))

    mn = np.array(norm["min"],   dtype=np.float32).reshape(1, 1, -1)
    sc = np.array(norm["scale"], dtype=np.float32).reshape(1, 1, -1)
    mn_flat = mn.reshape(-1)
    sc_flat = sc.reshape(-1)

    # velocity index — match training: use all _y/_x cols in feat_cols (includes derived)
    vel_idx = None
    if use_velocity:
        vel_idx = [i for i, c in enumerate(feat_cols) if c.endswith("_y") or c.endswith("_x")]

    test_csv = data_dir / "test.csv"
    val_csv  = data_dir / "val.csv"
    if not test_csv.exists():
        print(f"  SKIP — test CSV not found: {test_csv}"); return None

    print(f"  Loading test CSV ({test_csv.name}) ...")
    test_data = load_video_data(test_csv, feat_cols)
    if test_only:
        val_data = {}
        print(f"  test: {len(test_data)} videos  [val: skipped --test-only]")
    else:
        val_data = load_video_data(val_csv, feat_cols) if val_csv.exists() else {}
        print(f"  test: {len(test_data)} videos  val: {len(val_data)} videos")

    print(f"  Building windows (stride={eval_stride}) ...")
    test_vd = build_video_windows(test_data, window_size, vel_idx, mn_flat, sc_flat, stride=eval_stride)
    val_vd  = build_video_windows(val_data,  window_size, vel_idx, mn_flat, sc_flat, stride=eval_stride)

    name = exp_dir.name.lower().replace("-", "_")

    splits_to_run = [("test", test_vd)]
    if not test_only and val_vd:
        splits_to_run.append(("val", val_vd))

    # Persistent cache dir alongside metrics.json — survives script crashes
    cache_dir = exp_dir / ".int8eval_cache"
    cache_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_dir  = tmp_path / "out"
        work_dir = tmp_path / "work"
        out_dir.mkdir(); work_dir.mkdir()

        for split_name, vd in splits_to_run:
            cached_scores = cache_dir / f"{split_name}_scores.npy"
            cached_vids   = cache_dir / f"{split_name}_vids.txt"

            if cached_scores.exists() and cached_vids.exists():
                print(f"  Loading cached {split_name} scores ...")
                fall_scores = np.load(cached_scores)
                vid_order   = cached_vids.read_text().splitlines()
            else:
                all_wins = np.concatenate([wins for _, wins, _, _ in vd.values()], axis=0)
                npy_path = tmp_path / f"{split_name}_windows.npy"
                np.save(npy_path, all_wins)
                print(f"  STedgeAI validate --mode host ({split_name}, {len(all_wins):,} windows) ...")
                pred_csv = run_validate_host(keras_path, f"{name}_{split_name}", npy_path, out_dir, work_dir)
                if pred_csv is None:
                    print(f"  FAIL — validate returned no CSV"); return None
                preds = load_predictions(pred_csv)
                fall_scores = preds[:, 1]
                vid_order   = list(vd.keys())
                np.save(cached_scores, fall_scores)
                cached_vids.write_text("\n".join(vid_order))

            offset = 0
            for vid in vid_order:
                n = len(vd[vid][1])  # number of windows for this video
                vd[vid] = (vd[vid][0], fall_scores[offset:offset+n])
                offset += n

        # {vid: (has_fall, scores)}
        test_vscores = {v: (hf, sc_) for v, (hf, sc_) in test_vd.items()}
        val_vscores  = {v: (hf, sc_) for v, (hf, sc_) in val_vd.items()}

    # threshold from metrics.json (trained during float eval)
    trained_thr = mets.get("unified_eval", {}).get("threshold", 0.5)
    print(f"  Using trained threshold={trained_thr}")

    if test_only:
        # Use postproc params from float evaluation (already val-optimised)
        ev = mets.get("unified_eval", {}).get("event", {})
        fixed_vw = ev.get("vote_window", 10)
        fixed_vk = ev.get("vote_k", 6)
        print(f"  --test-only: fixed postproc v{fixed_vw}k{fixed_vk} (from float eval)")
        test_ev = apply_vote_metrics(test_vscores, trained_thr, fixed_vw, fixed_vk)
        pp_sweep = [{"vote_window": fixed_vw, "vote_k": fixed_vk,
                     "val": {}, "test": test_ev}]
        best_pp = pp_sweep[0]
    else:
        pp_sweep = []
        for vw, vk in VOTE_COMBOS:
            val_ev  = apply_vote_metrics(val_vscores,  trained_thr, vw, vk)
            test_ev = apply_vote_metrics(test_vscores, trained_thr, vw, vk)
            pp_sweep.append({"vote_window": vw, "vote_k": vk, "val": val_ev, "test": test_ev})
            print(f"  v{vw}k{vk}  val={val_ev['min_pr']:.4f}  test={test_ev['min_pr']:.4f}")
        best_pp = max(pp_sweep, key=lambda r: r["val"]["min_pr"])
    result = {
        "threshold":   trained_thr,
        "eval_stride": eval_stride,
        "min_precision":      best_pp["test"]["min_pr"],
        "fall_precision":     best_pp["test"]["fall_precision"],
        "fall_recall":        best_pp["test"]["fall_recall"],
        "nfall_precision":    best_pp["test"]["nfall_precision"],
        "nfall_recall":       best_pp["test"]["nfall_recall"],
        "vote_window":        best_pp["vote_window"],
        "vote_k":             best_pp["vote_k"],
        "val_min_precision":  best_pp["val"].get("min_pr"),
        "pp_sweep": pp_sweep,
    }
    print(f"  Best postproc: v{best_pp['vote_window']}k{best_pp['vote_k']}")
    print(f"  INT8 test MinPR = {result['min_precision']:.4f}")

    mets["stedgeai_host_eval"] = result
    met_path.write_text(json.dumps(mets, indent=2))
    print(f"  Saved to metrics.json")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", nargs="+", required=True)
    ap.add_argument("--eval-stride", type=int, default=5)
    ap.add_argument("--test-only", action="store_true",
                    help="Skip val split; use float-eval postproc params (faster)")
    args = ap.parse_args()

    for d in args.exp_dir:
        evaluate_one(Path(d), eval_stride=args.eval_stride, test_only=args.test_only)


if __name__ == "__main__":
    main()
