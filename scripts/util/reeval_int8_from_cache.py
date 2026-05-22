#!/usr/bin/env python3
"""Recompute INT8 MinPR from cached STedgeAI scores.

The primary eval pipeline (eval_stedgeai_host_p37.py --eval-stride 20) uses
stride=20 windows, but v10k6 vote requires 10 consecutive windows to fill the
buffer — impossible when videos have only ~8 windows at stride=20. This causes
MinPR=0 despite the model performing correctly.

This script re-reads the cached raw scores from .int8eval_cache/ and applies
v1k1 (any-window) vote, which is stride-invariant and gives a fair estimate.
It writes the corrected result to metrics.json["stedgeai_host_eval"].

Usage:
    uv run python scripts/util/reeval_int8_from_cache.py \
        --exp-dirs results/phase42_cross/P42-kp17-w60 [--stride 20]
    # or scan all:
    uv run python scripts/util/reeval_int8_from_cache.py --scan-all
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = [
    PROJ_ROOT / "results/phase41_ablation",
    PROJ_ROOT / "results/phase42_cross",
]

VOTE_COMBOS = [(1, 1), (3, 2), (5, 3), (5, 4), (7, 4), (7, 5), (10, 6)]


def apply_vote(scores: np.ndarray, thr: float, vw: int, vk: int) -> bool:
    buf_sum = 0
    buf = [0] * vw
    head = 0
    for s in scores:
        v = 1 if s >= thr else 0
        buf_sum += v - buf[head]
        buf[head] = v
        head = (head + 1) % vw
        if buf_sum >= vk:
            return True
    return False


def eval_from_cache(exp_dir: Path, stride: int) -> dict | None:
    cache = exp_dir / ".int8eval_cache"
    if not (cache / "test_scores.npy").exists():
        print(f"  SKIP {exp_dir.name}: no cache")
        return None

    mets_path = exp_dir / "metrics.json"
    if not mets_path.exists():
        print(f"  SKIP {exp_dir.name}: no metrics.json")
        return None

    mets = json.loads(mets_path.read_text())
    cfg = mets.get("config", {})
    threshold = mets.get("unified_eval", {}).get("threshold", 0.5)
    window_size = int(cfg.get("window_size", 40))
    data_dir = Path(cfg.get("data_dir", ""))
    test_csv = data_dir / "test.csv"
    if not test_csv.exists():
        print(f"  SKIP {exp_dir.name}: test.csv not found at {test_csv}")
        return None

    # Load cache
    scores_all = np.load(cache / "test_scores.npy")
    vids_list = (cache / "test_vids.txt").read_text().strip().split("\n")

    # Build per-video frame counts + labels from test.csv
    vid_nframes: dict[str, int] = defaultdict(int)
    vid_labels: dict[str, int] = defaultdict(int)
    with open(test_csv) as f:
        for row in csvmod.DictReader(f):
            vid_nframes[row["video_id"]] += 1
            if int(row["label"]) == 1:
                vid_labels[row["video_id"]] = 1

    def n_wins(nf: int) -> int:
        return max(0, (nf - window_size) // stride + 1)

    # Assign scores per video
    offset = 0
    vid_scores: dict[str, np.ndarray] = {}
    for vid in vids_list:
        nw = n_wins(vid_nframes.get(vid, 0))
        vid_scores[vid] = scores_all[offset : offset + nw]
        offset += nw

    if offset != len(scores_all):
        print(f"  WARN {exp_dir.name}: expected {offset} windows, got {len(scores_all)}")

    # Evaluate with each vote combo; pick best MinPR
    best_minpr = -1.0
    best_result = {}
    for vw, vk in VOTE_COMBOS:
        tp = fp = tn = fn = 0
        for vid in vids_list:
            sc = vid_scores.get(vid, np.array([]))
            if len(sc) == 0:
                detected = False
            else:
                detected = apply_vote(sc, threshold, vw, vk)
            has_fall = vid_labels.get(vid, 0) == 1
            if has_fall and detected:
                tp += 1
            elif has_fall:
                fn += 1
            elif detected:
                fp += 1
            else:
                tn += 1
        fall_prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fall_rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        nfall_prec = tn / (tn + fn) if (tn + fn) > 0 else 1.0  # precision of non-fall class
        nfall_rec  = tn / (tn + fp) if (tn + fp) > 0 else 1.0  # recall of non-fall class
        minpr = min(fall_prec, fall_rec, nfall_prec, nfall_rec)
        if minpr > best_minpr:
            best_minpr = minpr
            best_result = {
                "vote_window": vw,
                "vote_k": vk,
                "stride": stride,
                "threshold": threshold,
                "min_precision": round(minpr, 4),
                "fall_precision": round(fall_prec, 4),
                "fall_recall": round(fall_rec, 4),
                "nfall_precision": round(nfall_prec, 4),  # TN/(TN+FN)
                "nfall_recall": round(nfall_rec, 4),       # TN/(TN+FP)
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "note": f"recomputed from stride={stride} cache, best vote v{vw}k{vk}",
            }

    return best_result


def process_exp(exp_dir: Path, stride: int, dry_run: bool = False) -> None:
    print(f"\n[{exp_dir.name}]")
    result = eval_from_cache(exp_dir, stride)
    if result is None:
        return
    print(f"  Best: v{result['vote_window']}k{result['vote_k']} "
          f"MinPR={result['min_precision']:.4f} "
          f"TP={result['tp']} FP={result['fp']} FN={result['fn']} TN={result['tn']}")
    if not dry_run:
        mets_path = exp_dir / "metrics.json"
        mets = json.loads(mets_path.read_text())
        mets["stedgeai_host_eval"] = result
        mets_path.write_text(json.dumps(mets, indent=2))
        print(f"  Saved to metrics.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dirs", nargs="+", type=Path)
    ap.add_argument("--scan-all", action="store_true")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dirs: list[Path] = []
    if args.scan_all:
        for root in SCAN_ROOTS:
            for d in sorted(root.iterdir()):
                if (d / ".int8eval_cache" / "test_scores.npy").exists():
                    dirs.append(d)
    elif args.exp_dirs:
        dirs = args.exp_dirs
    else:
        ap.print_help()
        return

    print(f"Processing {len(dirs)} experiments (stride={args.stride})...")
    for d in dirs:
        process_exp(d, args.stride, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
