#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Keep this analysis from competing with the training run's GPU allocation.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf

import train_baseline as tb


@dataclass(frozen=True)
class Rule:
    name: str
    params: dict[str, Any]

    @property
    def label(self) -> str:
        compact = ",".join(f"{key}={value}" for key, value in self.params.items())
        return f"{self.name}({compact})"


def load_exp(exp_dir: Path) -> tuple[tb.BaselineConfig, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    cfg = json.loads((exp_dir / "run_config.resolved.json").read_text(encoding="utf-8"))
    config = tb.BaselineConfig(**cfg)
    project_root = Path(".").resolve()
    x, _, y_eval, groups, _, _, _, _, _ = tb.prepare_data(config, project_root, None)
    model = tf.keras.models.load_model(
        exp_dir / "model.keras",
        custom_objects={
            "SparseFocalLoss": tb.SparseFocalLoss,
            "TemporalAttention": tb.TemporalAttention,
        },
        compile=False,
    )
    pos_cols = np.array(config.positive_labels)
    scores = {}
    for split in ["val", "test"]:
        raw = model.predict(x[split], batch_size=config.batch_size, verbose=0)
        scores[split] = raw[:, pos_cols].sum(axis=1).astype(np.float32)
    return config, scores, y_eval, groups


def apply_consecutive(binary: np.ndarray, min_consecutive: int) -> np.ndarray:
    if min_consecutive <= 1:
        return binary.astype(np.int32)
    out = np.zeros_like(binary, dtype=np.int32)
    start = None
    for idx, value in enumerate(binary):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= min_consecutive:
                out[start:idx] = 1
            start = None
    if start is not None and len(binary) - start >= min_consecutive:
        out[start:] = 1
    return out


def hysteresis_hit(scores: np.ndarray, low: float, high: float, min_consecutive: int) -> bool:
    mask = scores >= low
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            run = scores[start:idx]
            if len(run) >= min_consecutive and float(run.max()) >= high:
                return True
            start = None
    if start is not None:
        run = scores[start:]
        if len(run) >= min_consecutive and float(run.max()) >= high:
            return True
    return False


VideoData = list[tuple[int, np.ndarray]]


def build_video_data(y_true: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> VideoData:
    video_ids = np.unique(groups)
    videos: VideoData = []
    for video_id in video_ids:
        mask = groups == video_id
        videos.append((int(y_true[mask].max()), scores[mask]))
    return videos


def video_predictions(videos: VideoData, rule: Rule) -> tuple[np.ndarray, np.ndarray]:
    v_true = np.empty(len(videos), dtype=np.int32)
    v_pred = np.empty(len(videos), dtype=np.int32)
    for idx, (label, ss) in enumerate(videos):
        v_true[idx] = label
        p = rule.params
        if rule.name == "consecutive":
            binary = (ss >= p["threshold"]).astype(np.int32)
            v_pred[idx] = int(apply_consecutive(binary, p["min_consecutive"]).max())
        elif rule.name == "consecutive_count":
            binary = (ss >= p["threshold"]).astype(np.int32)
            filtered = apply_consecutive(binary, p["min_consecutive"])
            v_pred[idx] = int(filtered.max() and int(binary.sum()) >= p["min_count"])
        elif rule.name == "consecutive_ratio":
            binary = (ss >= p["threshold"]).astype(np.int32)
            filtered = apply_consecutive(binary, p["min_consecutive"])
            v_pred[idx] = int(filtered.max() and float(binary.mean()) >= p["min_ratio"])
        elif rule.name == "topk_mean":
            k = min(int(p["top_k"]), len(ss))
            top = np.partition(ss, -k)[-k:]
            v_pred[idx] = int(float(top.mean()) >= p["threshold"])
        elif rule.name == "hysteresis":
            v_pred[idx] = int(hysteresis_hit(ss, p["low"], p["high"], p["min_consecutive"]))
        else:
            raise ValueError(f"Unknown rule: {rule.name}")
    return v_true, v_pred


def metrics(v_true: np.ndarray, v_pred: np.ndarray) -> dict[str, Any]:
    tp = int(((v_true == 1) & (v_pred == 1)).sum())
    tn = int(((v_true == 0) & (v_pred == 0)).sum())
    fp = int(((v_true == 0) & (v_pred == 1)).sum())
    fn = int(((v_true == 1) & (v_pred == 0)).sum())
    fall_p = tp / (tp + fp) if tp + fp else 0.0
    nfall_p = tn / (tn + fn) if tn + fn else 0.0
    fall_r = tp / (tp + fn) if tp + fn else 0.0
    nfall_r = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * fall_p * fall_r / (fall_p + fall_r) if fall_p + fall_r else 0.0
    return {
        "min_pr": min(fall_p, nfall_p, fall_r, nfall_r),
        "fall_precision": fall_p,
        "nfall_precision": nfall_p,
        "fall_recall": fall_r,
        "nfall_recall": nfall_r,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def build_rules() -> list[Rule]:
    rules: list[Rule] = []
    thresholds = [round(float(v), 3) for v in np.arange(0.35, 0.751, 0.025)]
    for threshold in thresholds:
        for mc in range(1, 10):
            rules.append(Rule("consecutive", {"threshold": threshold, "min_consecutive": mc}))
        for mc in range(1, 8):
            for count in range(2, 13):
                rules.append(Rule("consecutive_count", {"threshold": threshold, "min_consecutive": mc, "min_count": count}))
            for ratio in [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.20]:
                rules.append(Rule("consecutive_ratio", {"threshold": threshold, "min_consecutive": mc, "min_ratio": ratio}))
        for top_k in [2, 3, 4, 5, 7, 9]:
            rules.append(Rule("topk_mean", {"threshold": threshold, "top_k": top_k}))
    for high in [round(float(v), 3) for v in np.arange(0.45, 0.851, 0.025)]:
        for gap in [0.05, 0.10, 0.15, 0.20]:
            low = round(max(0.05, high - gap), 3)
            for mc in range(2, 10):
                rules.append(Rule("hysteresis", {"low": low, "high": high, "min_consecutive": mc}))
    return rules


def evaluate_rule(rule: Rule, videos: dict[str, VideoData]) -> dict[str, Any]:
    result = {"rule": rule.name, "params": rule.params, "label": rule.label}
    for split in ["val", "test"]:
        v_true, v_pred = video_predictions(videos[split], rule)
        result[split] = metrics(v_true, v_pred)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Phase 10 video-level postprocessing rules for an existing model.")
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase10_postprocess_sweep"))
    parser.add_argument("--target-min-pr", type=float, default=0.92)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config, scores, y_eval, groups = load_exp(args.exp_dir)
    videos = {
        split: build_video_data(y_eval[split], scores[split], groups[split])
        for split in ["val", "test"]
    }
    rules = build_rules()
    rows = [evaluate_rule(rule, videos) for rule in rules]
    rows.sort(
        key=lambda item: (
            item["val"]["min_pr"],
            item["val"]["fall_recall"],
            item["val"]["fall_precision"],
            item["val"]["f1"],
        ),
        reverse=True,
    )
    best = rows[0]
    out_json = args.output_dir / f"{args.exp_dir.name}_postprocess_sweep.json"
    out_json.write_text(json.dumps({"experiment": args.exp_dir.name, "best": best, "rows": rows}, indent=2), encoding="utf-8")

    out_csv = args.output_dir / f"{args.exp_dir.name}_postprocess_sweep.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rank", "rule", "params",
            "val_min_pr", "val_fall_p", "val_nfall_p", "val_fall_r", "val_nfall_r", "val_fp", "val_fn",
            "test_min_pr", "test_fall_p", "test_nfall_p", "test_fall_r", "test_nfall_r", "test_fp", "test_fn",
        ])
        for rank, row in enumerate(rows, start=1):
            writer.writerow([
                rank, row["rule"], json.dumps(row["params"], sort_keys=True),
                row["val"]["min_pr"], row["val"]["fall_precision"], row["val"]["nfall_precision"],
                row["val"]["fall_recall"], row["val"]["nfall_recall"], row["val"]["fp"], row["val"]["fn"],
                row["test"]["min_pr"], row["test"]["fall_precision"], row["test"]["nfall_precision"],
                row["test"]["fall_recall"], row["test"]["nfall_recall"], row["test"]["fp"], row["test"]["fn"],
            ])

    print(f"experiment={args.exp_dir.name} rules={len(rows)}")
    print(f"best_by_val={best['label']}")
    for split in ["val", "test"]:
        m = best[split]
        print(
            f"{split}: MinPR={m['min_pr']:.4f} FallP={m['fall_precision']:.4f} "
            f"NFallP={m['nfall_precision']:.4f} FallR={m['fall_recall']:.4f} "
            f"NFallR={m['nfall_recall']:.4f} FP={m['fp']} FN={m['fn']}"
        )
    print(f"\ntop {args.top} by val MinPR")
    for idx, row in enumerate(rows[: args.top], start=1):
        vm = row["val"]
        tm = row["test"]
        print(
            f"{idx:02d} {row['label']} | "
            f"val={vm['min_pr']:.4f} fp={vm['fp']} fn={vm['fn']} | "
            f"test={tm['min_pr']:.4f} fp={tm['fp']} fn={tm['fn']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
