#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        PrecisionRecallDisplay,
        RocCurveDisplay,
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ModuleNotFoundError as exc:
    plt = np = pd = tf = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


ENGINEERED_RAW = ["HSSC_y", "HSSC_x", "RWHC", "VHSSC"]
ENGINEERED_FILTERED = ["HSSC_y", "HSSC_x", "RWHC", "VHSSC", "AHSSC", "AHSSC_x"]
FEATURE_SETS = {
    "minimal": [0, 5, 6, 11, 12],
    "kp7": [0, 5, 6, 7, 8, 11, 12],
    "kp8": [0, 5, 6, 7, 8, 9, 11, 12],
    "kp12": list(range(13)),
    "all": list(range(17)),
}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def ensure_deps() -> None:
    if IMPORT_ERROR is not None:
        raise SystemExit(
            f"Missing Python dependency: {IMPORT_ERROR.name}. "
            "Install project training dependencies first."
        ) from IMPORT_ERROR


def parse_csv_ints(raw: str | list[int]) -> list[int]:
    if isinstance(raw, list):
        return [int(value) for value in raw]
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


@dataclass
class BaselineConfig:
    experiment_id: str
    model_type: str
    preprocessing: str
    source_csv: str = ""
    input_mode: str = "split_csv"
    split_dir: str = "dataset/splits"
    train_csv: str = ""
    val_csv: str = ""
    test_csv: str = ""
    output_root: str = "results/baselines_phase0"
    feature_set: str = "kp12"
    label_column: str = "label"
    positive_labels: list[int] = field(default_factory=lambda: [1])
    label_mode: str = "segment_max"
    data_scope: str = "no_by"
    window_start_sec: float = 5.0
    window_end_sec: float = 9.0
    target_steps: int = 60
    train_positive_stride: int = 1
    train_negative_stride: int = 5
    eval_stride: int = 1
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    seed: int = 42
    dropout_rate: float = 0.2
    early_stop_patience: int = 10
    threshold_count: int = 19
    min_val_recall: float = 0.0
    min_val_precision: float = 0.0
    min_consecutive_values: list[int] = field(default_factory=lambda: [1, 3, 5])
    tcn_channels: list[int] = field(default_factory=lambda: [32, 32, 64, 96])
    tcn_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    tcn_kernel_size: int = 3
    gru_units: list[int] = field(default_factory=lambda: [64, 32])
    conv_pre_layers: int = 0
    conv_pre_filters: int = 64
    conv_pre_kernel: int = 5
    bidirectional: bool = False
    temporal_attention: bool = False
    num_classes: int = 2
    focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    noise_std: float = 0.0
    export_tflite: bool = True
    quantize_int8: bool = True
    representative_samples: int = 256
    quant_eval_max_windows: int = 5000
    max_rows: int | None = None
    max_windows_per_split: int | None = None
    quiet: bool = False


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configurable TCN/GRU fall-detection baseline.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root")
    parser.add_argument("--experiment-id")
    parser.add_argument("--model-type", choices=["tcn", "gru"])
    parser.add_argument("--preprocessing", choices=["raw", "filtered"])
    parser.add_argument("--source-csv")
    parser.add_argument("--input-mode", choices=["split_csv", "source_csv"])
    parser.add_argument("--split-dir")
    parser.add_argument("--train-csv")
    parser.add_argument("--val-csv")
    parser.add_argument("--test-csv")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS))
    parser.add_argument("--label-column")
    parser.add_argument("--positive-labels")
    parser.add_argument("--label-mode", choices=["segment_max", "last_frame"])
    parser.add_argument("--data-scope", choices=["all", "no_by"], default=None)
    parser.add_argument("--window-start-sec", type=float)
    parser.add_argument("--window-end-sec", type=float)
    parser.add_argument("--target-steps", type=int)
    parser.add_argument("--train-positive-stride", type=int)
    parser.add_argument("--train-negative-stride", type=int)
    parser.add_argument("--eval-stride", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dropout-rate", type=float)
    parser.add_argument("--early-stop-patience", type=int)
    parser.add_argument("--threshold-count", type=int)
    parser.add_argument("--min-val-recall", type=float)
    parser.add_argument("--min-val-precision", type=float)
    parser.add_argument("--min-consecutive-values", help="Comma-separated min-consecutive window counts to sweep (default: 1,3,5).")
    parser.add_argument("--tcn-channels")
    parser.add_argument("--tcn-dilations")
    parser.add_argument("--tcn-kernel-size", type=int)
    parser.add_argument("--gru-units")
    parser.add_argument("--conv-pre-layers", type=int)
    parser.add_argument("--conv-pre-filters", type=int)
    parser.add_argument("--conv-pre-kernel", type=int)
    parser.add_argument("--representative-samples", type=int)
    parser.add_argument("--quant-eval-max-windows", type=int)
    parser.add_argument("--export-tflite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--quantize-int8", action=argparse.BooleanOptionalAction)
    parser.add_argument("--smoke", action="store_true", help="Use small limits for a Colab/local smoke run.")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-windows-per-split", type=int)
    parser.add_argument("--quiet", action="store_true", help="Suppress model summary and epoch-level training output.")
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, help="Use Bidirectional GRU.")
    parser.add_argument("--temporal-attention", action=argparse.BooleanOptionalAction, help="Add additive temporal attention pooling after GRU layers (TFLite INT8 compatible).")
    parser.add_argument("--num-classes", type=int, help="Number of output classes (2=binary LB-2, 3=3-class LB-3). Positive classes are still defined by --positive-labels.")
    parser.add_argument("--focal-loss", action=argparse.BooleanOptionalAction, help="Use focal loss instead of cross-entropy.")
    parser.add_argument("--focal-alpha", type=float)
    parser.add_argument("--focal-gamma", type=float)
    parser.add_argument("--noise-std", type=float, help="Gaussian noise std for training augmentation (0 = off).")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> BaselineConfig:
    payload = load_config(args.config)
    overrides = {
        "experiment_id": args.experiment_id,
        "model_type": args.model_type,
        "preprocessing": args.preprocessing,
        "source_csv": args.source_csv,
        "input_mode": args.input_mode,
        "split_dir": args.split_dir,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "output_root": args.output_root,
        "feature_set": args.feature_set,
        "label_column": args.label_column,
        "label_mode": args.label_mode,
        "data_scope": args.data_scope,
        "window_start_sec": args.window_start_sec,
        "window_end_sec": args.window_end_sec,
        "target_steps": args.target_steps,
        "train_positive_stride": args.train_positive_stride,
        "train_negative_stride": args.train_negative_stride,
        "eval_stride": args.eval_stride,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "dropout_rate": args.dropout_rate,
        "early_stop_patience": args.early_stop_patience,
        "threshold_count": args.threshold_count,
        "min_val_recall": args.min_val_recall,
        "min_val_precision": args.min_val_precision,
        "tcn_kernel_size": args.tcn_kernel_size,
        "conv_pre_layers": args.conv_pre_layers,
        "conv_pre_filters": args.conv_pre_filters,
        "conv_pre_kernel": args.conv_pre_kernel,
        "representative_samples": args.representative_samples,
        "quant_eval_max_windows": args.quant_eval_max_windows,
        "max_rows": args.max_rows,
        "max_windows_per_split": args.max_windows_per_split,
    }
    for key, value in overrides.items():
        if value is not None:
            payload[key] = value
    if args.positive_labels is not None:
        payload["positive_labels"] = parse_csv_ints(args.positive_labels)
    if args.tcn_channels is not None:
        payload["tcn_channels"] = parse_csv_ints(args.tcn_channels)
    if args.tcn_dilations is not None:
        payload["tcn_dilations"] = parse_csv_ints(args.tcn_dilations)
    if args.gru_units is not None:
        payload["gru_units"] = parse_csv_ints(args.gru_units)
    if args.export_tflite is not None:
        payload["export_tflite"] = args.export_tflite
    if args.quantize_int8 is not None:
        payload["quantize_int8"] = args.quantize_int8
    if args.smoke:
        payload["epochs"] = min(int(payload.get("epochs", 1)), 1)
        payload["max_windows_per_split"] = int(payload.get("max_windows_per_split") or 512)
        payload["representative_samples"] = min(int(payload.get("representative_samples", 64)), 64)
        payload["quant_eval_max_windows"] = min(int(payload.get("quant_eval_max_windows", 256)), 256)
    if args.quiet:
        payload["quiet"] = True
    if args.bidirectional is not None:
        payload["bidirectional"] = args.bidirectional
    if args.temporal_attention is not None:
        payload["temporal_attention"] = args.temporal_attention
    if args.num_classes is not None:
        payload["num_classes"] = args.num_classes
    if args.focal_loss is not None:
        payload["focal_loss"] = args.focal_loss
    if args.focal_alpha is not None:
        payload["focal_alpha"] = args.focal_alpha
    if args.focal_gamma is not None:
        payload["focal_gamma"] = args.focal_gamma
    if args.noise_std is not None:
        payload["noise_std"] = args.noise_std
    if args.min_consecutive_values is not None:
        payload["min_consecutive_values"] = parse_csv_ints(args.min_consecutive_values)

    required = ["experiment_id", "model_type", "preprocessing"]
    if payload.get("input_mode", "split_csv") == "source_csv":
        required.append("source_csv")
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    payload["positive_labels"] = parse_csv_ints(payload.get("positive_labels", [1]))
    config = BaselineConfig(**payload)
    if config.model_type not in {"tcn", "gru"}:
        raise ValueError("--model-type must be tcn or gru.")
    if config.preprocessing not in {"raw", "filtered"}:
        raise ValueError("--preprocessing must be raw or filtered.")
    if config.input_mode not in {"split_csv", "source_csv"}:
        raise ValueError("--input-mode must be split_csv or source_csv.")
    if config.feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set={config.feature_set}.")
    return config


def resolve_path(project_root: Path, data_root: Path | None, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if data_root is not None and raw_path.startswith("dataset/"):
        return data_root / Path(raw_path).relative_to("dataset")
    return project_root / path


def infer_direction(video_id: str) -> str:
    parts = str(video_id).split("_")
    for part in parts:
        if part in {"BY", "FY", "SY", "N"}:
            return part
    return "UNKNOWN"


def feature_columns(columns: list[str], feature_set: str, preprocessing: str) -> list[str]:
    kp_indexes = FEATURE_SETS[feature_set]
    kp_cols = [
        f"kp{idx}_{axis}"
        for idx in kp_indexes
        for axis in ("y", "x", "s")
        if f"kp{idx}_{axis}" in columns
    ]
    engineered = ENGINEERED_FILTERED if preprocessing == "filtered" else ENGINEERED_RAW
    engineered_cols = [col for col in engineered if col in columns]
    if not kp_cols:
        raise ValueError("No keypoint feature columns found.")
    return kp_cols + engineered_cols


def load_split_video_ids(split_dir: Path) -> dict[str, set[str]]:
    split_ids: dict[str, set[str]] = {}
    for split in ["train", "val", "test"]:
        path = split_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Split CSV not found: {path}")
        ids = pd.read_csv(path, usecols=["video_id"])["video_id"].astype(str)
        split_ids[split] = set(ids.unique().tolist())
    return split_ids


def normalize_source_frame(df: pd.DataFrame, config: BaselineConfig) -> tuple[pd.DataFrame, list[str]]:
    if config.label_column not in df.columns:
        raise ValueError(f"Missing label column: {config.label_column}")
    required = {"video_id", "frame", "time_sec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    cols = feature_columns(df.columns.tolist(), config.feature_set, config.preprocessing)
    keep_cols = ["video_id", "frame", "time_sec", config.label_column] + cols
    if "direction" in df.columns:
        keep_cols.append("direction")
    df = df[keep_cols].replace([np.inf, -np.inf], np.nan)
    df = df.rename(columns={config.label_column: "source_label"})
    df["video_id"] = df["video_id"].astype(str)
    if "direction" not in df.columns:
        df["direction"] = df["video_id"].map(infer_direction)
    else:
        df["direction"] = df["direction"].astype(str)
    if config.data_scope == "no_by":
        df = df[df["direction"] != "BY"].copy()
    # eval_label: binary (1 = any positive class) — used for threshold selection and metrics
    df["eval_label"] = df["source_label"].isin(config.positive_labels).astype(np.int32)
    # label: training label — raw multi-class for LB-3, binary for LB-2
    if config.num_classes > 2:
        df["label"] = df["source_label"].clip(0, config.num_classes - 1).astype(np.int32)
    else:
        df["label"] = df["eval_label"]
    if df[cols].isna().any().any():
        df[cols] = df[cols].fillna(0.0)
    return df[["video_id", "frame", "time_sec", "direction", "label", "eval_label"] + cols], cols


def load_source_frame(config: BaselineConfig, project_root: Path, data_root: Path | None) -> tuple[pd.DataFrame, list[str]]:
    source_path = resolve_path(project_root, data_root, config.source_csv)
    if not source_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_path}")
    read_kwargs: dict[str, Any] = {"low_memory": False}
    if config.max_rows is not None:
        read_kwargs["nrows"] = config.max_rows
    log(f"loading source={source_path}")
    df = pd.read_csv(source_path, **read_kwargs)
    df, cols = normalize_source_frame(df, config)
    log(
        f"rows={len(df):,} videos={df['video_id'].nunique():,} "
        f"features={len(cols)} data_scope={config.data_scope}"
    )
    return df[["video_id", "frame", "time_sec", "direction", "label"] + cols], cols


def load_split_frames(
    config: BaselineConfig,
    project_root: Path,
    data_root: Path | None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    split_dir = resolve_path(project_root, data_root, config.split_dir)
    split_paths = {
        "train": config.train_csv,
        "val": config.val_csv,
        "test": config.test_csv,
    }
    frames: dict[str, pd.DataFrame] = {}
    feature_cols: list[str] | None = None
    for split in ["train", "val", "test"]:
        path = resolve_path(project_root, data_root, split_paths[split]) if split_paths[split] else split_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Split CSV not found: {path}")
        read_kwargs: dict[str, Any] = {"low_memory": False}
        if config.max_rows is not None:
            read_kwargs["nrows"] = config.max_rows
        log(f"loading {split} split={path}")
        df = pd.read_csv(path, **read_kwargs)
        normalized, cols = normalize_source_frame(df, config)
        if feature_cols is None:
            feature_cols = cols
        elif feature_cols != cols:
            raise ValueError(f"Feature columns differ in split {split}.")
        frames[split] = normalized
        log(
            f"{split} rows={len(normalized):,} videos={normalized['video_id'].nunique():,} "
            f"features={len(cols)} data_scope={config.data_scope}"
        )
    if feature_cols is None:
        raise ValueError("No split frames were loaded.")
    return frames, feature_cols


def window_label(labels: np.ndarray, mode: str) -> int:
    if mode == "segment_max":
        return int(labels.max())
    if mode == "last_frame":
        return int(labels[-1])
    raise ValueError(f"Unsupported label_mode={mode}")


def build_windows_for_split(
    df: pd.DataFrame,
    split_ids: set[str],
    feature_cols: list[str],
    config: BaselineConfig,
    *,
    training: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    split_df = df[df["video_id"].isin(split_ids)].sort_values(["video_id", "time_sec", "frame"])
    windows: list[np.ndarray] = []
    labels: list[int] = []        # multi-class training label (0/1/2 for LB-3)
    eval_labels: list[int] = []   # binary evaluation label (0 or 1)
    groups: list[str] = []
    directions: list[str] = []
    dist_rows: list[dict[str, Any]] = []
    for video_id, group in split_df.groupby("video_id", sort=False):
        segment = group[
            (group["time_sec"] >= config.window_start_sec)
            & (group["time_sec"] < config.window_end_sec)
        ]
        if len(segment) < config.target_steps:
            continue
        values = segment[feature_cols].to_numpy(dtype=np.float32)
        frame_labels = segment["label"].to_numpy(dtype=np.int32)
        frame_eval_labels = segment["eval_label"].to_numpy(dtype=np.int32)
        direction = str(segment["direction"].iloc[0])
        for start_idx in range(0, len(segment) - config.target_steps + 1):
            y = window_label(frame_labels[start_idx : start_idx + config.target_steps], config.label_mode)
            y_eval = window_label(frame_eval_labels[start_idx : start_idx + config.target_steps], config.label_mode)
            stride = config.eval_stride
            if training:
                # Use binary eval label for stride so positive stride applies to any fall-related class
                stride = config.train_positive_stride if y_eval == 1 else config.train_negative_stride
            if start_idx % stride != 0:
                continue
            chunk = values[start_idx : start_idx + config.target_steps]
            windows.append(chunk)
            labels.append(y)
            eval_labels.append(y_eval)
            groups.append(str(video_id))
            directions.append(direction)
            dist_rows.append(
                {
                    "video_id": str(video_id),
                    "direction": direction,
                    "label": y_eval,
                    "fall_ratio": float(frame_eval_labels[start_idx : start_idx + config.target_steps].mean()),
                    "confidence_mean": float(np.nanmean(chunk[:, 2::3])) if chunk.shape[1] >= 3 else 0.0,
                }
            )
            if config.max_windows_per_split and len(windows) >= config.max_windows_per_split:
                break
        if config.max_windows_per_split and len(windows) >= config.max_windows_per_split:
            break
    if not windows:
        raise ValueError("No windows were created. Check split, time range, and target_steps.")
    return (
        np.stack(windows).astype(np.float32),
        np.asarray(labels, dtype=np.int32),       # y_train: multi-class
        np.asarray(eval_labels, dtype=np.int32),  # y_eval:  binary
        np.asarray(groups),
        np.asarray(directions),
        pd.DataFrame(dist_rows),
    )


def minmax_normalize(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_v = x_train.min(axis=(0, 1), keepdims=True)
    max_v = x_train.max(axis=(0, 1), keepdims=True)
    scale = np.where((max_v - min_v) < 1e-6, 1.0, max_v - min_v)
    return (
        ((x_train - min_v) / scale).astype(np.float32),
        ((x_val - min_v) / scale).astype(np.float32),
        ((x_test - min_v) / scale).astype(np.float32),
        min_v.astype(np.float32),
        scale.astype(np.float32),
    )


def prepare_data(
    config: BaselineConfig,
    project_root: Path,
    data_root: Path | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], list[str], np.ndarray, np.ndarray, pd.DataFrame]:
    if config.input_mode == "split_csv":
        split_frames, cols = load_split_frames(config, project_root, data_root)
    else:
        df, cols = load_source_frame(config, project_root, data_root)
        split_dir = resolve_path(project_root, data_root, config.split_dir)
        split_ids = load_split_video_ids(split_dir)
        split_frames = {split: df[df["video_id"].isin(split_ids[split])] for split in ["train", "val", "test"]}
    built = {}
    for split in ["train", "val", "test"]:
        built[split] = build_windows_for_split(
            split_frames[split],
            set(split_frames[split]["video_id"].unique().tolist()),
            cols,
            config,
            training=(split == "train"),
        )
        # built[split]: (x, y_train, y_eval, groups, directions, dist_df)
        log(
            f"{split} windows={len(built[split][1]):,} "
            f"positive={int(built[split][2].sum()):,} videos={len(np.unique(built[split][3])):,}"
        )
    x_train, x_val, x_test, min_v, scale = minmax_normalize(built["train"][0], built["val"][0], built["test"][0])
    x      = {"train": x_train, "val": x_val, "test": x_test}
    y      = {split: built[split][1] for split in ["train", "val", "test"]}  # multi-class (model training)
    y_eval = {split: built[split][2] for split in ["train", "val", "test"]}  # binary     (evaluation)
    groups     = {split: built[split][3] for split in ["train", "val", "test"]}
    directions = {split: built[split][4] for split in ["train", "val", "test"]}
    dist = pd.concat(
        [built[split][5].assign(split=split) for split in ["train", "val", "test"]],
        ignore_index=True,
    )
    return x, y, y_eval, groups, directions, cols, min_v, scale, dist


def class_weight_from_labels(y: np.ndarray) -> dict[int, float]:
    n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    total = counts.sum()
    return {c: float(total / (n_classes * max(counts[c], 1.0))) for c in range(n_classes)}


def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, training: bool, noise_std: float = 0.0) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(len(x), 100_000), reshuffle_each_iteration=True)
        if noise_std > 0.0:
            def add_noise(xb: tf.Tensor, yb: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
                noise = tf.random.normal(tf.shape(xb), stddev=noise_std)
                return tf.clip_by_value(xb + noise, 0.0, 1.0), yb
            ds = ds.map(add_noise, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class SparseFocalLoss(tf.keras.losses.Loss):
    """Focal loss for sparse integer labels, compatible with 2-class softmax output."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # Gather the predicted probability for the true class
        idx = tf.stack([tf.range(tf.shape(y_true)[0]), y_true], axis=1)
        p_t = tf.gather_nd(y_pred, idx)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * tf.pow(1.0 - p_t, self.gamma)
        ce = -tf.math.log(p_t)
        return focal_weight * ce

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "alpha": self.alpha, "gamma": self.gamma}


def residual_tcn_block(x: tf.Tensor, filters: int, kernel_size: int, dilation: int, dropout: float, name: str) -> tf.Tensor:
    shortcut = x
    y = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation, use_bias=False, name=f"{name}_conv1")(x)
    y = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = tf.keras.layers.ReLU(name=f"{name}_relu1")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop")(y)
    y = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation, use_bias=False, name=f"{name}_conv2")(y)
    y = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(y)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding="same", use_bias=False, name=f"{name}_proj")(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    return tf.keras.layers.ReLU(name=f"{name}_out")(
        tf.keras.layers.Add(name=f"{name}_add")([shortcut, y])
    )


def build_tcn(config: BaselineConfig, input_shape: tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="pose_sequence")
    x = inputs
    for idx, (channels, dilation) in enumerate(zip(config.tcn_channels, config.tcn_dilations), start=1):
        x = residual_tcn_block(x, channels, config.tcn_kernel_size, dilation, config.dropout_rate, f"tcn_block_{idx}")
    avg = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    mx = tf.keras.layers.GlobalMaxPooling1D(name="gmp")(x)
    x = tf.keras.layers.Concatenate(name="pool_concat")([avg, mx])
    x = tf.keras.layers.Dense(config.tcn_channels[-1], activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="head_drop")(x)
    outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs, outputs, name=f"{config.experiment_id}_tcn")


class TemporalAttention(tf.keras.layers.Layer):
    """Additive temporal attention pooling.

    Computes a learned weighted sum over the time axis.
    All internal ops (Dense, Softmax, multiply, reduce_sum) are TFLite INT8 compatible.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.score_dense = tf.keras.layers.Dense(1, use_bias=True)
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, x: Any) -> Any:
        w = self.softmax(self.score_dense(x))  # (B, T, 1)
        return tf.reduce_sum(x * w, axis=1)    # (B, D)

    def get_config(self) -> dict[str, Any]:
        return super().get_config()


def build_gru(config: BaselineConfig, input_shape: tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="pose_sequence")
    x = inputs

    # Optional causal Conv1D pre-processing block for local temporal feature extraction.
    # Causal padding preserves causality (no future leakage) and is fully TFLite INT8 compatible.
    for i in range(config.conv_pre_layers):
        kernel = config.conv_pre_kernel if i == 0 else max(3, config.conv_pre_kernel - 2)
        x = tf.keras.layers.Conv1D(
            config.conv_pre_filters, kernel,
            padding="causal", use_bias=False, name=f"conv_pre_{i + 1}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"conv_pre_bn_{i + 1}")(x)
        x = tf.keras.layers.ReLU(name=f"conv_pre_relu_{i + 1}")(x)
    if config.conv_pre_layers > 0:
        x = tf.keras.layers.Dropout(config.dropout_rate, name="conv_pre_drop")(x)

    for idx, units in enumerate(config.gru_units, start=1):
        return_seq = (idx < len(config.gru_units)) or config.temporal_attention
        gru_cell = tf.keras.layers.GRU(
            units,
            return_sequences=return_seq,
            dropout=config.dropout_rate,
            recurrent_dropout=0.0,
            reset_after=True,
            name=f"gru_{idx}",
        )
        if config.bidirectional:
            x = tf.keras.layers.Bidirectional(gru_cell, name=f"bi_gru_{idx}")(x)
        else:
            x = gru_cell(x)
        if return_seq:
            x = tf.keras.layers.LayerNormalization(name=f"ln_{idx}")(x)
    head_units = config.gru_units[-1] * (2 if config.bidirectional else 1)
    if config.temporal_attention:
        x = TemporalAttention(name="temporal_attn")(x)
    x = tf.keras.layers.Dense(head_units, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="head_drop")(x)
    outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs, outputs, name=f"{config.experiment_id}_gru")


def build_model(config: BaselineConfig, input_shape: tuple[int, int]) -> tf.keras.Model:
    model = build_tcn(config, input_shape) if config.model_type == "tcn" else build_gru(config, input_shape)
    loss: Any = (
        SparseFocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
        if config.focal_loss
        else "sparse_categorical_crossentropy"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def train_model(model: tf.keras.Model, x: dict[str, np.ndarray], y: dict[str, np.ndarray], config: BaselineConfig) -> tf.keras.callbacks.History:
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stop_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    if not config.quiet:
        callbacks.append(
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: log(
                    f"epoch {epoch + 1}/{config.epochs} "
                    + " ".join(f"{k}={v:.4f}" for k, v in (logs or {}).items())
                )
            )
        )
    return model.fit(
        make_tf_dataset(x["train"], y["train"], config.batch_size, True, config.noise_std),
        validation_data=make_tf_dataset(x["val"], y["val"], config.batch_size, False),
        epochs=config.epochs,
        class_weight=class_weight_from_labels(y["train"]),
        callbacks=callbacks,
        verbose=0,
    )


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    config: BaselineConfig,
) -> dict[str, Any]:
    """2D sweep over (threshold × min_consecutive) evaluated at video level."""
    rows = []
    for threshold in np.linspace(0.05, 0.95, config.threshold_count):
        for min_consec in config.min_consecutive_values:
            v_true, _, v_pred = video_level_eval(y_true, y_score, groups, threshold, min_consec)
            cm = confusion_matrix(v_true, v_pred, labels=[0, 1])
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
            fall_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            nfall_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * fall_prec * rec / (fall_prec + rec) if (fall_prec + rec) > 0 else 0.0
            rows.append({
                "threshold": float(threshold),
                "min_consecutive": int(min_consec),
                "precision": fall_prec,
                "nfall_precision": nfall_prec,
                "recall": rec,
                "f1": f1,
                "accuracy": float(accuracy_score(v_true, v_pred)),
            })
    candidates = pd.DataFrame(rows)
    valid = candidates[
        (candidates["recall"] >= config.min_val_recall) &
        (candidates["precision"] >= config.min_val_precision) &
        (candidates["nfall_precision"] >= config.min_val_precision)
    ]
    chosen = (valid if not valid.empty else candidates).sort_values(["f1", "recall"], ascending=False).iloc[0]
    return {
        "threshold": float(chosen["threshold"]),
        "min_consecutive": int(chosen["min_consecutive"]),
        "sweep": rows,
    }


def save_threshold_sweep(threshold_payload: dict[str, Any], output_dir: Path) -> None:
    sweep = pd.DataFrame(threshold_payload["sweep"])
    sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
    # Plot the best row per threshold (max F1 across min_consecutive values)
    best = sweep.loc[sweep.groupby("threshold")["f1"].idxmax()].reset_index(drop=True)
    sel_thresh = float(threshold_payload["threshold"])
    sel_consec = int(threshold_payload.get("min_consecutive", 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in ["precision", "recall", "f1", "accuracy"]:
        ax.plot(best["threshold"], best[col], marker="o", linewidth=1.4, label=col)
    ax.axvline(sel_thresh, color="black", linestyle="--", linewidth=1.2, label=f"selected (min_consec={sel_consec})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Validation threshold sweep (video-level, best min_consecutive per threshold)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_sweep.png", dpi=160)
    plt.close(fig)


def metrics_for(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    split: str,
    directions: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    pred = y_pred if y_pred is not None else (y_score >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    result = {
        "split": split,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "positive_support": int((y_true == 1).sum()),
        "negative_support": int((y_true == 0).sum()),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, pred, labels=[0, 1], digits=4, zero_division=0),
    }
    if directions is not None:
        by_direction = {}
        for direction in sorted(set(directions.tolist())):
            mask = directions == direction
            if int((y_true[mask] == 1).sum()) > 0:
                by_direction[direction] = float(recall_score(y_true[mask], pred[mask], zero_division=0))
        result["by_direction_recall"] = by_direction
    return result


def representative_dataset(x_calib: np.ndarray, max_samples: int):
    for idx in range(min(len(x_calib), max_samples)):
        yield [x_calib[idx : idx + 1].astype(np.float32)]


def _make_serving_fn(model: tf.keras.Model, input_shape: tuple[int, ...]) -> Any:
    """Return a concrete function with training=False to avoid CudnnRNNV3 ops."""
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, *input_shape], dtype=tf.float32)])
    def serving_fn(x: tf.Tensor) -> tf.Tensor:
        return model(x, training=False)

    return serving_fn.get_concrete_function()


def export_tflite(model: tf.keras.Model, x_train: np.ndarray, output_dir: Path, config: BaselineConfig) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    if not config.export_tflite:
        return paths

    # Concrete function forces training=False, preventing GPU-only ops (CudnnRNNV3)
    # from being embedded in the exported graph.
    input_shape: tuple[int, ...] = x_train.shape[1:]
    try:
        concrete_fn = _make_serving_fn(model, input_shape)
    except Exception as exc:
        paths["serving_fn_error"] = repr(exc)
        return paths

    try:
        fp32_path = output_dir / "model_fp32.tflite"
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)
        fp32_path.write_bytes(converter.convert())
        paths["model_fp32_tflite"] = str(fp32_path)
    except Exception as exc:
        paths["fp32_export_error"] = repr(exc)

    if config.quantize_int8:
        try:
            int8_path = output_dir / "model_int8.tflite"
            # Rebuild concrete_fn for int8 converter (each converter needs its own reference)
            concrete_fn_q = _make_serving_fn(model, input_shape)
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn_q], model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: representative_dataset(x_train, config.representative_samples)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            int8_path.write_bytes(converter.convert())
            paths["model_int8_tflite"] = str(int8_path)
            paths["model_int8_size_kb"] = round(int8_path.stat().st_size / 1024.0, 3)
        except Exception as exc:
            paths["int8_export_error"] = repr(exc)

    return paths


def quantize_value(value: np.ndarray, quantization: tuple[float, int], dtype: np.dtype) -> np.ndarray:
    scale, zero_point = quantization
    if scale == 0:
        return value.astype(dtype)
    info = np.iinfo(dtype)
    return np.clip(np.round(value / scale + zero_point), info.min, info.max).astype(dtype)


def dequantize_value(value: np.ndarray, quantization: tuple[float, int]) -> np.ndarray:
    scale, zero_point = quantization
    if scale == 0:
        return value.astype(np.float32)
    return (value.astype(np.float32) - zero_point) * scale


def predict_tflite(tflite_path: Path, x_eval: np.ndarray, positive_labels: list[int] | None = None) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    scores = np.zeros(len(x_eval), dtype=np.float32)
    for idx, item in enumerate(x_eval):
        tensor = item[None, :, :].astype(np.float32)
        if np.issubdtype(input_detail["dtype"], np.integer):
            tensor = quantize_value(tensor, input_detail["quantization"], input_detail["dtype"])
        interpreter.set_tensor(input_detail["index"], tensor)
        interpreter.invoke()
        out = interpreter.get_tensor(output_detail["index"])
        if np.issubdtype(output_detail["dtype"], np.integer):
            out = dequantize_value(out, output_detail["quantization"])
        flat = out.reshape(-1)
        pos_cls = positive_labels if positive_labels is not None else [1]
        scores[idx] = float(sum(flat[c] for c in pos_cls if c < len(flat)))
    return scores


def apply_consecutive_rule(binary_pred: np.ndarray, min_consecutive: int) -> np.ndarray:
    if min_consecutive <= 1:
        return binary_pred.astype(np.int32)
    filtered = np.zeros_like(binary_pred, dtype=np.int32)
    start = None
    for idx, value in enumerate(binary_pred):
        if value == 1 and start is None:
            start = idx
        elif value == 0 and start is not None:
            if idx - start >= min_consecutive:
                filtered[start:idx] = 1
            start = None
    if start is not None and len(binary_pred) - start >= min_consecutive:
        filtered[start:] = 1
    return filtered


def video_level_eval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    min_consecutive: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate window-level scores to video level with consecutive rule.

    Returns (v_true, v_score, v_pred):
      v_score  — max-pool per video (continuous, used for AUC)
      v_pred   — binary after threshold + consecutive filtering
    """
    video_ids = np.unique(groups)
    v_true = np.empty(len(video_ids), dtype=np.int32)
    v_score = np.empty(len(video_ids), dtype=np.float32)
    v_pred = np.empty(len(video_ids), dtype=np.int32)
    for i, vid in enumerate(video_ids):
        mask = groups == vid
        v_true[i] = int(y_true[mask].max())
        v_score[i] = float(y_score[mask].max())
        binary_windows = (y_score[mask] >= threshold).astype(np.int32)
        filtered = apply_consecutive_rule(binary_windows, min_consecutive)
        v_pred[i] = int(filtered.max()) if len(filtered) > 0 else 0
    return v_true, v_score, v_pred


def plot_class_metrics_bar(
    metrics_dict: dict[str, Any],
    output_dir: Path,
    *,
    prefix: str = "",
    title_prefix: str = "",
) -> None:
    """Grouped bar chart: Precision / Recall / F1 for Fall and Non-Fall classes."""
    cm = np.array(metrics_dict["confusion_matrix"])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    def safe_div(a: int, b: int) -> float:
        return a / b if b else 0.0

    fall_p = safe_div(tp, tp + fp)
    fall_r = safe_div(tp, tp + fn)
    fall_f1 = safe_div(2 * fall_p * fall_r, fall_p + fall_r)
    nfall_p = safe_div(tn, tn + fn)
    nfall_r = safe_div(tn, tn + fp)
    nfall_f1 = safe_div(2 * nfall_p * nfall_r, nfall_p + nfall_r)

    cats = ["Precision", "Recall", "F1"]
    fall_vals = [fall_p, fall_r, fall_f1]
    nfall_vals = [nfall_p, nfall_r, nfall_f1]

    x = np.arange(len(cats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w / 2, fall_vals, w, label="Fall", color="#d73027", alpha=0.85)
    b2 = ax.bar(x + w / 2, nfall_vals, w, label="Non-Fall", color="#4575b4", alpha=0.85)
    for bar in (*b1, *b2):
        ax.annotate(
            f"{bar.get_height():.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Score")
    ax.set_title(f"{title_prefix} Per-Class Metrics (Fall vs Non-Fall)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0.0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    support_text = f"Fall n={tp + fn:,}  Non-Fall n={tn + fp:,}"
    ax.text(0.98, 0.02, support_text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}class_metrics.png", dpi=160)
    plt.close(fig)


def plot_direction_recall(
    by_direction_recall: dict[str, float],
    output_dir: Path,
    *,
    prefix: str = "",
    title_prefix: str = "",
) -> None:
    """Bar chart of fall recall broken down by fall direction (FY, SY, ...)."""
    if not by_direction_recall:
        return
    dirs = sorted(by_direction_recall)
    recalls = [by_direction_recall[d] for d in dirs]
    colors = ["#fc8d59", "#fee090", "#e0f3f8", "#91bfdb"][: len(dirs)]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(dirs, recalls, color=colors, alpha=0.85, edgecolor="gray", linewidth=0.6)
    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Recall")
    ax.set_title(f"{title_prefix} Fall Recall by Direction")
    ax.set_ylim(0.0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}direction_recall.png", dpi=160)
    plt.close(fig)


def plot_history(history: tf.keras.callbacks.History, path: Path) -> None:
    hist = pd.DataFrame(history.history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for col in ["loss", "val_loss"]:
        if col in hist:
            axes[0].plot(hist[col], label=col)
    for col in ["accuracy", "val_accuracy"]:
        if col in hist:
            axes[1].plot(hist[col], label=col)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xlabel("Epoch")
    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    output_dir: Path,
    *,
    prefix: str = "",
    title_prefix: str = "Float",
    y_pred: np.ndarray | None = None,
) -> None:
    pred = y_pred if y_pred is not None else (y_score >= threshold).astype(np.int32)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        pred,
        labels=[0, 1],
        display_labels=["Normal", "Fall"],
        cmap="Blues",
        colorbar=False,
        values_format="d",
        ax=ax,
    )
    ax.set_title(f"{title_prefix} confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}confusion_matrix.png", dpi=160)
    plt.close(fig)

    if len(np.unique(y_true)) == 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name=title_prefix)
        ax.set_title(f"{title_prefix} ROC curve")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}roc_curve.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name=title_prefix)
        ax.set_title(f"{title_prefix} PR curve")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}pr_curve.png", dpi=160)
        plt.close(fig)


def save_window_distribution(dist: pd.DataFrame, output_dir: Path) -> None:
    dist.to_csv(output_dir / "window_distribution.csv", index=False)

    label_map = {0: "Non-Fall", 1: "Fall"}
    dist = dist.copy()
    dist["class"] = dist["label"].map(label_map)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Window count by split and class (Fall / Non-Fall)
    ct_class = pd.crosstab(dist["split"], dist["class"])
    for col in ["Fall", "Non-Fall"]:
        if col not in ct_class.columns:
            ct_class[col] = 0
    ct_class[["Fall", "Non-Fall"]].plot(
        kind="bar", ax=axes[0], color=["#d73027", "#4575b4"], alpha=0.85
    )
    axes[0].set_title("Windows by Class")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    # Window count by split and direction
    ct_dir = pd.crosstab(dist["split"], dist["direction"])
    ct_dir.plot(kind="bar", ax=axes[1], alpha=0.85)
    axes[1].set_title("Windows by Direction")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=0)

    # Windows-per-video ratio: Fall vs Non-Fall per split (shows the imbalance)
    ratio_rows = []
    for split_name, grp in dist.groupby("split"):
        for cls_name, cls_grp in grp.groupby("class"):
            n_videos = cls_grp["video_id"].nunique()
            n_windows = len(cls_grp)
            if n_videos > 0:
                ratio_rows.append({"split": split_name, "class": cls_name, "windows/video": n_windows / n_videos})
    if ratio_rows:
        ratio_df = pd.DataFrame(ratio_rows).pivot(index="split", columns="class", values="windows/video")
        for col in ["Fall", "Non-Fall"]:
            if col not in ratio_df.columns:
                ratio_df[col] = 0.0
        ratio_df[["Fall", "Non-Fall"]].plot(
            kind="bar", ax=axes[2], color=["#d73027", "#4575b4"], alpha=0.85
        )
        axes[2].set_title("Windows per Video (Fall vs Non-Fall)")
        axes[2].set_ylabel("Avg Windows / Video")
        axes[2].tick_params(axis="x", rotation=0)
    else:
        axes[2].set_visible(False)

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output_dir / "window_distribution.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_deps()
    args = parse_args()
    config = make_config(args)
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve() if args.data_root else None
    tf.keras.utils.set_random_seed(config.seed)

    output_root = resolve_path(project_root, None, config.output_root)
    output_dir = output_root / config.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.resolved.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False))

    x, y, y_eval, groups, directions, feature_cols, min_v, scale, dist = prepare_data(config, project_root, data_root)
    (output_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2, ensure_ascii=False))
    (output_dir / "normalization.json").write_text(
        json.dumps({"method": "minmax", "min": min_v.reshape(-1).tolist(), "scale": scale.reshape(-1).tolist()}, indent=2),
        encoding="utf-8",
    )
    save_window_distribution(dist, output_dir)

    model = build_model(config, (config.target_steps, len(feature_cols)))
    if not config.quiet:
        model.summary(print_fn=lambda line: log(f"model {line}"))
    history = train_model(model, x, y, config)
    model.save(output_dir / "model.keras")
    plot_history(history, output_dir / "training_curve.png")

    # Sum probabilities of all positive classes — works for both LB-2 (class 1) and LB-3 (class 1+2)
    pos_cols = np.array(config.positive_labels)
    raw_preds = {split: model.predict(x[split], batch_size=config.batch_size, verbose=0) for split in ["train", "val", "test"]}
    scores = {split: raw_preds[split][:, pos_cols].sum(axis=1) for split in ["train", "val", "test"]}
    threshold_payload = select_threshold(y_eval["val"], scores["val"], groups["val"], config)
    save_threshold_sweep(threshold_payload, output_dir)
    threshold = float(threshold_payload["threshold"])
    min_consecutive = int(threshold_payload.get("min_consecutive", 1))

    # Window-level metrics — always use binary y_eval for evaluation
    metrics: dict[str, Any] = {
        f"{split}_float": metrics_for(y_eval[split], scores[split], threshold, split, directions[split])
        for split in ["train", "val", "test"]
    }

    # Window-level plots: confusion matrix + class-metrics bar + direction recall
    for split_name, title in [("test", "Float Test"), ("val", "Float Val")]:
        m = metrics[f"{split_name}_float"]
        plot_confusion_curve(y_eval[split_name], scores[split_name], threshold, output_dir,
                             prefix=f"{split_name}_" if split_name != "test" else "",
                             title_prefix=title)
        plot_class_metrics_bar(m, output_dir,
                               prefix=f"{split_name}_" if split_name != "test" else "",
                               title_prefix=title)
        if m.get("by_direction_recall"):
            plot_direction_recall(m["by_direction_recall"], output_dir,
                                  prefix=f"{split_name}_" if split_name != "test" else "",
                                  title_prefix=title)

    # Video-level metrics: threshold + consecutive rule per video, then aggregate
    for split_name in ["val", "test"]:
        v_true, v_score, v_pred = video_level_eval(
            y_eval[split_name], scores[split_name], groups[split_name],
            threshold, min_consecutive,
        )
        v_metrics = metrics_for(v_true, v_score, threshold, f"{split_name}_video", y_pred=v_pred)
        metrics[f"{split_name}_video"] = v_metrics
        log(
            f"video-level {split_name}: videos={len(v_true)} "
            f"f1={v_metrics['f1']:.4f} recall={v_metrics['recall']:.4f} "
            f"precision={v_metrics['precision']:.4f} min_consecutive={min_consecutive}"
        )
        plot_confusion_curve(v_true, v_score, threshold, output_dir,
                             prefix=f"video_{split_name}_",
                             title_prefix=f"Video-Level {split_name.capitalize()}",
                             y_pred=v_pred)
        plot_class_metrics_bar(v_metrics, output_dir,
                               prefix=f"video_{split_name}_",
                               title_prefix=f"Video-Level {split_name.capitalize()}")

    # Print classification reports cleanly
    for split_name in ["val", "test"]:
        report = metrics.get(f"{split_name}_video", {}).get("classification_report", "")
        if report:
            log(f"=== {split_name.upper()} VIDEO-LEVEL REPORT ===")
            print(report, flush=True)

    export_paths = export_tflite(model, x["train"], output_dir, config)
    quant_report: dict[str, Any] = {"export": export_paths, "runtime": {}}
    if "model_int8_tflite" in export_paths:
        eval_count = len(x["test"]) if config.quant_eval_max_windows == 0 else min(len(x["test"]), config.quant_eval_max_windows)
        q_score = predict_tflite(Path(export_paths["model_int8_tflite"]), x["test"][:eval_count], config.positive_labels)
        q_metrics = metrics_for(y_eval["test"][:eval_count], q_score, threshold, "test_int8", directions["test"][:eval_count])
        metrics["test_int8"] = q_metrics
        plot_confusion_curve(
            y_eval["test"][:eval_count],
            q_score,
            threshold,
            output_dir,
            prefix="int8_",
            title_prefix="INT8",
        )
        plot_class_metrics_bar(q_metrics, output_dir, prefix="int8_", title_prefix="INT8")
        quant_report["runtime"]["test_int8"] = q_metrics
        quant_report["runtime"]["delta_f1"] = (
            None if metrics["test_float"]["f1"] is None else float(metrics["test_float"]["f1"] - q_metrics["f1"])
        )
    (output_dir / "quantization_report.json").write_text(json.dumps(quant_report, indent=2, ensure_ascii=False))

    payload = {
        "experiment_id": config.experiment_id,
        "model_type": config.model_type,
        "preprocessing": config.preprocessing,
        "feature_set": config.feature_set,
        "data_scope": config.data_scope,
        "threshold_selection": threshold_payload,
        "metrics": metrics,
        "export_paths": export_paths,
        "split_sizes": {
            split: int(len(y[split])) for split in ["train", "val", "test"]
        },
        "video_counts": {
            split: int(len(np.unique(groups[split]))) for split in ["train", "val", "test"]
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log(
        f"done {config.experiment_id} "
        f"val_f1={metrics['val_float']['f1']:.4f} (video={metrics['val_video']['f1']:.4f}) "
        f"test_f1={metrics['test_float']['f1']:.4f} (video={metrics['test_video']['f1']:.4f}) "
        f"threshold={threshold:.3f} min_consecutive={min_consecutive}"
    )


if __name__ == "__main__":
    main()
