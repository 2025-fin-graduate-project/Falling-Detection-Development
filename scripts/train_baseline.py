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
    tcn_channels: list[int] = field(default_factory=lambda: [32, 32, 64, 96])
    tcn_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    tcn_kernel_size: int = 3
    gru_units: list[int] = field(default_factory=lambda: [64, 32])
    export_tflite: bool = True
    quantize_int8: bool = True
    representative_samples: int = 256
    quant_eval_max_windows: int = 5000
    max_rows: int | None = None
    max_windows_per_split: int | None = None


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
    parser.add_argument("--tcn-channels")
    parser.add_argument("--tcn-dilations")
    parser.add_argument("--tcn-kernel-size", type=int)
    parser.add_argument("--gru-units")
    parser.add_argument("--representative-samples", type=int)
    parser.add_argument("--quant-eval-max-windows", type=int)
    parser.add_argument("--export-tflite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--quantize-int8", action=argparse.BooleanOptionalAction)
    parser.add_argument("--smoke", action="store_true", help="Use small limits for a Colab/local smoke run.")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-windows-per-split", type=int)
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
        "tcn_kernel_size": args.tcn_kernel_size,
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
    df["label"] = df["source_label"].isin(config.positive_labels).astype(np.int32)
    if df[cols].isna().any().any():
        df[cols] = df[cols].fillna(0.0)
    return df[["video_id", "frame", "time_sec", "direction", "label"] + cols], cols


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    split_df = df[df["video_id"].isin(split_ids)].sort_values(["video_id", "time_sec", "frame"])
    windows: list[np.ndarray] = []
    labels: list[int] = []
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
        direction = str(segment["direction"].iloc[0])
        for start_idx in range(0, len(segment) - config.target_steps + 1):
            y = window_label(frame_labels[start_idx : start_idx + config.target_steps], config.label_mode)
            stride = config.eval_stride
            if training:
                stride = config.train_positive_stride if y == 1 else config.train_negative_stride
            if start_idx % stride != 0:
                continue
            chunk = values[start_idx : start_idx + config.target_steps]
            windows.append(chunk)
            labels.append(y)
            groups.append(str(video_id))
            directions.append(direction)
            dist_rows.append(
                {
                    "video_id": str(video_id),
                    "direction": direction,
                    "label": y,
                    "fall_ratio": float(frame_labels[start_idx : start_idx + config.target_steps].mean()),
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
        np.asarray(labels, dtype=np.int32),
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
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], list[str], np.ndarray, np.ndarray, pd.DataFrame]:
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
        log(
            f"{split} windows={len(built[split][1]):,} "
            f"positive={int(built[split][1].sum()):,} videos={len(np.unique(built[split][2])):,}"
        )
    x_train, x_val, x_test, min_v, scale = minmax_normalize(built["train"][0], built["val"][0], built["test"][0])
    x = {"train": x_train, "val": x_val, "test": x_test}
    y = {split: built[split][1] for split in ["train", "val", "test"]}
    meta = {split: built[split][3] for split in ["train", "val", "test"]}
    dist = pd.concat(
        [built[split][4].assign(split=split) for split in ["train", "val", "test"]],
        ignore_index=True,
    )
    return x, y, meta, cols, min_v, scale, dist


def class_weight_from_labels(y: np.ndarray) -> dict[int, float]:
    counts = np.bincount(y, minlength=2).astype(np.float32)
    total = counts.sum()
    return {
        0: float(total / (2.0 * max(counts[0], 1.0))),
        1: float(total / (2.0 * max(counts[1], 1.0))),
    }


def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(len(x), 100_000), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


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
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs, outputs, name=f"{config.experiment_id}_tcn")


def build_gru(config: BaselineConfig, input_shape: tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="pose_sequence")
    x = inputs
    for idx, units in enumerate(config.gru_units, start=1):
        x = tf.keras.layers.GRU(
            units,
            return_sequences=(idx < len(config.gru_units)),
            dropout=config.dropout_rate,
            recurrent_dropout=0.0,
            reset_after=True,
            name=f"gru_{idx}",
        )(x)
    x = tf.keras.layers.Dense(config.gru_units[-1], activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="head_drop")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs, outputs, name=f"{config.experiment_id}_gru")


def build_model(config: BaselineConfig, input_shape: tuple[int, int]) -> tf.keras.Model:
    model = build_tcn(config, input_shape) if config.model_type == "tcn" else build_gru(config, input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
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
    return model.fit(
        make_tf_dataset(x["train"], y["train"], config.batch_size, True),
        validation_data=make_tf_dataset(x["val"], y["val"], config.batch_size, False),
        epochs=config.epochs,
        class_weight=class_weight_from_labels(y["train"]),
        callbacks=callbacks,
        verbose=1,
    )


def select_threshold(y_true: np.ndarray, y_score: np.ndarray, config: BaselineConfig) -> dict[str, Any]:
    rows = []
    for threshold in np.linspace(0.05, 0.95, config.threshold_count):
        pred = (y_score >= threshold).astype(np.int32)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, pred)),
            }
        )
    candidates = pd.DataFrame(rows)
    valid = candidates[candidates["recall"] >= config.min_val_recall]
    chosen = (valid if not valid.empty else candidates).sort_values(["f1", "recall"], ascending=False).iloc[0]
    return {"threshold": float(chosen["threshold"]), "sweep": rows}


def save_threshold_sweep(threshold_payload: dict[str, Any], output_dir: Path) -> None:
    sweep = pd.DataFrame(threshold_payload["sweep"])
    sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in ["precision", "recall", "f1", "accuracy"]:
        ax.plot(sweep["threshold"], sweep[col], marker="o", linewidth=1.4, label=col)
    ax.axvline(float(threshold_payload["threshold"]), color="black", linestyle="--", linewidth=1.2, label="selected")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Validation threshold sweep")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_sweep.png", dpi=160)
    plt.close(fig)


def metrics_for(y_true: np.ndarray, y_score: np.ndarray, threshold: float, split: str, directions: np.ndarray | None = None) -> dict[str, Any]:
    pred = (y_score >= threshold).astype(np.int32)
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


def export_tflite(model: tf.keras.Model, x_train: np.ndarray, output_dir: Path, config: BaselineConfig) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    if not config.export_tflite:
        return paths
    try:
        fp32_path = output_dir / "model_fp32.tflite"
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        fp32_path.write_bytes(converter.convert())
        paths["model_fp32_tflite"] = str(fp32_path)
    except Exception as exc:
        paths["fp32_export_error"] = repr(exc)
    if config.quantize_int8:
        try:
            int8_path = output_dir / "model_int8.tflite"
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
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


def predict_tflite(tflite_path: Path, x_eval: np.ndarray) -> np.ndarray:
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
        scores[idx] = out.reshape(-1)[1]
    return scores


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
) -> None:
    pred = (y_score >= threshold).astype(np.int32)
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pd.crosstab(dist["split"], dist["label"]).plot(kind="bar", ax=axes[0])
    axes[0].set_title("Window labels")
    axes[0].set_ylabel("Count")
    pd.crosstab(dist["split"], dist["direction"]).plot(kind="bar", ax=axes[1])
    axes[1].set_title("Directions")
    axes[1].set_ylabel("Count")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
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

    x, y, directions, feature_cols, min_v, scale, dist = prepare_data(config, project_root, data_root)
    (output_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2, ensure_ascii=False))
    (output_dir / "normalization.json").write_text(
        json.dumps({"method": "minmax", "min": min_v.reshape(-1).tolist(), "scale": scale.reshape(-1).tolist()}, indent=2),
        encoding="utf-8",
    )
    save_window_distribution(dist, output_dir)

    model = build_model(config, (config.target_steps, len(feature_cols)))
    model.summary(print_fn=lambda line: log(f"model {line}"))
    history = train_model(model, x, y, config)
    model.save(output_dir / "model.keras")
    plot_history(history, output_dir / "training_curve.png")

    scores = {split: model.predict(x[split], batch_size=config.batch_size, verbose=0)[:, 1] for split in ["train", "val", "test"]}
    threshold_payload = select_threshold(y["val"], scores["val"], config)
    save_threshold_sweep(threshold_payload, output_dir)
    threshold = float(threshold_payload["threshold"])
    metrics = {
        f"{split}_float": metrics_for(y[split], scores[split], threshold, split, directions[split])
        for split in ["train", "val", "test"]
    }
    plot_confusion_curve(y["test"], scores["test"], threshold, output_dir)

    export_paths = export_tflite(model, x["train"], output_dir, config)
    quant_report: dict[str, Any] = {"export": export_paths, "runtime": {}}
    if "model_int8_tflite" in export_paths:
        eval_count = len(x["test"]) if config.quant_eval_max_windows == 0 else min(len(x["test"]), config.quant_eval_max_windows)
        q_score = predict_tflite(Path(export_paths["model_int8_tflite"]), x["test"][:eval_count])
        q_metrics = metrics_for(y["test"][:eval_count], q_score, threshold, "test_int8", directions["test"][:eval_count])
        metrics["test_int8"] = q_metrics
        plot_confusion_curve(
            y["test"][:eval_count],
            q_score,
            threshold,
            output_dir,
            prefix="int8_",
            title_prefix="INT8",
        )
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
        "split_sizes": {split: int(len(y[split])) for split in ["train", "val", "test"]},
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log(
        f"done {config.experiment_id} val_f1={metrics['val_float']['f1']:.4f} "
        f"test_f1={metrics['test_float']['f1']:.4f} threshold={threshold:.3f}"
    )


if __name__ == "__main__":
    main()
