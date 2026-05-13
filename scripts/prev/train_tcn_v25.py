#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from scripts.train_tcn_v2 import (
    build_tcn_v2_model,
    class_weight_from_labels,
    compile_model,
    evaluate_model,
    export_tflite_artifacts,
    load_source_frame,
    log,
    make_tf_dataset,
    normalize_splits,
    save_run_metadata,
    save_visualization_artifacts,
    set_seed,
)


@dataclass
class TcnV25TrainConfig:
    input_format: str
    csv_path: str
    sqlite_path: str
    sqlite_table: str
    output_dir: str
    monitor_start_sec: float
    monitor_end_sec: float
    target_steps: int
    label_mode: str
    batch_size: int
    epochs: int
    learning_rate: float
    random_state: int
    kernel_size: int
    dropout_rate: float
    dilations: list[int]
    channels: list[int]
    export_tflite: bool
    train_positive_stride: int
    train_negative_stride: int
    eval_stride: int


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TCN v2.5 with sliding-window supervision.")
    parser.add_argument("--input-format", choices=["csv", "sqlite"], default="csv")
    parser.add_argument("--csv-path", default="dataset/final_dataset.csv")
    parser.add_argument("--sqlite-path", default="dataset/final_dataset.sqlite")
    parser.add_argument("--sqlite-table", default="fall_frames")
    parser.add_argument("--output-dir", default="artifacts/tcn_v25")
    parser.add_argument("--monitor-start-sec", type=float, default=4.0)
    parser.add_argument("--monitor-end-sec", type=float, default=10.0)
    parser.add_argument("--target-steps", type=int, default=60)
    parser.add_argument("--label-mode", choices=["segment_max", "last_frame"], default="last_frame")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout-rate", type=float, default=0.15)
    parser.add_argument("--dilations", default="1,2,4,8")
    parser.add_argument("--channels", default="32,32,64,96")
    parser.add_argument("--train-positive-stride", type=int, default=1)
    parser.add_argument("--train-negative-stride", type=int, default=5)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--export-tflite", action="store_true")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> TcnV25TrainConfig:
    dilations = parse_int_list(args.dilations)
    channels = parse_int_list(args.channels)
    if len(dilations) != len(channels):
        raise ValueError("--dilations and --channels must have the same length.")
    if args.train_positive_stride < 1 or args.train_negative_stride < 1 or args.eval_stride < 1:
        raise ValueError("Stride values must be >= 1.")

    return TcnV25TrainConfig(
        input_format=args.input_format,
        csv_path=args.csv_path,
        sqlite_path=args.sqlite_path,
        sqlite_table=args.sqlite_table,
        output_dir=args.output_dir,
        monitor_start_sec=args.monitor_start_sec,
        monitor_end_sec=args.monitor_end_sec,
        target_steps=args.target_steps,
        label_mode=args.label_mode,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        kernel_size=args.kernel_size,
        dropout_rate=args.dropout_rate,
        dilations=dilations,
        channels=channels,
        export_tflite=args.export_tflite,
        train_positive_stride=args.train_positive_stride,
        train_negative_stride=args.train_negative_stride,
        eval_stride=args.eval_stride,
    )


def _window_label(labels: np.ndarray, mode: str) -> int:
    if mode == "segment_max":
        return int(labels.max())
    if mode == "last_frame":
        return int(labels[-1])
    raise ValueError(f"Unsupported label_mode: {mode}")


def build_monitoring_segments(
    df: pd.DataFrame,
    feature_cols: list[str],
    monitor_start_sec: float,
    monitor_end_sec: float,
) -> list[dict[str, object]]:
    df = df.sort_values(["video_id", "time_sec", "frame"]).reset_index(drop=True)
    segments: list[dict[str, object]] = []

    for video_id, group in df.groupby("video_id", sort=False):
        segment = group[
            (group["time_sec"] >= monitor_start_sec)
            & (group["time_sec"] < monitor_end_sec)
        ]
        if segment.empty:
            continue

        labels = segment["label"].to_numpy(dtype=np.int32)
        segments.append(
            {
                "video_id": str(video_id),
                "values": segment[feature_cols].to_numpy(dtype=np.float32),
                "labels": labels,
                "segment_label": int(labels.max()),
            }
        )

    if not segments:
        raise ValueError("No monitoring segments were created. Check time window and input source.")
    return segments


def split_segments(segments: list[dict[str, object]], random_state: int) -> dict[str, list[dict[str, object]]]:
    video_ids = np.asarray([segment["video_id"] for segment in segments])
    video_labels = np.asarray([segment["segment_label"] for segment in segments], dtype=np.int32)
    groups = video_ids.copy()
    indexes = np.arange(len(segments))

    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
    train_val_idx, test_idx = next(outer.split(indexes, video_labels, groups))

    inner_groups = groups[train_val_idx]
    inner_labels = video_labels[train_val_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=random_state)
    train_idx_rel, val_idx_rel = next(inner.split(indexes[train_val_idx], inner_labels, inner_groups))

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    return {
        "train": [segments[idx] for idx in train_idx],
        "val": [segments[idx] for idx in val_idx],
        "test": [segments[idx] for idx in test_idx],
    }


def build_sliding_window_dataset(
    segments: list[dict[str, object]],
    target_steps: int,
    label_mode: str,
    *,
    training: bool,
    positive_stride: int,
    negative_stride: int,
    eval_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []

    for segment in segments:
        values = segment["values"]
        frame_labels = segment["labels"]
        video_id = segment["video_id"]

        if len(values) < target_steps:
            continue

        for start_idx in range(0, len(values) - target_steps + 1):
            label = _window_label(frame_labels[start_idx : start_idx + target_steps], label_mode)
            if training:
                stride = positive_stride if label == 1 else negative_stride
                if start_idx % stride != 0:
                    continue
            elif start_idx % eval_stride != 0:
                continue

            windows.append(values[start_idx : start_idx + target_steps])
            labels.append(label)
            groups.append(video_id)

    if not windows:
        raise ValueError("No sliding windows were created. Check target_steps and monitoring range.")

    return (
        np.stack(windows).astype(np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(groups),
    )


def describe_window_split(name: str, y: np.ndarray, groups: np.ndarray) -> str:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    return (
        f"{name} windows={len(y)} videos={len(np.unique(groups))} "
        f"positives={positives} negatives={negatives}"
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)
    set_seed(config.random_state)

    log(
        "tcn v2.5 training start "
        f"input_format={config.input_format} target_steps={config.target_steps} "
        f"dilations={config.dilations} channels={config.channels} "
        f"strides(train_pos={config.train_positive_stride}, train_neg={config.train_negative_stride}, eval={config.eval_stride})"
    )

    df, feature_cols = load_source_frame(config)
    segments = build_monitoring_segments(
        df=df,
        feature_cols=feature_cols,
        monitor_start_sec=config.monitor_start_sec,
        monitor_end_sec=config.monitor_end_sec,
    )
    split_segments_map = split_segments(segments, config.random_state)

    x_train, y_train, train_groups = build_sliding_window_dataset(
        split_segments_map["train"],
        config.target_steps,
        config.label_mode,
        training=True,
        positive_stride=config.train_positive_stride,
        negative_stride=config.train_negative_stride,
        eval_stride=config.eval_stride,
    )
    x_val, y_val, val_groups = build_sliding_window_dataset(
        split_segments_map["val"],
        config.target_steps,
        config.label_mode,
        training=False,
        positive_stride=config.train_positive_stride,
        negative_stride=config.train_negative_stride,
        eval_stride=config.eval_stride,
    )
    x_test, y_test, test_groups = build_sliding_window_dataset(
        split_segments_map["test"],
        config.target_steps,
        config.label_mode,
        training=False,
        positive_stride=config.train_positive_stride,
        negative_stride=config.train_negative_stride,
        eval_stride=config.eval_stride,
    )

    log(describe_window_split("train", y_train, train_groups))
    log(describe_window_split("val", y_val, val_groups))
    log(describe_window_split("test", y_test, test_groups))

    x_train, x_val, x_test, mean, std = normalize_splits(x_train, x_val, x_test)

    split_sizes = {
        "train_windows": int(len(y_train)),
        "val_windows": int(len(y_val)),
        "test_windows": int(len(y_test)),
        "train_videos": int(len(np.unique(train_groups))),
        "val_videos": int(len(np.unique(val_groups))),
        "test_videos": int(len(np.unique(test_groups))),
    }
    log(
        "video split sizes "
        f"train={split_sizes['train_videos']} val={split_sizes['val_videos']} test={split_sizes['test_videos']}"
    )

    train_ds = make_tf_dataset(x_train, y_train, config.batch_size, training=True)
    val_ds = make_tf_dataset(x_val, y_val, config.batch_size, training=False)
    class_weight = class_weight_from_labels(y_train)
    log(f"class weights={class_weight}")

    model = build_tcn_v2_model(config, input_shape=(config.target_steps, len(feature_cols)))
    model = compile_model(model, config.learning_rate)
    model.summary(print_fn=lambda line: log(f"model {line}"))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall",
            mode="max",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keras_path = output_dir / "tcn_v25.keras"
    model.save(keras_path)
    log(f"saved keras model to {keras_path}")

    metrics = {
        "train": evaluate_model(model, x_train, y_train, "train"),
        "val": evaluate_model(model, x_val, y_val, "val"),
        "test": evaluate_model(model, x_test, y_test, "test"),
    }

    export_paths = {"keras": str(keras_path)}
    if config.export_tflite:
        log("tflite export start")
        export_paths.update(export_tflite_artifacts(model, x_train, output_dir, "tcn_v25"))
        log("tflite export done")
    export_paths.update(save_visualization_artifacts(output_dir, history, metrics))

    metadata_path = save_run_metadata(
        config=config,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        split_sizes=split_sizes,
        metrics=metrics,
        history=history,
        export_paths=export_paths,
    )

    for split_name in ["train", "val", "test"]:
        split_metrics = metrics[split_name]
        log(
            f"{split_name} accuracy={split_metrics['accuracy']:.4f} "
            f"precision={split_metrics['precision']:.4f} "
            f"recall={split_metrics['recall']:.4f} "
            f"macro_f1={split_metrics['macro_f1']:.4f}"
        )

    log(f"metadata written to {metadata_path}")
    log(f"config snapshot keys={list(asdict(config).keys())}")


if __name__ == "__main__":
    main()
