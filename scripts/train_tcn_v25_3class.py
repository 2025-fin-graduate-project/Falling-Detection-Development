#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

from scripts.train_tcn_v2 import (
    build_tcn_v2_model,
    load_source_frame,
    log,
    make_tf_dataset,
    normalize_splits,
    save_run_metadata,
    save_visualization_artifacts,
    set_seed,
    parse_int_list,
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
    class_weight_0: float = 1.0
    class_weight_1: float = 2.0
    class_weight_2: float = 5.0
    decision_threshold: float | None = None
    min_val_recall: float = 0.80

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TCN v2.5 with 3-class sliding-window supervision.")
    parser.add_argument("--input-format", choices=["csv", "sqlite"], default="csv")
    parser.add_argument("--csv-path", default="dataset/final_dataset_3class.csv")
    parser.add_argument("--sqlite-path", default="dataset/final_dataset.sqlite")
    parser.add_argument("--sqlite-table", default="fall_frames")
    parser.add_argument("--output-dir", default="artifacts/tcn_v25_3class")
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
    parser.add_argument("--train-negative-stride", type=int, default=1)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument("--class-weight-0", type=float, default=1.0)
    parser.add_argument("--class-weight-1", type=float, default=2.0)
    parser.add_argument("--class-weight-2", type=float, default=5.0)
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--min-val-recall", type=float, default=0.80)
    return parser.parse_args()

def make_config(args: argparse.Namespace) -> TcnV25TrainConfig:
    dilations = parse_int_list(args.dilations)
    channels = parse_int_list(args.channels)
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
        class_weight_0=args.class_weight_0,
        class_weight_1=args.class_weight_1,
        class_weight_2=args.class_weight_2,
        decision_threshold=args.decision_threshold,
        min_val_recall=args.min_val_recall,
    )

def _window_label(labels: np.ndarray, mode: str) -> int:
    if mode == "segment_max":
        return int(labels.max())
    if mode == "last_frame":
        return int(labels[-1])
    raise ValueError(f"Unsupported label_mode: {mode}")

def load_from_csv_3class(config: TcnV25TrainConfig) -> tuple[pd.DataFrame, list[str]]:
    csv_path = Path(config.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    log(f"loading csv source from {csv_path} (3-class)")
    df = pd.read_csv(csv_path)
    from scripts.train_tcn_v2 import get_feature_columns
    feature_cols = get_feature_columns(df.columns.tolist())
    keep_cols = ["video_id", "frame", "time_sec", "label_3class"] + feature_cols
    df = df[keep_cols].rename(columns={"label_3class": "label"})
    log(f"csv rows loaded={len(df)} feature_count={len(feature_cols)}")
    return df, feature_cols

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

    return {
        "train": [segments[idx] for idx in train_val_idx[train_idx_rel]],
        "val": [segments[idx] for idx in train_val_idx[val_idx_rel]],
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
                stride = positive_stride if label > 0 else negative_stride
                if start_idx % stride != 0:
                    continue
            elif start_idx % eval_stride != 0:
                continue

            windows.append(values[start_idx : start_idx + target_steps])
            labels.append(label)
            groups.append(video_id)

    return (
        np.stack(windows).astype(np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(groups),
    )

def describe_window_split(name: str, y: np.ndarray, groups: np.ndarray) -> str:
    c0, c1, c2 = (y == 0).sum(), (y == 1).sum(), (y == 2).sum()
    return f"{name} windows={len(y)} videos={len(np.unique(groups))} 0={c0} 1={c1} 2={c2}"

def multiclass_compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model

def main() -> None:
    args = parse_args()
    config = make_config(args)
    set_seed(config.random_state)

    log(f"tcn v2.5 3-class start steps={config.target_steps} channels={config.channels}")

    df, feature_cols = load_from_csv_3class(config)
    segments = build_monitoring_segments(df, feature_cols, config.monitor_start_sec, config.monitor_end_sec)
    split_map = split_segments(segments, config.random_state)

    x_train, y_train, train_groups = build_sliding_window_dataset(split_map["train"], config.target_steps, config.label_mode, training=True, positive_stride=config.train_positive_stride, negative_stride=config.train_negative_stride, eval_stride=config.eval_stride)
    x_val, y_val, val_groups = build_sliding_window_dataset(split_map["val"], config.target_steps, config.label_mode, training=False, positive_stride=config.train_positive_stride, negative_stride=config.train_negative_stride, eval_stride=config.eval_stride)
    x_test, y_test, test_groups = build_sliding_window_dataset(split_map["test"], config.target_steps, config.label_mode, training=False, positive_stride=config.train_positive_stride, negative_stride=config.train_negative_stride, eval_stride=config.eval_stride)

    log(describe_window_split("train", y_train, train_groups))
    log(describe_window_split("val", y_val, val_groups))
    log(describe_window_split("test", y_test, test_groups))

    x_train, x_val, x_test, mean, std = normalize_splits(x_train, x_val, x_test)

    train_ds = make_tf_dataset(x_train, y_train, config.batch_size, training=True)
    val_ds = make_tf_dataset(x_val, y_val, config.batch_size, training=False)
    
    class_weight = {0: float(config.class_weight_0), 1: float(config.class_weight_1), 2: float(config.class_weight_2)}
    log(f"class weights={class_weight}")

    model = build_tcn_v2_model(config, input_shape=(config.target_steps, len(feature_cols)), num_classes=3)
    model = multiclass_compile_model(model, config.learning_rate)

    history = model.fit(train_ds, validation_data=val_ds, epochs=config.epochs, class_weight=class_weight, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)])

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "tcn_v25_3class.keras")
    
    y_test_pred = np.argmax(model.predict(x_test), axis=-1)
    log(f"test accuracy={accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred))

if __name__ == "__main__":
    main()
