#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import tensorflow as tf

from scripts.train_tcn_v2 import (
    build_clip_dataset,
    build_split_metrics,
    class_weight_from_labels,
    compile_model,
    export_tflite_artifacts,
    load_source_frame,
    log,
    make_tf_dataset,
    normalize_splits,
    parse_int_list,
    save_run_metadata,
    set_seed,
    split_dataset,
    train_model,
)


@dataclass
class GruTrainConfig:
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
    dropout_rate: float
    hidden_sizes: list[int]
    export_tflite: bool
    decision_threshold: float | None
    min_val_recall: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU v2 from CSV or SQLite.")
    parser.add_argument("--input-format", choices=["csv", "sqlite"], default="csv")
    parser.add_argument("--csv-path", default="dataset/final_dataset.csv")
    parser.add_argument("--sqlite-path", default="dataset/final_dataset.sqlite")
    parser.add_argument("--sqlite-table", default="fall_frames")
    parser.add_argument("--output-dir", default="artifacts/gru_v2")
    parser.add_argument("--monitor-start-sec", type=float, default=5.0)
    parser.add_argument("--monitor-end-sec", type=float, default=9.0)
    parser.add_argument("--target-steps", type=int, default=60)
    parser.add_argument("--label-mode", choices=["segment_max", "last_frame"], default="segment_max")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dropout-rate", type=float, default=0.15)
    parser.add_argument("--hidden-sizes", default="64,32")
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--min-val-recall", type=float, default=0.80)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> GruTrainConfig:
    hidden_sizes = parse_int_list(args.hidden_sizes)
    if args.decision_threshold is not None and not 0.0 <= args.decision_threshold <= 1.0:
        raise ValueError("--decision-threshold must be between 0 and 1.")
    if not 0.0 <= args.min_val_recall <= 1.0:
        raise ValueError("--min-val-recall must be between 0 and 1.")
    return GruTrainConfig(
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
        dropout_rate=args.dropout_rate,
        hidden_sizes=hidden_sizes,
        export_tflite=args.export_tflite,
        decision_threshold=args.decision_threshold,
        min_val_recall=args.min_val_recall,
    )


def build_gru_v2_model(config: GruTrainConfig, input_shape: tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="pose_sequence")
    x = inputs

    for layer_idx, hidden_size in enumerate(config.hidden_sizes, start=1):
        return_sequences = layer_idx < len(config.hidden_sizes)
        x = tf.keras.layers.GRU(
            hidden_size,
            return_sequences=return_sequences,
            dropout=config.dropout_rate,
            recurrent_dropout=0.0,
            name=f"gru_{layer_idx}",
        )(x)

    x = tf.keras.layers.Dense(config.hidden_sizes[-1], activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="head_drop")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="stm32_gru_v2_classifier")


def main() -> None:
    args = parse_args()
    config = make_config(args)
    set_seed(config.random_state)

    log(
        "gru v2 training start "
        f"input_format={config.input_format} target_steps={config.target_steps} "
        f"hidden_sizes={config.hidden_sizes}"
    )

    df, feature_cols = load_source_frame(config)
    x, y, groups = build_clip_dataset(
        df=df,
        feature_cols=feature_cols,
        monitor_start_sec=config.monitor_start_sec,
        monitor_end_sec=config.monitor_end_sec,
        target_steps=config.target_steps,
        label_mode=config.label_mode,
    )
    log(f"clip dataset ready clips={len(x)} positives={int(y.sum())} negatives={int((y == 0).sum())}")

    splits = split_dataset(x, y, groups, config.random_state)
    x_train, x_val, x_test = x[splits["train"]], x[splits["val"]], x[splits["test"]]
    y_train, y_val, y_test = y[splits["train"]], y[splits["val"]], y[splits["test"]]
    x_train, x_val, x_test, mean, std = normalize_splits(x_train, x_val, x_test)

    split_sizes = {name: int(len(indexes)) for name, indexes in splits.items()}
    log(f"split sizes train={split_sizes['train']} val={split_sizes['val']} test={split_sizes['test']}")

    train_ds = make_tf_dataset(x_train, y_train, config.batch_size, training=True)
    val_ds = make_tf_dataset(x_val, y_val, config.batch_size, training=False)
    class_weight = class_weight_from_labels(y_train)
    log(f"class weights={class_weight}")

    model = build_gru_v2_model(config, input_shape=(config.target_steps, len(feature_cols)))
    model = compile_model(model, config.learning_rate)
    model.summary(print_fn=lambda line: log(f"model {line}"))

    history = train_model(model, train_ds, val_ds, config.epochs, class_weight)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keras_path = output_dir / "gru_v2.keras"
    model.save(keras_path)
    log(f"saved keras model to {keras_path}")

    metrics = build_split_metrics(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        decision_threshold=config.decision_threshold,
        min_val_recall=config.min_val_recall,
    )

    export_paths = {"keras": str(keras_path)}
    if config.export_tflite:
        log("tflite export start")
        export_paths.update(export_tflite_artifacts(model, x_train, output_dir, "gru_v2"))
        log("tflite export done")

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
