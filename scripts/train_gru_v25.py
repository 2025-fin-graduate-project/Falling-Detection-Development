#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.train_gru_v2 import build_gru_v2_model
from scripts.train_tcn_v2 import (
    build_split_metrics,
    class_weight_from_labels,
    compile_model,
    export_tflite_artifacts,
    load_source_frame,
    log,
    make_tf_dataset,
    normalize_splits,
    save_run_metadata,
    save_visualization_artifacts,
    set_seed,
    train_model,
)
from scripts.train_tcn_v25 import (
    build_monitoring_segments,
    build_sliding_window_dataset,
    describe_window_split,
    parse_int_list,
    split_segments,
)


@dataclass
class GruV25TrainConfig:
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
    train_positive_stride: int
    train_negative_stride: int
    eval_stride: int
    decision_threshold: float | None = None
    min_val_recall: float = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU v2.5 with sliding-window supervision.")
    parser.add_argument("--input-format", choices=["csv", "sqlite"], default="csv")
    parser.add_argument("--csv-path", default="dataset/final_dataset.csv")
    parser.add_argument("--sqlite-path", default="dataset/final_dataset.sqlite")
    parser.add_argument("--sqlite-table", default="fall_frames")
    parser.add_argument("--output-dir", default="artifacts/gru_v25")
    parser.add_argument("--monitor-start-sec", type=float, default=4.0)
    parser.add_argument("--monitor-end-sec", type=float, default=10.0)
    parser.add_argument("--target-steps", type=int, default=60)
    parser.add_argument("--label-mode", choices=["segment_max", "last_frame"], default="last_frame")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dropout-rate", type=float, default=0.15)
    parser.add_argument("--hidden-sizes", default="64,32")
    parser.add_argument("--train-positive-stride", type=int, default=1)
    parser.add_argument("--train-negative-stride", type=int, default=5)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--min-val-recall", type=float, default=0.80)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> GruV25TrainConfig:
    hidden_sizes = parse_int_list(args.hidden_sizes)
    if args.train_positive_stride < 1 or args.train_negative_stride < 1 or args.eval_stride < 1:
        raise ValueError("Stride values must be >= 1.")
    if args.decision_threshold is not None and not 0.0 <= args.decision_threshold <= 1.0:
        raise ValueError("--decision-threshold must be between 0 and 1.")
    if not 0.0 <= args.min_val_recall <= 1.0:
        raise ValueError("--min-val-recall must be between 0 and 1.")

    return GruV25TrainConfig(
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
        train_positive_stride=args.train_positive_stride,
        train_negative_stride=args.train_negative_stride,
        eval_stride=args.eval_stride,
        decision_threshold=args.decision_threshold,
        min_val_recall=args.min_val_recall,
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)
    set_seed(config.random_state)

    log(
        "gru v2.5 training start "
        f"input_format={config.input_format} target_steps={config.target_steps} "
        f"hidden_sizes={config.hidden_sizes} "
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
        "train_videos": int(len(set(train_groups.tolist()))),
        "val_videos": int(len(set(val_groups.tolist()))),
        "test_videos": int(len(set(test_groups.tolist()))),
    }
    log(
        "video split sizes "
        f"train={split_sizes['train_videos']} val={split_sizes['val_videos']} test={split_sizes['test_videos']}"
    )

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
    keras_path = output_dir / "gru_v25.keras"
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
        export_paths.update(export_tflite_artifacts(model, x_train, output_dir, "gru_v25"))
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
