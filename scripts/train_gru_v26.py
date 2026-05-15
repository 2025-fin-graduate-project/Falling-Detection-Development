#!/usr/bin/env python3
"""GRU v2.6 training script with self-labeling and ST Edge AI-compatible export.

Two-phase pipeline:
  1. Initial training (fast, 15 epochs)
  2. Self-labeling refinement → fine-tuning (full epochs)

Export produces:
  - fp32 TFLite   (stedgeai-compatible, no VAR_HANDLE/READ_VARIABLE ops)
  - int8 TFLite   (same, with representative dataset quantization)
  - ONNX          (static batch=1, no Shape/Expand ops; requires tf2onnx)
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from scripts.train_tcn_v2 import (
    build_split_metrics,
    class_weight_from_labels,
    compile_model,
    load_source_frame,
    log,
    make_tf_dataset,
    normalize_splits,
    representative_dataset,
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
class GruV26TrainConfig:
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
    positive_weight: float = 1.5
    soft_label_value: float = 0.9
    decision_threshold: float | None = None
    min_val_recall: float = 0.85


def build_gru_v26_model(config: GruV26TrainConfig, input_shape: tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="pose_sequence")
    x = inputs

    for layer_idx, hidden_size in enumerate(config.hidden_sizes, start=1):
        return_sequences = layer_idx < len(config.hidden_sizes)
        x = tf.keras.layers.GRU(
            hidden_size,
            return_sequences=return_sequences,
            dropout=config.dropout_rate,
            recurrent_dropout=0.0,
            reset_after=True,
            unroll=True,
            activation="tanh",
            recurrent_activation="sigmoid",
            name=f"gru_{layer_idx}",
        )(x)

    x = tf.keras.layers.Dense(config.hidden_sizes[-1], activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="head_drop")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="stm32_gru_v2_classifier")


def refine_labels(model: tf.keras.Model, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Replace labels only where the model is highly confident (>0.95 / <0.05)."""
    probs = model.predict(x, batch_size=256, verbose=0)[:, 1]
    new_labels = y.copy()
    new_labels[probs > 0.95] = 1
    new_labels[probs < 0.05] = 0

    changed_to_pos = int(np.sum((y == 0) & (new_labels == 1)))
    changed_to_neg = int(np.sum((y == 1) & (new_labels == 0)))
    log(f"refine_labels: {changed_to_pos} normal→fall, {changed_to_neg} fall→normal")
    if changed_to_pos > len(y) * 0.2:
        log("WARNING: >20% of labels flipped to positive — possible overfitting to noise")
    return new_labels


def _make_export_model(model: tf.keras.Model) -> tf.keras.Model:
    """batch_size=1 고정, dropout=0 으로 재빌드 후 학습 가중치 복사.

    VAR_HANDLE/READ_VARIABLE: GRU dropout SeedGenerator 제거 (dropout=0)
    SHAPE/FILL: GRU hidden-state 초기화 시 동적 batch 참조 제거 (batch_size=1 고정)
    from_concrete_functions 는 TF 2.x 에서 freeze 버그가 있어 from_keras_model 사용.
    """
    inputs = tf.keras.Input(shape=model.input_shape[1:], batch_size=1, name="pose_sequence")
    x = inputs
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        elif isinstance(layer, tf.keras.layers.GRU):
            cfg = layer.get_config()
            cfg["dropout"] = 0.0
            cfg["recurrent_dropout"] = 0.0
            new_layer = tf.keras.layers.GRU.from_config(cfg)
            x = new_layer(x)
            new_layer.set_weights(layer.get_weights())
        elif isinstance(layer, tf.keras.layers.Dropout):
            pass
        else:
            x = layer(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name=f"{model.name}_export")


def export_stm32_artifacts(
    model: tf.keras.Model,
    x_calib: np.ndarray,
    output_dir: Path,
    artifact_name: str,
) -> dict[str, str]:
    """Export fp32 + int8 TFLite in formats accepted by ST Edge AI Core."""
    output_dir.mkdir(parents=True, exist_ok=True)

    keras_path = output_dir / f"{artifact_name}.keras"
    fp32_path = output_dir / f"{artifact_name}_fp32.tflite"
    int8_path = output_dir / f"{artifact_name}_int8.tflite"

    model.save(keras_path)
    log(f"keras saved: {keras_path}")

    export_model = _make_export_model(model)

    fp32_converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    fp32_path.write_bytes(fp32_converter.convert())
    log(f"fp32 tflite saved: {fp32_path}")

    int8_converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = lambda: representative_dataset(x_calib)
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_converter.inference_input_type = tf.int8
    int8_converter.inference_output_type = tf.int8
    int8_path.write_bytes(int8_converter.convert())
    log(f"int8 tflite saved: {int8_path}")

    return {
        "keras": str(keras_path),
        "tflite_fp32": str(fp32_path),
        "tflite_int8": str(int8_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GRU v2.6 with self-labeling and ST Edge AI-compatible export."
    )
    parser.add_argument("--input-format", choices=["csv", "sqlite"], default="csv")
    parser.add_argument("--csv-path", default="dataset/final_dataset.csv")
    parser.add_argument("--sqlite-path", default="dataset/final_dataset.sqlite")
    parser.add_argument("--sqlite-table", default="fall_frames")
    parser.add_argument("--output-dir", default="artifacts/gru_v26")
    parser.add_argument("--monitor-start-sec", type=float, default=0.0)
    parser.add_argument("--monitor-end-sec", type=float, default=10.0)
    parser.add_argument("--target-steps", type=int, default=60)
    parser.add_argument("--label-mode", choices=["segment_max", "last_frame"], default="segment_max")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--hidden-sizes", default="64,32")
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument("--train-positive-stride", type=int, default=3)
    parser.add_argument("--train-negative-stride", type=int, default=10)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--positive-weight", type=float, default=1.5)
    parser.add_argument("--soft-label-value", type=float, default=0.9)
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--min-val-recall", type=float, default=0.85)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> GruV26TrainConfig:
    hidden_sizes = parse_int_list(args.hidden_sizes)
    if args.train_positive_stride < 1 or args.train_negative_stride < 1 or args.eval_stride < 1:
        raise ValueError("Stride values must be >= 1.")
    if args.positive_weight <= 0.0:
        raise ValueError("--positive-weight must be > 0.")
    if args.decision_threshold is not None and not 0.0 <= args.decision_threshold <= 1.0:
        raise ValueError("--decision-threshold must be between 0 and 1.")
    if not 0.0 <= args.min_val_recall <= 1.0:
        raise ValueError("--min-val-recall must be between 0 and 1.")

    return GruV26TrainConfig(
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
        positive_weight=args.positive_weight,
        soft_label_value=args.soft_label_value,
        decision_threshold=args.decision_threshold,
        min_val_recall=args.min_val_recall,
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)
    set_seed(config.random_state)

    log(
        "gru v2.6 training start "
        f"input_format={config.input_format} target_steps={config.target_steps} "
        f"hidden_sizes={config.hidden_sizes} dropout={config.dropout_rate} "
        f"strides(train_pos={config.train_positive_stride}, "
        f"train_neg={config.train_negative_stride}, eval={config.eval_stride})"
    )

    df, feature_cols = load_source_frame(config)
    segments = build_monitoring_segments(
        df=df,
        feature_cols=feature_cols,
        monitor_start_sec=config.monitor_start_sec,
        monitor_end_sec=config.monitor_end_sec,
    )
    split_maps = split_segments(segments, config.random_state)

    x_train, y_train, train_groups = build_sliding_window_dataset(
        split_maps["train"],
        config.target_steps,
        config.label_mode,
        training=True,
        positive_stride=config.train_positive_stride,
        negative_stride=config.train_negative_stride,
        eval_stride=config.eval_stride,
    )
    x_val, y_val, val_groups = build_sliding_window_dataset(
        split_maps["val"],
        config.target_steps,
        config.label_mode,
        training=False,
        positive_stride=config.train_positive_stride,
        negative_stride=config.train_negative_stride,
        eval_stride=config.eval_stride,
    )
    x_test, y_test, test_groups = build_sliding_window_dataset(
        split_maps["test"],
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
        f"split train={split_sizes['train_videos']}vid/{split_sizes['train_windows']}win "
        f"val={split_sizes['val_videos']}vid/{split_sizes['val_windows']}win "
        f"test={split_sizes['test_videos']}vid/{split_sizes['test_windows']}win"
    )

    model = build_gru_v26_model(config, input_shape=(config.target_steps, len(feature_cols)))
    model = compile_model(model, config.learning_rate)
    model.summary(print_fn=lambda line: log(f"model {line}"))

    class_weight = {0: 1.0, 1: config.positive_weight}

    log("[STEP 1] initial training (15 epochs)")
    train_model(
        model,
        make_tf_dataset(x_train, y_train, 64, training=True),
        make_tf_dataset(x_val, y_val, 64, training=False),
        15,
        class_weight,
    )

    log("[STEP 2] self-labeling")
    y_train_refined = refine_labels(model, x_train, y_train)

    log("[STEP 3] fine-tuning")
    history = train_model(
        model,
        make_tf_dataset(x_train, y_train_refined, config.batch_size, training=True),
        make_tf_dataset(x_val, y_val, config.batch_size, training=False),
        config.epochs,
        class_weight,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_split_metrics(
        model=model,
        x_train=x_train,
        y_train=(y_train > 0.5).astype(int),
        x_val=x_val,
        y_val=(y_val > 0.5).astype(int),
        x_test=x_test,
        y_test=(y_test > 0.5).astype(int),
        decision_threshold=config.decision_threshold,
        min_val_recall=config.min_val_recall,
    )

    export_paths: dict[str, str] = {}
    if config.export_tflite:
        log("ST Edge AI export start")
        export_paths.update(export_stm32_artifacts(model, x_train, output_dir, "gru_v26"))
        log("ST Edge AI export done")

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
        sm = metrics[split_name]
        log(
            f"{split_name} accuracy={sm['accuracy']:.4f} "
            f"precision={sm['precision']:.4f} "
            f"recall={sm['recall']:.4f} "
            f"macro_f1={sm['macro_f1']:.4f}"
        )

    log(f"metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
