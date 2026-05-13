#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

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
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupShuffleSplit
except ModuleNotFoundError as exc:
    plt = np = pd = tf = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


KP_COUNT = 17
ENGINEERED_COLS = ["HSSC_y", "HSSC_x", "RWHC", "VHSSC", "AHSSC", "AHSSC_x"]
DEFAULT_MODELS = ["gru_64_32", "gru_96_48", "gru_128_64", "gru_64_32_light"]
MODEL_SPECS = {
    "gru_64_32": {"units": (64, 32), "head": 32, "dropout_scale": 1.0},
    "gru_96_48": {"units": (96, 48), "head": 48, "dropout_scale": 1.0},
    "gru_128_64": {"units": (128, 64), "head": 64, "dropout_scale": 1.0},
    "gru_64_32_light": {"units": (64, 32), "head": 16, "dropout_scale": 0.5},
}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def ensure_runtime_deps() -> None:
    if IMPORT_ERROR is not None:
        raise SystemExit(
            f"[ERROR] Missing Python dependency: {IMPORT_ERROR.name}. "
            "Install the training dependencies in the active environment first."
        ) from IMPORT_ERROR


def set_seed(seed: int) -> None:
    ensure_runtime_deps()
    np.random.seed(seed)
    tf.random.set_seed(seed)


@dataclass
class FilteredGruSuiteConfig:
    csv_path: str
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
    train_positive_stride: int
    train_negative_stride: int
    eval_stride: int
    min_val_recall: float
    min_consecutive_values: list[int]
    threshold_count: int
    models: list[str]
    max_rows: int | None
    export_stm32: bool
    quant_eval_max_windows: int


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_models(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return DEFAULT_MODELS.copy()
    models = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {DEFAULT_MODELS}")
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3-4 GRU-family models on final_dataset_filtered.csv with unified reports."
    )
    parser.add_argument("--csv-path", default="dataset/final_dataset_filtered.csv")
    parser.add_argument("--output-dir", default="artifacts/filtered_gru_suite")
    parser.add_argument("--monitor-start-sec", type=float, default=0.0)
    parser.add_argument("--monitor-end-sec", type=float, default=10.0)
    parser.add_argument("--target-steps", type=int, default=60)
    parser.add_argument("--label-mode", choices=["segment_max", "last_frame"], default="segment_max")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--train-positive-stride", type=int, default=2)
    parser.add_argument("--train-negative-stride", type=int, default=12)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--min-val-recall", type=float, default=0.86)
    parser.add_argument("--min-consecutive-values", default="1,3,5")
    parser.add_argument("--threshold-count", type=int, default=19)
    parser.add_argument("--models", default="all")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--skip-stm32-export", action="store_true")
    parser.add_argument(
        "--quant-eval-max-windows",
        type=int,
        default=5000,
        help="Max test windows for stateful fp32/int8 TFLite evaluation. Use 0 for all.",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> FilteredGruSuiteConfig:
    if args.train_positive_stride < 1 or args.train_negative_stride < 1 or args.eval_stride < 1:
        raise ValueError("Stride values must be >= 1.")
    if not 0.0 <= args.min_val_recall <= 1.0:
        raise ValueError("--min-val-recall must be between 0 and 1.")
    if args.threshold_count < 3:
        raise ValueError("--threshold-count must be >= 3.")

    return FilteredGruSuiteConfig(
        csv_path=args.csv_path,
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
        train_positive_stride=args.train_positive_stride,
        train_negative_stride=args.train_negative_stride,
        eval_stride=args.eval_stride,
        min_val_recall=args.min_val_recall,
        min_consecutive_values=parse_int_list(args.min_consecutive_values),
        threshold_count=args.threshold_count,
        models=parse_models(args.models),
        max_rows=args.max_rows,
        export_stm32=not args.skip_stm32_export,
        quant_eval_max_windows=args.quant_eval_max_windows,
    )


def get_feature_columns(columns: list[str]) -> list[str]:
    kp_cols = [
        f"kp{idx}_{axis}"
        for idx in range(KP_COUNT)
        for axis in ("y", "x", "s")
        if f"kp{idx}_{axis}" in columns
    ]
    feature_cols = kp_cols + [col for col in ENGINEERED_COLS if col in columns]
    missing_engineered = [col for col in ENGINEERED_COLS if col not in columns]
    if missing_engineered:
        log(f"warning missing engineered feature columns={missing_engineered}")
    if not feature_cols:
        raise ValueError("No feature columns found. Expected kp* plus filtered engineered columns.")
    return feature_cols


def load_filtered_frame(config: FilteredGruSuiteConfig) -> tuple[pd.DataFrame, list[str]]:
    csv_path = Path(config.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Filtered CSV not found: {csv_path}")

    log(f"loading filtered csv from {csv_path}")
    read_kwargs = {"low_memory": False}
    if config.max_rows is not None:
        read_kwargs["nrows"] = config.max_rows
    df = pd.read_csv(csv_path, **read_kwargs)

    required_cols = {"video_id", "frame", "time_sec", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feature_cols = get_feature_columns(df.columns.tolist())
    keep_cols = ["video_id", "frame", "time_sec", "label"] + feature_cols
    df = df[keep_cols].replace([np.inf, -np.inf], np.nan)
    if df[feature_cols].isna().any().any():
        nan_counts = df[feature_cols].isna().sum()
        bad = nan_counts[nan_counts > 0].sort_values(ascending=False).head(10).to_dict()
        log(f"warning filling feature NaN values with train-time neutral 0.0, top columns={bad}")
        df[feature_cols] = df[feature_cols].fillna(0.0)

    log(f"filtered rows loaded={len(df):,} feature_count={len(feature_cols)}")
    return df, feature_cols


def window_label(labels: np.ndarray, mode: str) -> int:
    if mode == "segment_max":
        return int(labels.max())
    if mode == "last_frame":
        return int(labels[-1])
    raise ValueError(f"Unsupported label_mode: {mode}")


def build_sliding_windows(
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
        video_id = str(segment["video_id"])
        if len(values) < target_steps:
            continue

        for start_idx in range(0, len(values) - target_steps + 1):
            label = window_label(frame_labels[start_idx : start_idx + target_steps], label_mode)
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


def normalize_splits(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_val - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(len(x), 100_000), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weight_from_labels(y: np.ndarray) -> dict[int, float]:
    counts = np.bincount(y, minlength=2).astype(np.float32)
    total = counts.sum()
    return {
        0: float(total / (2.0 * max(counts[0], 1.0))),
        1: float(total / (2.0 * max(counts[1], 1.0))),
    }


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


def prepare_dataset(
    config: FilteredGruSuiteConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str], np.ndarray, np.ndarray, dict[str, int]]:
    df, feature_cols = load_filtered_frame(config)
    segments = build_monitoring_segments(
        df=df,
        feature_cols=feature_cols,
        monitor_start_sec=config.monitor_start_sec,
        monitor_end_sec=config.monitor_end_sec,
    )
    split_map = split_segments(segments, config.random_state)

    x_train, y_train, train_groups = build_sliding_windows(
        split_map["train"],
        config.target_steps,
        config.label_mode,
        training=True,
        positive_stride=config.train_positive_stride,
        negative_stride=config.train_negative_stride,
        eval_stride=config.eval_stride,
    )
    x_val, y_val, val_groups = build_sliding_windows(
        split_map["val"],
        config.target_steps,
        config.label_mode,
        training=False,
        positive_stride=config.train_positive_stride,
        negative_stride=config.train_negative_stride,
        eval_stride=config.eval_stride,
    )
    x_test, y_test, test_groups = build_sliding_windows(
        split_map["test"],
        config.target_steps,
        config.label_mode,
        training=False,
        positive_stride=config.train_positive_stride,
        negative_stride=config.train_negative_stride,
        eval_stride=config.eval_stride,
    )

    log(f"train windows={len(y_train):,} positives={int(y_train.sum()):,} negatives={int((y_train == 0).sum()):,}")
    log(f"val windows={len(y_val):,} positives={int(y_val.sum()):,} negatives={int((y_val == 0).sum()):,}")
    log(f"test windows={len(y_test):,} positives={int(y_test.sum()):,} negatives={int((y_test == 0).sum()):,}")

    x_train, x_val, x_test, mean, std = normalize_splits(x_train, x_val, x_test)
    x_splits = {"train": x_train, "val": x_val, "test": x_test}
    y_splits = {"train": y_train, "val": y_val, "test": y_test}
    split_sizes = {
        "train_windows": int(len(y_train)),
        "val_windows": int(len(y_val)),
        "test_windows": int(len(y_test)),
        "train_videos": int(len(np.unique(train_groups))),
        "val_videos": int(len(np.unique(val_groups))),
        "test_videos": int(len(np.unique(test_groups))),
    }
    return x_splits, y_splits, feature_cols, mean, std, split_sizes


def build_stm32_gru_sequence(model_name: str, input_shape: tuple[int, int], dropout_rate: float) -> tf.keras.Model:
    spec = MODEL_SPECS[model_name]
    units1, units2 = spec["units"]
    head_units = spec["head"]
    effective_dropout = dropout_rate * spec["dropout_scale"]

    inputs = tf.keras.Input(shape=input_shape, name="pose_sequence")
    x = tf.keras.layers.GRU(
        units1,
        return_sequences=True,
        dropout=effective_dropout,
        recurrent_dropout=0.0,
        reset_after=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="gru_1",
    )(inputs)
    x = tf.keras.layers.GRU(
        units2,
        return_sequences=False,
        dropout=effective_dropout,
        recurrent_dropout=0.0,
        reset_after=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="gru_2",
    )(x)
    x = tf.keras.layers.Dropout(effective_dropout, name="head_drop")(x)
    x = tf.keras.layers.Dense(head_units, activation="relu", name="head_dense")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs, outputs, name=f"filtered_{model_name}")


def build_model(model_name: str, input_shape: tuple[int, int], dropout_rate: float) -> tf.keras.Model:
    return build_stm32_gru_sequence(model_name, input_shape, dropout_rate)


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_stateful_step_model(sequence_model: tf.keras.Model) -> tf.keras.Model:
    """Expose GRU states as IO ports for the STM32 frame-by-frame runtime."""
    gru_1 = sequence_model.get_layer("gru_1")
    gru_2 = sequence_model.get_layer("gru_2")
    dense = sequence_model.get_layer("head_dense")
    classifier = sequence_model.get_layer("classifier")
    units1 = int(gru_1.units)
    units2 = int(gru_2.units)
    input_dim = int(sequence_model.input_shape[-1])

    h1_in = tf.keras.Input(shape=(units1,), batch_size=1, name="h1_in")
    h2_in = tf.keras.Input(shape=(units2,), batch_size=1, name="h2_in")
    pose_in = tf.keras.Input(shape=(1, input_dim), batch_size=1, name="pose_feature")

    step_gru_1 = tf.keras.layers.GRU(
        units1,
        return_sequences=True,
        return_state=True,
        dropout=0.0,
        recurrent_dropout=0.0,
        reset_after=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="gru_1_step",
    )
    step_gru_2 = tf.keras.layers.GRU(
        units2,
        return_sequences=False,
        return_state=True,
        dropout=0.0,
        recurrent_dropout=0.0,
        reset_after=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="gru_2_step",
    )
    seq_1, h1_out = step_gru_1(pose_in, initial_state=h1_in)
    last_2, h2_out = step_gru_2(seq_1, initial_state=h2_in)
    x = tf.keras.layers.Dense(dense.units, activation="relu", name="head_dense_step")(last_2)
    scores = tf.keras.layers.Dense(2, activation="softmax", name="fall_scores")(x)

    step_model = tf.keras.Model(
        inputs=[h1_in, h2_in, pose_in],
        outputs=[h1_out, scores, h2_out],
        name=f"{sequence_model.name}_stateful_step",
    )
    step_model.get_layer("gru_1_step").set_weights(gru_1.get_weights())
    step_model.get_layer("gru_2_step").set_weights(gru_2.get_weights())
    step_model.get_layer("head_dense_step").set_weights(dense.get_weights())
    step_model.get_layer("fall_scores").set_weights(classifier.get_weights())
    return step_model


def representative_stateful_dataset(x_calib: np.ndarray, units1: int, units2: int, max_samples: int = 256):
    sample_count = min(len(x_calib), max_samples)
    for idx in range(sample_count):
        h1 = np.zeros((1, units1), dtype=np.float32)
        h2 = np.zeros((1, units2), dtype=np.float32)
        frame_idx = min(x_calib.shape[1] - 1, idx % x_calib.shape[1])
        pose = x_calib[idx : idx + 1, frame_idx : frame_idx + 1, :].astype(np.float32)
        yield [h1, h2, pose]


def export_stateful_tflite(
    sequence_model: tf.keras.Model,
    x_calib: np.ndarray,
    model_dir: Path,
    model_name: str,
) -> tuple[tf.keras.Model, dict[str, str]]:
    step_model = build_stateful_step_model(sequence_model)
    step_path = model_dir / f"{model_name}_stateful_step.keras"
    fp32_path = model_dir / f"{model_name}_stateful_fp32.tflite"
    int8_path = model_dir / f"{model_name}_stateful_int8.tflite"
    step_model.save(step_path)

    fp32_converter = tf.lite.TFLiteConverter.from_keras_model(step_model)
    fp32_path.write_bytes(fp32_converter.convert())

    units1 = int(sequence_model.get_layer("gru_1").units)
    units2 = int(sequence_model.get_layer("gru_2").units)
    int8_converter = tf.lite.TFLiteConverter.from_keras_model(step_model)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = lambda: representative_stateful_dataset(x_calib, units1, units2)
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_converter.inference_input_type = tf.int8
    int8_converter.inference_output_type = tf.int8
    int8_path.write_bytes(int8_converter.convert())

    return step_model, {
        "stateful_step_keras": str(step_path),
        "stateful_tflite_fp32": str(fp32_path),
        "stateful_tflite_int8": str(int8_path),
    }


def _quantize_value(value: np.ndarray, quantization: tuple[float, int], dtype: np.dtype) -> np.ndarray:
    scale, zero_point = quantization
    if scale == 0:
        return value.astype(dtype)
    quantized = np.round(value / scale + zero_point)
    info = np.iinfo(dtype)
    return np.clip(quantized, info.min, info.max).astype(dtype)


def _dequantize_value(value: np.ndarray, quantization: tuple[float, int]) -> np.ndarray:
    scale, zero_point = quantization
    if scale == 0:
        return value.astype(np.float32)
    return (value.astype(np.float32) - zero_point) * scale


def _set_tflite_input(interpreter: tf.lite.Interpreter, detail: dict[str, object], value: np.ndarray) -> None:
    dtype = detail["dtype"]
    tensor = value.astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        tensor = _quantize_value(tensor, detail["quantization"], dtype)
    else:
        tensor = tensor.astype(dtype)
    interpreter.set_tensor(detail["index"], tensor)


def _get_tflite_output(interpreter: tf.lite.Interpreter, detail: dict[str, object]) -> np.ndarray:
    tensor = interpreter.get_tensor(detail["index"])
    if np.issubdtype(detail["dtype"], np.integer):
        return _dequantize_value(tensor, detail["quantization"])
    return tensor.astype(np.float32)


def predict_tflite_stateful(tflite_path: Path, x: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inputs = {detail["name"]: detail for detail in input_details}
    outputs = {detail["name"]: detail for detail in output_details}
    h1_in = next(value for key, value in inputs.items() if "h1" in key)
    h2_in = next(value for key, value in inputs.items() if "h2" in key)
    pose_in = next(value for key, value in inputs.items() if "pose" in key)
    score_out = next(value for value in output_details if int(value["shape"][-1]) == 2)
    state_outputs = [value for value in output_details if int(value["shape"][-1]) != 2]
    h1_out = next(value for value in state_outputs if int(value["shape"][-1]) == int(h1_in["shape"][-1]))
    h2_out = next(value for value in state_outputs if int(value["shape"][-1]) == int(h2_in["shape"][-1]))

    scores = np.zeros((len(x), 2), dtype=np.float32)
    for window_idx, window in enumerate(x):
        h1 = np.zeros(h1_in["shape"], dtype=np.float32)
        h2 = np.zeros(h2_in["shape"], dtype=np.float32)
        score = np.array([[1.0, 0.0]], dtype=np.float32)
        for frame_idx in range(window.shape[0]):
            pose = window[frame_idx : frame_idx + 1][None, :, :].astype(np.float32)
            _set_tflite_input(interpreter, h1_in, h1)
            _set_tflite_input(interpreter, h2_in, h2)
            _set_tflite_input(interpreter, pose_in, pose)
            interpreter.invoke()
            h1 = _get_tflite_output(interpreter, h1_out)
            score = _get_tflite_output(interpreter, score_out)
            h2 = _get_tflite_output(interpreter, h2_out)
        scores[window_idx] = score.reshape(-1)[:2]
    return scores[:, 1]


def predict_keras_stateful(step_model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    units1 = int(step_model.input_shape[0][-1])
    units2 = int(step_model.input_shape[1][-1])
    scores = np.zeros((len(x), 2), dtype=np.float32)
    for window_idx, window in enumerate(x):
        h1 = np.zeros((1, units1), dtype=np.float32)
        h2 = np.zeros((1, units2), dtype=np.float32)
        score = np.array([[1.0, 0.0]], dtype=np.float32)
        for frame_idx in range(window.shape[0]):
            pose = window[frame_idx : frame_idx + 1][None, :, :].astype(np.float32)
            h1, score, h2 = step_model.predict([h1, h2, pose], verbose=0)
        scores[window_idx] = score.reshape(-1)[:2]
    return scores[:, 1]


def fit_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    class_weight: dict[int, float],
) -> tf.keras.callbacks.History:
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        ),
    ]
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )


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


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, split: str) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "split": split,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_positive_rate": float(fp / (tn + fp)) if (tn + fp) else 0.0,
        "false_negative_rate": float(fn / (tp + fn)) if (tp + fn) else 0.0,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "positive_support": int((y_true == 1).sum()),
        "negative_support": int((y_true == 0).sum()),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1], digits=4, zero_division=0),
    }


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
    min_consecutive_values: list[int],
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        raw = (y_score >= threshold).astype(np.int32)
        for min_consecutive in min_consecutive_values:
            pred = apply_consecutive_rule(raw, min_consecutive)
            rows.append(
                {
                    "threshold": float(threshold),
                    "min_consecutive": int(min_consecutive),
                    "accuracy": float(accuracy_score(y_true, pred)),
                    "precision": float(precision_score(y_true, pred, zero_division=0)),
                    "recall": float(recall_score(y_true, pred, zero_division=0)),
                    "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
                }
            )
    return pd.DataFrame(rows)


def select_threshold(sweep_df: pd.DataFrame, min_val_recall: float) -> dict[str, float | int]:
    qualified = sweep_df[sweep_df["recall"] >= min_val_recall]
    if qualified.empty:
        selected = sweep_df.sort_values(["macro_f1", "recall"], ascending=False).iloc[0]
        source = "max_macro_f1_no_recall_constraint"
    else:
        selected = qualified.sort_values(["macro_f1", "precision"], ascending=False).iloc[0]
        source = "max_macro_f1_with_recall_constraint"
    return {
        "source": source,
        "threshold": float(selected["threshold"]),
        "min_consecutive": int(selected["min_consecutive"]),
        "val_accuracy": float(selected["accuracy"]),
        "val_precision": float(selected["precision"]),
        "val_recall": float(selected["recall"]),
        "val_macro_f1": float(selected["macro_f1"]),
    }


def evaluate_with_selection(
    y_true: np.ndarray,
    y_score: np.ndarray,
    split: str,
    threshold: float,
    min_consecutive: int,
) -> dict[str, object]:
    pred = apply_consecutive_rule((y_score >= threshold).astype(np.int32), min_consecutive)
    return classification_metrics(y_true, pred, y_score, split)


def plot_history(history: tf.keras.callbacks.History, output_path: Path, title: str) -> None:
    hist = pd.DataFrame(history.history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for col in ["loss", "val_loss"]:
        if col in hist:
            axes[0].plot(hist[col], label=col)
    axes[0].set_title(f"{title} loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for col in ["accuracy", "val_accuracy"]:
        if col in hist:
            axes[1].plot(hist[col], label=col)
    axes[1].set_title(f"{title} accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["Normal", "Fall"],
        cmap="Blues",
        values_format="d",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_curves(y_true: np.ndarray, y_score: np.ndarray, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=axes[0], name=title)
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=axes[1], name=title)
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_runtime_comparison(runtime_df: pd.DataFrame, output_path: Path, title: str) -> None:
    metric_cols = ["accuracy", "precision", "recall", "macro_f1"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    runtime_df.set_index("runtime")[metric_cols].plot(kind="bar", ax=ax)
    ax.axhline(0.90, color="crimson", linestyle="--", linewidth=1.2, label="0.90 target")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_model_report(
    model_name: str,
    model_dir: Path,
    config: FilteredGruSuiteConfig,
    feature_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    split_sizes: dict[str, int],
    history: tf.keras.callbacks.History,
    threshold_payload: dict[str, float | int],
    metrics: dict[str, dict[str, object]],
    export_paths: dict[str, str],
    runtime_metrics: dict[str, dict[str, object]],
) -> Path:
    metadata_path = model_dir / "run_metadata.json"
    payload = {
        "model_name": model_name,
        "config": asdict(config),
        "feature_columns": feature_cols,
        "normalization": {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()},
        "split_sizes": split_sizes,
        "threshold_selection": threshold_payload,
        "metrics": metrics,
        "runtime_metrics": runtime_metrics,
        "export_paths": export_paths,
        "history": history.history,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return metadata_path


def train_one_model(
    model_name: str,
    config: FilteredGruSuiteConfig,
    x_splits: dict[str, np.ndarray],
    y_splits: dict[str, np.ndarray],
    feature_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    split_sizes: dict[str, int],
    output_dir: Path,
) -> dict[str, object]:
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    log(f"training model={model_name}")
    model = build_model(model_name, (config.target_steps, len(feature_cols)), config.dropout_rate)
    model = compile_model(model, config.learning_rate)
    model.summary(print_fn=lambda line: log(f"{model_name} {line}"))

    train_ds = make_tf_dataset(x_splits["train"], y_splits["train"], config.batch_size, training=True)
    val_ds = make_tf_dataset(x_splits["val"], y_splits["val"], config.batch_size, training=False)
    class_weight = class_weight_from_labels(y_splits["train"])
    history = fit_model(model, train_ds, val_ds, config.epochs, class_weight)

    keras_path = model_dir / f"{model_name}.keras"
    model.save(keras_path)
    export_paths = {"keras": str(keras_path)}

    scores = {
        "train": model.predict(x_splits["train"], batch_size=config.batch_size, verbose=0)[:, 1],
        "val": model.predict(x_splits["val"], batch_size=config.batch_size, verbose=0)[:, 1],
        "test": model.predict(x_splits["test"], batch_size=config.batch_size, verbose=0)[:, 1],
    }
    thresholds = np.linspace(0.05, 0.95, config.threshold_count, dtype=np.float32)
    sweep_df = threshold_sweep(
        y_splits["val"],
        scores["val"],
        thresholds,
        config.min_consecutive_values,
    )
    sweep_path = model_dir / "threshold_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    threshold_payload = select_threshold(sweep_df, config.min_val_recall)

    threshold = float(threshold_payload["threshold"])
    min_consecutive = int(threshold_payload["min_consecutive"])
    metrics = {
        split: evaluate_with_selection(y_splits[split], scores[split], split, threshold, min_consecutive)
        for split in ["train", "val", "test"]
    }

    test_pred = apply_consecutive_rule((scores["test"] >= threshold).astype(np.int32), min_consecutive)
    plot_history(history, model_dir / "history.png", model_name)
    plot_confusion(y_splits["test"], test_pred, model_dir / "confusion_matrix_test.png", f"{model_name} test")
    plot_curves(y_splits["test"], scores["test"], model_dir / "roc_pr_test.png", model_name)

    runtime_metrics: dict[str, dict[str, object]] = {
        "keras_sequence": metrics["test"],
    }
    if config.export_stm32:
        try:
            step_model, stm32_paths = export_stateful_tflite(model, x_splits["train"], model_dir, model_name)
            export_paths.update(stm32_paths)

            eval_limit = len(x_splits["test"]) if config.quant_eval_max_windows == 0 else config.quant_eval_max_windows
            eval_limit = min(len(x_splits["test"]), eval_limit)
            x_quant = x_splits["test"][:eval_limit]
            y_quant = y_splits["test"][:eval_limit]

            keras_step_scores = predict_keras_stateful(step_model, x_quant)
            fp32_scores = predict_tflite_stateful(Path(stm32_paths["stateful_tflite_fp32"]), x_quant)
            int8_scores = predict_tflite_stateful(Path(stm32_paths["stateful_tflite_int8"]), x_quant)

            runtime_scores = {
                "keras_stateful_step": keras_step_scores,
                "tflite_stateful_fp32": fp32_scores,
                "tflite_stateful_int8": int8_scores,
            }
            runtime_rows = []
            for runtime_name, runtime_score in runtime_scores.items():
                runtime_pred = apply_consecutive_rule(
                    (runtime_score >= threshold).astype(np.int32),
                    min_consecutive,
                )
                runtime_metric = classification_metrics(y_quant, runtime_pred, runtime_score, runtime_name)
                runtime_metrics[runtime_name] = runtime_metric
                runtime_rows.append(
                    {
                        "runtime": runtime_name,
                        "eval_windows": int(eval_limit),
                        "accuracy": runtime_metric["accuracy"],
                        "precision": runtime_metric["precision"],
                        "recall": runtime_metric["recall"],
                        "macro_f1": runtime_metric["macro_f1"],
                        "roc_auc": runtime_metric["roc_auc"],
                        "pr_auc": runtime_metric["pr_auc"],
                    }
                )
            runtime_df = pd.DataFrame(runtime_rows)
            runtime_df.to_csv(model_dir / "stm32_runtime_comparison.csv", index=False)
            plot_runtime_comparison(
                runtime_df,
                model_dir / "stm32_runtime_comparison.png",
                f"{model_name} STM32 export/runtime metrics",
            )
        except Exception as exc:  # Keep the suite usable if a converter/runtime op is unsupported.
            export_paths["stm32_export_error"] = repr(exc)
            log(f"warning STM32 export/eval failed for {model_name}: {exc!r}")

    metadata_path = save_model_report(
        model_name=model_name,
        model_dir=model_dir,
        config=config,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        split_sizes=split_sizes,
        history=history,
        threshold_payload=threshold_payload,
        metrics=metrics,
        export_paths=export_paths,
        runtime_metrics=runtime_metrics,
    )

    test_metrics = metrics["test"]
    log(
        f"{model_name} test accuracy={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} threshold={threshold:.3f} "
        f"min_consecutive={min_consecutive}"
    )

    return {
        "model": model_name,
        "keras_path": str(keras_path),
        "metadata_path": str(metadata_path),
        "stateful_tflite_fp32": export_paths.get("stateful_tflite_fp32"),
        "stateful_tflite_int8": export_paths.get("stateful_tflite_int8"),
        "stm32_export_error": export_paths.get("stm32_export_error"),
        "threshold": threshold,
        "min_consecutive": min_consecutive,
        **{f"test_{key}": value for key, value in test_metrics.items() if isinstance(value, (int, float)) or value is None},
        "val_macro_f1": threshold_payload["val_macro_f1"],
        "val_recall": threshold_payload["val_recall"],
        "val_precision": threshold_payload["val_precision"],
        "int8_macro_f1": runtime_metrics.get("tflite_stateful_int8", {}).get("macro_f1"),
        "int8_accuracy": runtime_metrics.get("tflite_stateful_int8", {}).get("accuracy"),
    }


def save_suite_plots(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    metric_cols = ["test_accuracy", "test_precision", "test_recall", "test_macro_f1"]
    available = [col for col in metric_cols if col in comparison_df.columns]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    comparison_df.set_index("model")[available].plot(kind="bar", ax=ax)
    ax.axhline(0.90, color="crimson", linestyle="--", linewidth=1.2, label="0.90 target")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Filtered GRU suite test metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = make_config(args)
    set_seed(config.random_state)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "suite_config.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False))

    log(
        "filtered gru suite start "
        f"models={config.models} target_steps={config.target_steps} "
        f"strides(pos={config.train_positive_stride}, neg={config.train_negative_stride}, eval={config.eval_stride})"
    )
    x_splits, y_splits, feature_cols, mean, std, split_sizes = prepare_dataset(config)

    rows = []
    for model_name in config.models:
        rows.append(
            train_one_model(
                model_name=model_name,
                config=config,
                x_splits=x_splits,
                y_splits=y_splits,
                feature_cols=feature_cols,
                mean=mean,
                std=std,
                split_sizes=split_sizes,
                output_dir=output_dir,
            )
        )
        tf.keras.backend.clear_session()

    comparison_df = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
    comparison_csv = output_dir / "model_comparison.csv"
    comparison_json = output_dir / "model_comparison.json"
    comparison_df.to_csv(comparison_csv, index=False)
    comparison_json.write_text(comparison_df.to_json(orient="records", indent=2, force_ascii=False))
    save_suite_plots(comparison_df, output_dir)

    best = comparison_df.iloc[0]
    log(f"comparison written to {comparison_csv}")
    log(
        f"best model={best['model']} test_macro_f1={best['test_macro_f1']:.4f} "
        f"test_accuracy={best['test_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
