#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit


ENGINEERED_COLS = ["HSSC_y", "HSSC_x", "RWHC", "VHSSC"]
DEFAULT_DILATIONS = [1, 2, 4, 8]
DEFAULT_CHANNELS = [32, 32, 64, 96]


@dataclass
class TrainConfig:
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
    decision_threshold: float | None = None
    min_val_recall: float = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deployment-oriented TCN v2 from CSV or SQLite.")
    parser.add_argument("--input-format", choices=["csv", "sqlite"], default="csv")
    parser.add_argument("--csv-path", default="dataset/final_dataset.csv")
    parser.add_argument("--sqlite-path", default="dataset/final_dataset.sqlite")
    parser.add_argument("--sqlite-table", default="fall_frames")
    parser.add_argument("--output-dir", default="artifacts/tcn_v2")
    parser.add_argument("--monitor-start-sec", type=float, default=5.0)
    parser.add_argument("--monitor-end-sec", type=float, default=9.0)
    parser.add_argument("--target-steps", type=int, default=60)
    parser.add_argument("--label-mode", choices=["segment_max", "last_frame"], default="segment_max")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout-rate", type=float, default=0.15)
    parser.add_argument("--dilations", default="1,2,4,8")
    parser.add_argument("--channels", default="32,32,64,96")
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--min-val-recall", type=float, default=0.80)
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def make_config(args: argparse.Namespace) -> TrainConfig:
    dilations = parse_int_list(args.dilations)
    channels = parse_int_list(args.channels)
    if len(dilations) != len(channels):
        raise ValueError("--dilations and --channels must have the same length.")
    if args.decision_threshold is not None and not 0.0 <= args.decision_threshold <= 1.0:
        raise ValueError("--decision-threshold must be between 0 and 1.")
    if not 0.0 <= args.min_val_recall <= 1.0:
        raise ValueError("--min-val-recall must be between 0 and 1.")

    return TrainConfig(
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
        decision_threshold=args.decision_threshold,
        min_val_recall=args.min_val_recall,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_feature_columns(columns: list[str]) -> list[str]:
    kp_cols = [col for col in columns if col.startswith("kp")]
    feature_cols = kp_cols + [col for col in ENGINEERED_COLS if col in columns]
    if not feature_cols:
        raise ValueError("No feature columns found. Expected kp* and engineered feature columns.")
    return feature_cols


def load_source_frame(config: TrainConfig) -> tuple[pd.DataFrame, list[str]]:
    if config.input_format == "csv":
        return load_from_csv(config)
    return load_from_sqlite(config)


def load_from_csv(config: TrainConfig) -> tuple[pd.DataFrame, list[str]]:
    csv_path = Path(config.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    log(f"loading csv source from {csv_path}")
    df = pd.read_csv(csv_path)
    feature_cols = get_feature_columns(df.columns.tolist())
    keep_cols = ["video_id", "frame", "time_sec", "label"] + feature_cols
    df = df[keep_cols]
    log(f"csv rows loaded={len(df)} feature_count={len(feature_cols)}")
    return df, feature_cols


def load_from_sqlite(config: TrainConfig) -> tuple[pd.DataFrame, list[str]]:
    sqlite_path = Path(config.sqlite_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_path}")

    log(f"loading sqlite source from {sqlite_path} table={config.sqlite_table}")
    with sqlite3.connect(sqlite_path) as conn:
        pragma = conn.execute(f"PRAGMA table_info({config.sqlite_table})").fetchall()
        if not pragma:
            raise ValueError(f"Table not found or empty schema: {config.sqlite_table}")

        columns = [row[1] for row in pragma]
        feature_cols = get_feature_columns(columns)
        select_cols = ["video_id", "frame", "time_sec", "label"] + feature_cols
        quoted_cols = ", ".join(f'"{col}"' for col in select_cols)
        query = f"""
            SELECT {quoted_cols}
            FROM "{config.sqlite_table}"
            WHERE time_sec >= ? AND time_sec < ?
            ORDER BY video_id, time_sec, frame
        """
        df = pd.read_sql_query(
            query,
            conn,
            params=[config.monitor_start_sec, config.monitor_end_sec],
        )

    log(f"sqlite rows loaded={len(df)} feature_count={len(feature_cols)}")
    return df, feature_cols


def _resample_feature_matrix(values: np.ndarray, target_steps: int) -> np.ndarray:
    if len(values) == target_steps:
        return values.astype(np.float32)
    if len(values) == 1:
        return np.repeat(values.astype(np.float32), target_steps, axis=0)

    src_x = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_steps, dtype=np.float32)
    out = np.empty((target_steps, values.shape[1]), dtype=np.float32)
    for feat_idx in range(values.shape[1]):
        out[:, feat_idx] = np.interp(dst_x, src_x, values[:, feat_idx])
    return out


def _segment_label(labels: np.ndarray, mode: str) -> int:
    if mode == "segment_max":
        return int(labels.max())
    if mode == "last_frame":
        return int(labels[-1])
    raise ValueError(f"Unsupported label_mode: {mode}")


def build_clip_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    monitor_start_sec: float,
    monitor_end_sec: float,
    target_steps: int,
    label_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = df.sort_values(["video_id", "time_sec", "frame"]).reset_index(drop=True)

    clips: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []

    for video_id, group in df.groupby("video_id", sort=False):
        segment = group[
            (group["time_sec"] >= monitor_start_sec)
            & (group["time_sec"] < monitor_end_sec)
        ]
        if segment.empty:
            continue

        values = segment[feature_cols].to_numpy(dtype=np.float32)
        clip = _resample_feature_matrix(values, target_steps)
        label = _segment_label(segment["label"].to_numpy(dtype=np.int32), label_mode)

        clips.append(clip)
        labels.append(label)
        groups.append(str(video_id))

    if not clips:
        raise ValueError("No clips were created. Check time window and input source.")

    return (
        np.stack(clips).astype(np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(groups),
    )


def split_dataset(x: np.ndarray, y: np.ndarray, groups: np.ndarray, random_state: int) -> dict[str, np.ndarray]:
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
    train_val_idx, test_idx = next(outer.split(x, y, groups))

    inner_groups = groups[train_val_idx]
    inner_y = y[train_val_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=random_state)
    train_idx_rel, val_idx_rel = next(inner.split(x[train_val_idx], inner_y, inner_groups))

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def compute_normalization(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_splits(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean, std = compute_normalization(x_train)
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_val - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
        mean,
        std,
    )


def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, training: bool = False) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(x), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weight_from_labels(y: np.ndarray) -> dict[int, float]:
    counts = np.bincount(y, minlength=2).astype(np.float32)
    total = counts.sum()
    return {
        0: float(total / (2.0 * max(counts[0], 1.0))),
        1: float(total / (2.0 * max(counts[1], 1.0))),
    }


def _safe_metric(metric_fn) -> float | None:
    try:
        return float(metric_fn())
    except ValueError:
        return None


def _safe_curve(curve_fn, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]] | None:
    try:
        curve_values = curve_fn(y_true, y_score)
    except ValueError:
        return None

    if len(curve_values) == 3:
        x_values, y_values, thresholds = curve_values
    else:
        x_values, y_values = curve_values
        thresholds = np.array([], dtype=np.float32)

    return {
        "x": np.asarray(x_values, dtype=np.float32).tolist(),
        "y": np.asarray(y_values, dtype=np.float32).tolist(),
        "thresholds": np.asarray(thresholds, dtype=np.float32).tolist(),
    }


class SparseBinaryPrecision(tf.keras.metrics.Metric):
    def __init__(self, name: str = "precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        positive_true = tf.equal(y_true, 1)
        positive_pred = tf.equal(y_pred, 1)

        tp = tf.reduce_sum(tf.cast(tf.logical_and(positive_true, positive_pred), self.dtype))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(positive_true), positive_pred), self.dtype))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)

    def result(self):
        return tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_positives)

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)


class SparseBinaryRecall(tf.keras.metrics.Metric):
    def __init__(self, name: str = "recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        positive_true = tf.equal(y_true, 1)
        positive_pred = tf.equal(y_pred, 1)

        tp = tf.reduce_sum(tf.cast(tf.logical_and(positive_true, positive_pred), self.dtype))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(positive_true, tf.logical_not(positive_pred)), self.dtype))

        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)

    def result(self):
        return tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_negatives)

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_negatives.assign(0.0)


def residual_tcn_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float,
    block_name: str,
) -> tf.Tensor:
    shortcut = x

    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        use_bias=False,
        name=f"{block_name}_conv1",
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn1")(y)
    y = tf.keras.layers.ReLU(name=f"{block_name}_relu1")(y)
    y = tf.keras.layers.Dropout(dropout_rate, name=f"{block_name}_drop1")(y)

    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        use_bias=False,
        name=f"{block_name}_conv2",
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn2")(y)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=f"{block_name}_proj",
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{block_name}_proj_bn")(shortcut)

    out = tf.keras.layers.Add(name=f"{block_name}_add")([shortcut, y])
    out = tf.keras.layers.ReLU(name=f"{block_name}_relu2")(out)
    return out


def build_tcn_v2_model(config: TrainConfig, input_shape: tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="pose_sequence")
    x = inputs

    for block_idx, (filters, dilation_rate) in enumerate(zip(config.channels, config.dilations), start=1):
        x = residual_tcn_block(
            x=x,
            filters=filters,
            kernel_size=config.kernel_size,
            dilation_rate=dilation_rate,
            dropout_rate=config.dropout_rate,
            block_name=f"tcn_v2_block_{block_idx}",
        )

    avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name="gmp")(x)
    x = tf.keras.layers.Concatenate(name="pool_concat")([avg_pool, max_pool])
    x = tf.keras.layers.Dense(config.channels[-1], activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="head_drop")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="stm32_tcn_v2_classifier")


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            SparseBinaryPrecision(name="precision"),
            SparseBinaryRecall(name="recall"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    class_weight: dict[int, float],
) -> tf.keras.callbacks.History:
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
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


def select_decision_threshold(
    y_true: np.ndarray,
    positive_scores: np.ndarray,
    min_recall: float,
) -> dict[str, float]:
    thresholds = np.unique(np.concatenate([positive_scores.astype(np.float32), np.array([0.5], dtype=np.float32)]))
    best_any: dict[str, float] | None = None
    best_with_floor: dict[str, float] | None = None

    for threshold in thresholds:
        pred = (positive_scores >= threshold).astype(np.int32)
        precision = float(precision_score(y_true, pred, zero_division=0))
        recall = float(recall_score(y_true, pred, zero_division=0))
        macro_f1 = float(f1_score(y_true, pred, average="macro", zero_division=0))
        candidate = {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "macro_f1": macro_f1,
        }
        if best_any is None or candidate["macro_f1"] > best_any["macro_f1"]:
            best_any = candidate
        if recall >= min_recall and (best_with_floor is None or candidate["macro_f1"] > best_with_floor["macro_f1"]):
            best_with_floor = candidate

    selected = best_with_floor or best_any
    if selected is None:
        return {
            "threshold": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "macro_f1": 0.0,
        }
    return selected


def evaluate_model(
    model: tf.keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    split_name: str,
    threshold: float = 0.5,
) -> dict[str, object]:
    prob = model.predict(x, verbose=0)
    positive_scores = prob[:, 1]
    pred = (positive_scores >= threshold).astype(np.int32)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / max(tn + fp, 1))
    false_positive_rate = float(fp / max(fp + tn, 1))
    false_negative_rate = float(fn / max(fn + tp, 1))

    return {
        "split": split_name,
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": _safe_metric(lambda: balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "mcc": _safe_metric(lambda: matthews_corrcoef(y, pred)),
        "roc_auc": _safe_metric(lambda: roc_auc_score(y, positive_scores)),
        "pr_auc": _safe_metric(lambda: average_precision_score(y, positive_scores)),
        "positive_support": int((y == 1).sum()),
        "negative_support": int((y == 0).sum()),
        "predicted_positive_rate": float((pred == 1).mean()),
        "mean_positive_score": float(np.mean(positive_scores)),
        "decision_threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "roc_curve": _safe_curve(roc_curve, y, positive_scores),
        "pr_curve": _safe_curve(precision_recall_curve, y, positive_scores),
        "classification_report": classification_report(y, pred, digits=4, zero_division=0),
    }


def build_split_metrics(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    decision_threshold: float | None,
    min_val_recall: float,
) -> dict[str, dict[str, object]]:
    selected_threshold = decision_threshold
    threshold_source = "cli"
    threshold_payload: dict[str, float] | None = None

    if selected_threshold is None:
        val_prob = model.predict(x_val, verbose=0)
        threshold_payload = select_decision_threshold(y_val, val_prob[:, 1], min_val_recall)
        selected_threshold = threshold_payload["threshold"]
        threshold_source = "validation_search"
        log(
            "selected decision threshold "
            f"threshold={selected_threshold:.4f} "
            f"precision={threshold_payload['precision']:.4f} "
            f"recall={threshold_payload['recall']:.4f} "
            f"macro_f1={threshold_payload['macro_f1']:.4f}"
        )
    else:
        log(f"using fixed decision threshold threshold={selected_threshold:.4f}")

    metrics = {
        "train": evaluate_model(model, x_train, y_train, "train", threshold=selected_threshold),
        "val": evaluate_model(model, x_val, y_val, "val", threshold=selected_threshold),
        "test": evaluate_model(model, x_test, y_test, "test", threshold=selected_threshold),
        "threshold_selection": {
            "source": threshold_source,
            "selected_threshold": float(selected_threshold),
            "min_val_recall": float(min_val_recall),
        },
    }
    if threshold_payload is not None:
        metrics["threshold_selection"]["val_search_metrics"] = threshold_payload
    return metrics


def save_visualization_artifacts(
    output_dir: Path,
    history: tf.keras.callbacks.History,
    metrics: dict[str, dict[str, object]],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    history_path = output_dir / "training_history.png"
    summary_path = output_dir / "evaluation_dashboard.png"

    history_df = pd.DataFrame(history.history)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    if {"loss", "val_loss"}.issubset(history_df.columns):
        history_df[["loss", "val_loss"]].plot(ax=axes[0, 0], marker="o")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].axis("off")

    acc_cols = [col for col in ["accuracy", "val_accuracy"] if col in history_df.columns]
    if acc_cols:
        history_df[acc_cols].plot(ax=axes[0, 1], marker="o")
        axes[0, 1].set_title("Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].axis("off")

    pr_cols = [col for col in ["precision", "val_precision", "recall", "val_recall"] if col in history_df.columns]
    if pr_cols:
        history_df[pr_cols].plot(ax=axes[1, 0], marker="o")
        axes[1, 0].set_title("Precision / Recall")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis("off")

    lr_cols = [col for col in ["learning_rate", "lr"] if col in history_df.columns]
    if lr_cols:
        history_df[lr_cols].plot(ax=axes[1, 1], marker="o", color="darkorange")
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(history_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    summary_rows = []
    for split_name in ["train", "val", "test"]:
        split_metrics = metrics[split_name]
        summary_rows.append(
            {
                "split": split_name,
                "accuracy": split_metrics["accuracy"],
                "precision": split_metrics["precision"],
                "recall": split_metrics["recall"],
                "specificity": split_metrics["specificity"],
                "macro_f1": split_metrics["macro_f1"],
                "roc_auc": split_metrics["roc_auc"] or np.nan,
                "pr_auc": split_metrics["pr_auc"] or np.nan,
            }
        )
    summary_df = pd.DataFrame(summary_rows).set_index("split")
    summary_df.plot(kind="bar", ax=axes[0, 0])
    axes[0, 0].set_title("Split Metrics")
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].grid(True, axis="y", alpha=0.3)
    axes[0, 0].tick_params(axis="x", rotation=0)

    test_cm = np.asarray(metrics["test"]["confusion_matrix"])
    im = axes[0, 1].imshow(test_cm, cmap="Blues")
    axes[0, 1].set_title("Test Confusion Matrix")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Actual")
    axes[0, 1].set_xticks([0, 1], ["Normal", "Fall"])
    axes[0, 1].set_yticks([0, 1], ["Normal", "Fall"])
    for row_idx in range(test_cm.shape[0]):
        for col_idx in range(test_cm.shape[1]):
            axes[0, 1].text(col_idx, row_idx, int(test_cm[row_idx, col_idx]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    for split_name in ["train", "val", "test"]:
        roc_payload = metrics[split_name].get("roc_curve")
        if roc_payload:
            axes[1, 0].plot(roc_payload["x"], roc_payload["y"], label=f"{split_name} (AUC={metrics[split_name]['roc_auc']:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    axes[1, 0].set_title("ROC Curve")
    axes[1, 0].set_xlabel("False Positive Rate")
    axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    for split_name in ["train", "val", "test"]:
        pr_payload = metrics[split_name].get("pr_curve")
        if pr_payload:
            axes[1, 1].plot(pr_payload["x"], pr_payload["y"], label=f"{split_name} (AP={metrics[split_name]['pr_auc']:.3f})")
    axes[1, 1].set_title("Precision-Recall Curve")
    axes[1, 1].set_xlabel("Recall")
    axes[1, 1].set_ylabel("Precision")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "training_history_plot": str(history_path),
        "evaluation_dashboard_plot": str(summary_path),
    }


def representative_dataset(x: np.ndarray, max_samples: int = 200):
    sample_count = min(len(x), max_samples)
    for idx in range(sample_count):
        yield [x[idx : idx + 1].astype(np.float32)]


def export_tflite_artifacts(
    model: tf.keras.Model,
    x_calib: np.ndarray,
    output_dir: Path,
    artifact_name: str,
) -> dict[str, str]:
    keras_path = output_dir / f"{artifact_name}.keras"
    fp32_path = output_dir / f"{artifact_name}_fp32.tflite"
    int8_path = output_dir / f"{artifact_name}_int8.tflite"

    model.save(keras_path)

    fp32_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp32_path.write_bytes(fp32_converter.convert())

    int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = lambda: representative_dataset(x_calib)
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_converter.inference_input_type = tf.int8
    int8_converter.inference_output_type = tf.int8
    int8_path.write_bytes(int8_converter.convert())

    return {
        "keras": str(keras_path),
        "tflite_fp32": str(fp32_path),
        "tflite_int8": str(int8_path),
    }


def save_run_metadata(
    config: TrainConfig,
    feature_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    split_sizes: dict[str, int],
    metrics: dict[str, dict[str, object]],
    history: tf.keras.callbacks.History,
    export_paths: dict[str, str],
) -> Path:
    output_dir = Path(config.output_dir)
    metadata_path = output_dir / "run_metadata.json"
    payload = {
        "config": asdict(config),
        "feature_columns": feature_cols,
        "normalization": {
            "mean": mean.reshape(-1).tolist(),
            "std": std.reshape(-1).tolist(),
        },
        "split_sizes": split_sizes,
        "metrics": metrics,
        "history": history.history,
        "export_paths": export_paths,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return metadata_path


def main() -> None:
    args = parse_args()
    config = make_config(args)
    set_seed(config.random_state)

    log(
        "tcn v2 training start "
        f"input_format={config.input_format} target_steps={config.target_steps} "
        f"dilations={config.dilations} channels={config.channels}"
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

    model = build_tcn_v2_model(config, input_shape=(config.target_steps, len(feature_cols)))
    model = compile_model(model, config.learning_rate)
    model.summary(print_fn=lambda line: log(f"model {line}"))

    history = train_model(model, train_ds, val_ds, config.epochs, class_weight)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keras_path = output_dir / "tcn_v2.keras"
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
        export_paths.update(export_tflite_artifacts(model, x_train, output_dir, "tcn_v2"))
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


if __name__ == "__main__":
    main()
