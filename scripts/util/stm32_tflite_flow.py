import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit


ENGINEERED_COLS = ["HSSC_y", "HSSC_x", "RWHC", "VHSSC"]


@dataclass
class FlowConfig:
    file_path: str
    monitor_start_sec: float = 5.0
    monitor_end_sec: float = 9.0
    target_steps: int = 60
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    random_state: int = 42
    label_mode: str = "segment_max"
    output_dir: str = "artifacts"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    kp_cols = [c for c in df.columns if c.startswith("kp")]
    return kp_cols + ENGINEERED_COLS


def _resample_feature_matrix(values: np.ndarray, target_steps: int) -> np.ndarray:
    if len(values) == target_steps:
        return values.astype(np.float32)

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


def load_clip_dataset(config: FlowConfig):
    df = pd.read_csv(config.file_path)
    df = df.sort_values(["video_id", "time_sec", "frame"]).reset_index(drop=True)
    feature_cols = get_feature_columns(df)

    clips = []
    labels = []
    video_ids = []

    for video_id, group in df.groupby("video_id", sort=False):
        segment = group[
            (group["time_sec"] >= config.monitor_start_sec)
            & (group["time_sec"] < config.monitor_end_sec)
        ]
        if segment.empty:
            continue

        values = segment[feature_cols].to_numpy(dtype=np.float32)
        clip = _resample_feature_matrix(values, config.target_steps)
        label = _segment_label(segment["label"].to_numpy(dtype=np.int32), config.label_mode)

        clips.append(clip)
        labels.append(label)
        video_ids.append(video_id)

    x = np.stack(clips).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)
    groups = np.asarray(video_ids)
    return x, y, groups, feature_cols


def split_dataset(x, y, groups, random_state=42):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
    train_val_idx, test_idx = next(outer.split(x, y, groups))

    inner_groups = groups[train_val_idx]
    inner_y = y[train_val_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=random_state)
    train_idx_rel, val_idx_rel = next(inner.split(x[train_val_idx], inner_y, inner_groups))

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def compute_normalization(x_train: np.ndarray):
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_splits(x_train, x_val, x_test):
    mean, std = compute_normalization(x_train)
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_val - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
        mean,
        std,
    )


def make_tf_dataset(x, y, batch_size, training=False):
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


def residual_tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate, block_name):
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


def build_tcn_model(input_shape, num_classes=2, base_filters=32, kernel_size=3, dropout_rate=0.15):
    inputs = tf.keras.Input(shape=input_shape, name="pose_sequence")
    x = inputs
    for block_idx, dilation_rate in enumerate([1, 2, 4, 8], start=1):
        filters = base_filters if block_idx < 3 else base_filters * 2
        x = residual_tcn_block(
            x=x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout_rate=dropout_rate,
            block_name=f"tcn_block_{block_idx}",
        )
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(base_filters, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_drop")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="stm32_tcn_classifier")
    return model


def build_gru_model(input_shape, num_classes=2, hidden_size=64, dropout_rate=0.15):
    inputs = tf.keras.Input(shape=input_shape, name="pose_sequence")
    x = tf.keras.layers.GRU(
        hidden_size,
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=0.0,
        name="gru_1",
    )(inputs)
    x = tf.keras.layers.GRU(
        hidden_size // 2,
        return_sequences=False,
        dropout=dropout_rate,
        recurrent_dropout=0.0,
        name="gru_2",
    )(x)
    x = tf.keras.layers.Dense(hidden_size // 2, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_drop")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="stm32_gru_classifier")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            SparseBinaryPrecision(name="precision"),
            SparseBinaryRecall(name="recall"),
        ],
    )
    return model


def train_model(model, train_ds, val_ds, epochs, class_weight):
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
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate_model(model, x, y, split_name="test"):
    prob = model.predict(x, verbose=0)
    pred = prob.argmax(axis=1)
    metrics = {
        "split": split_name,
        "accuracy": float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, digits=4, zero_division=0),
    }
    return metrics, prob, pred


def representative_dataset(x: np.ndarray, max_samples: int = 200):
    sample_count = min(len(x), max_samples)
    for idx in range(sample_count):
        yield [x[idx : idx + 1].astype(np.float32)]


def export_tflite_artifacts(model, x_calib, output_dir, artifact_name):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    keras_path = output_path / f"{artifact_name}.keras"
    model.save(keras_path)

    fp32_path = output_path / f"{artifact_name}_fp32.tflite"
    fp32_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp32_path.write_bytes(fp32_converter.convert())

    int8_path = output_path / f"{artifact_name}_int8.tflite"
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


def save_run_metadata(config, feature_cols, mean, std, split_sizes, metrics, export_paths):
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "run_metadata.json"
    payload = {
        "config": asdict(config),
        "feature_columns": feature_cols,
        "normalization": {
            "mean": mean.reshape(-1).tolist(),
            "std": std.reshape(-1).tolist(),
        },
        "split_sizes": split_sizes,
        "metrics": metrics,
        "export_paths": export_paths,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return str(metadata_path)
