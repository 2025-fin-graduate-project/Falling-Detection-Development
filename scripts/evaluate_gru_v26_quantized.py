#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_tcn_v2 import load_source_frame, log, set_seed
from scripts.train_tcn_v25 import build_monitoring_segments, build_sliding_window_dataset, describe_window_split, split_segments


@dataclass
class EvalConfig:
    input_format: str
    csv_path: str
    sqlite_path: str
    sqlite_table: str
    monitor_start_sec: float
    monitor_end_sec: float
    target_steps: int
    label_mode: str
    random_state: int
    train_positive_stride: int
    train_negative_stride: int
    eval_stride: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GRU v2.6 int8 TFLite on the notebook test split.")
    parser.add_argument("--artifact-dir", default="artifacts/gru_v26_final_notebook")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--keras-model", default=None)
    parser.add_argument("--fp32-tflite", default=None)
    parser.add_argument("--int8-tflite", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def load_eval_config(metadata: dict[str, object]) -> EvalConfig:
    raw = metadata["config"]
    csv_path = str(raw["csv_path"])
    sqlite_path = str(raw["sqlite_path"])
    if not Path(csv_path).exists() and (PROJECT_ROOT / "dataset" / "final_dataset.csv").exists():
        csv_path = str(PROJECT_ROOT / "dataset" / "final_dataset.csv")
    if not Path(sqlite_path).exists() and (PROJECT_ROOT / "dataset" / "final_dataset.sqlite").exists():
        sqlite_path = str(PROJECT_ROOT / "dataset" / "final_dataset.sqlite")
    return EvalConfig(
        input_format=str(raw["input_format"]),
        csv_path=csv_path,
        sqlite_path=sqlite_path,
        sqlite_table=str(raw["sqlite_table"]),
        monitor_start_sec=float(raw["monitor_start_sec"]),
        monitor_end_sec=float(raw["monitor_end_sec"]),
        target_steps=int(raw["target_steps"]),
        label_mode=str(raw["label_mode"]),
        random_state=int(raw["random_state"]),
        train_positive_stride=int(raw["train_positive_stride"]),
        train_negative_stride=int(raw["train_negative_stride"]),
        eval_stride=int(raw["eval_stride"]),
    )


def prepare_notebook_test_split(config: EvalConfig, metadata: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    set_seed(config.random_state)
    df, feature_cols = load_source_frame(config)
    segments = build_monitoring_segments(df, feature_cols, config.monitor_start_sec, config.monitor_end_sec)
    split_maps = split_segments(segments, config.random_state)

    x_train, y_train, group_train = build_sliding_window_dataset(
        split_maps["train"],
        config.target_steps,
        config.label_mode,
        training=True,
        positive_stride=config.train_positive_stride,
        negative_stride=config.train_negative_stride,
        eval_stride=config.eval_stride,
    )
    x_test, y_test, group_test = build_sliding_window_dataset(
        split_maps["test"],
        config.target_steps,
        config.label_mode,
        training=False,
        positive_stride=config.train_positive_stride,
        negative_stride=config.train_negative_stride,
        eval_stride=config.eval_stride,
    )

    log(describe_window_split("train", y_train, group_train))
    log(describe_window_split("test", y_test, group_test))

    norm = metadata["normalization"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    if mean.shape[0] != x_test.shape[-1] or std.shape[0] != x_test.shape[-1]:
        raise ValueError(
            f"Normalization shape mismatch: mean={mean.shape} std={std.shape} input_features={x_test.shape[-1]}"
        )

    x_test = ((x_test - mean) / std).astype(np.float32)
    return x_test, (y_test > 0.5).astype(np.int32)


def predict_keras(model_path: Path, x: np.ndarray, batch_size: int) -> np.ndarray:
    model = tf.keras.models.load_model(model_path)
    return model.predict(x, batch_size=batch_size, verbose=1)[:, 1].astype(np.float32)


def predict_tflite(model_path: Path, x: np.ndarray, batch_size: int) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    input_dtype = input_detail["dtype"]

    scores: list[np.ndarray] = []
    total = len(x)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        batch_scores = []
        for sample in x[start:stop]:
            input_sample = sample[None, ...]
            if input_dtype == np.int8:
                if input_scale == 0:
                    raise ValueError(f"Invalid int8 input scale in {model_path}: {input_scale}")
                input_sample = np.clip(np.round(input_sample / input_scale + input_zero_point), -128, 127).astype(np.int8)
            else:
                input_sample = input_sample.astype(input_dtype)

            interpreter.set_tensor(input_detail["index"], input_sample)
            interpreter.invoke()
            output = interpreter.get_tensor(output_detail["index"])
            if output_detail["dtype"] == np.int8:
                output = (output.astype(np.float32) - output_zero_point) * output_scale
            else:
                output = output.astype(np.float32)
            batch_scores.append(output[0, 1])
        scores.append(np.asarray(batch_scores, dtype=np.float32))
        log(f"{model_path.name}: predicted {stop}/{total}")
    return np.concatenate(scores)


def summarize(name: str, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (scores >= threshold).astype(np.int32)
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    normal_precision = tn / max(tn + fn, 1)
    normal_recall = tn / max(tn + fp, 1)
    fall_precision = tp / max(tp + fp, 1)
    fall_recall = tp / max(tp + fn, 1)
    normal_f1 = 2 * normal_precision * normal_recall / max(normal_precision + normal_recall, 1e-12)
    fall_f1 = 2 * fall_precision * fall_recall / max(fall_precision + fall_recall, 1e-12)
    macro_f1 = (normal_f1 + fall_f1) / 2
    report = (
        "              precision    recall  f1-score   support\n\n"
        f"      Normal     {normal_precision:0.4f}    {normal_recall:0.4f}    {normal_f1:0.4f}   {tn + fp:7d}\n"
        f"        Fall     {fall_precision:0.4f}    {fall_recall:0.4f}    {fall_f1:0.4f}   {tp + fn:7d}\n\n"
        f"    accuracy                         {(tp + tn) / max(len(y_true), 1):0.4f}   {len(y_true):7d}\n"
        f"   macro avg     {(normal_precision + fall_precision) / 2:0.4f}    {(normal_recall + fall_recall) / 2:0.4f}    {macro_f1:0.4f}   {len(y_true):7d}\n"
    )
    return {
        "name": name,
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "precision": float(fall_precision),
        "recall": float(fall_recall),
        "macro_f1": float(macro_f1),
        "predicted_positive_rate": float(np.mean(pred)),
        "mean_positive_score": float(np.mean(scores)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "classification_report": report,
    }


def print_summary(summary: dict[str, object]) -> None:
    print("\n" + "=" * 30)
    print(f"{summary['name']} 테스트 성능 결과 (Threshold: {summary['threshold']})")
    print("-" * 30)
    print(f"Accuracy:           {summary['accuracy']:.4f}")
    print(f"Precision:          {summary['precision']:.4f}")
    print(f"Recall:             {summary['recall']:.4f}")
    print(f"Macro F1-score:     {summary['macro_f1']:.4f}")
    print(f"Pred Positive Rate: {summary['predicted_positive_rate']:.4f}")
    print(f"Mean Fall Score:    {summary['mean_positive_score']:.4f}")
    print(f"Confusion Matrix:   {summary['confusion_matrix']}")
    print("=" * 30)
    print("\n[Classification Report]")
    print(summary["classification_report"])


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    metadata_path = Path(args.metadata) if args.metadata else artifact_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    config = load_eval_config(metadata)

    x_test, y_test = prepare_notebook_test_split(config, metadata)
    threshold = float(args.threshold if args.threshold is not None else metadata["config"]["decision_threshold"])

    keras_model = Path(args.keras_model) if args.keras_model else artifact_dir / "gru_v26_final.keras"
    fp32_tflite = Path(args.fp32_tflite) if args.fp32_tflite else artifact_dir / "gru_v26_fp32.tflite"
    int8_tflite = Path(args.int8_tflite) if args.int8_tflite else artifact_dir / "gru_v26_int8.tflite"

    results = {
        "test_samples": int(len(y_test)),
        "positive_support": int((y_test == 1).sum()),
        "negative_support": int((y_test == 0).sum()),
        "threshold": threshold,
        "models": [],
    }

    log(f"test samples={len(y_test)} positives={(y_test == 1).sum()} negatives={(y_test == 0).sum()}")
    for name, predictor, path in [
        ("keras", predict_keras, keras_model),
        ("tflite_fp32", predict_tflite, fp32_tflite),
        ("tflite_int8", predict_tflite, int8_tflite),
    ]:
        log(f"evaluating {name}: {path}")
        scores = predictor(path, x_test, args.batch_size)
        summary = summarize(name, y_test, scores, threshold)
        results["models"].append(summary)
        print_summary(summary)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        log(f"wrote {output_path}")


if __name__ == "__main__":
    main()
