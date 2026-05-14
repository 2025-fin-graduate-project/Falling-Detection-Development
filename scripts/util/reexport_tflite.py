#!/usr/bin/env python3
"""Re-export TFLite (fp32 + int8) for experiments whose TFLite conversion previously failed.

IMPORTANT: Must run with CUDA_VISIBLE_DEVICES="" to avoid CudnnRNNV3 GPU ops
being embedded in the graph (which TFLite cannot convert).

Usage:
    uv run python scripts/util/reexport_tflite.py --output-root results/gru_baseline_phase1 --ids P1-v01 P1-v02 P1-v03
    # or all completed experiments in a root:
    uv run python scripts/util/reexport_tflite.py --output-root results/gru_baseline_phase1
"""
from __future__ import annotations

# Disable GPU BEFORE tensorflow is imported — forces CPU GRU ops (no CudnnRNNV3)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def representative_dataset(x_calib: np.ndarray, max_samples: int):
    for idx in range(min(len(x_calib), max_samples)):
        yield [x_calib[idx : idx + 1].astype(np.float32)]


def reexport(exp_dir: Path, representative_samples: int = 256) -> dict:
    import tensorflow as tf

    keras_path = exp_dir / "model.keras"
    norm_path  = exp_dir / "normalization.json"
    feat_path  = exp_dir / "feature_columns.json"
    metrics_path = exp_dir / "metrics.json"

    if not keras_path.exists():
        return {"status": "skip", "reason": "model.keras not found"}

    print(f"  Loading {keras_path.name} ...", flush=True)
    model = tf.keras.models.load_model(str(keras_path))

    # Build a small representative dataset from normalization stats (unit box)
    norm = json.loads(norm_path.read_text()) if norm_path.exists() else None
    feat = json.loads(feat_path.read_text()) if feat_path.exists() else None
    n_features = len(feat) if feat else model.input_shape[-1]
    target_steps = model.input_shape[1]
    x_calib = np.random.uniform(0.0, 1.0, (representative_samples, target_steps, n_features)).astype(np.float32)

    result: dict = {}

    # Bidirectional GRU models fail TFLite conversion because the backward GRU
    # uses TensorListReserve with dynamic element_shape.
    # Fix: rebuild the model with unroll=True on all GRU cells (eliminates
    # TensorList ops by unrolling the time axis into explicit graph ops),
    # then transfer weights from the original model.
    def _rebuild_with_unroll(src_model: "tf.keras.Model") -> "tf.keras.Model":
        import json as _json
        cfg = _json.loads(src_model.to_json())

        def _patch_rnn(node: dict) -> None:
            """Recursively set unroll=True on all RNN layers."""
            cls = node.get("class_name", "")
            inner = node.get("config", {})
            if cls in ("GRU", "LSTM", "SimpleRNN"):
                inner["unroll"] = True
            # Bidirectional stores forward as "layer"; backward is auto-cloned
            # but we need to patch "layer" so the clone also gets unroll=True
            if cls == "Bidirectional":
                _patch_rnn(inner.get("layer", {}))
                _patch_rnn(inner.get("backward_layer", {}))
            for child in inner.get("layers", []):
                _patch_rnn(child)

        _patch_rnn(cfg)
        rebuilt = tf.keras.models.model_from_json(_json.dumps(cfg))
        rebuilt.build((1, target_steps, n_features))
        rebuilt.set_weights(src_model.get_weights())
        return rebuilt

    unrolled = _rebuild_with_unroll(model)

    def _make_converter(m: "tf.keras.Model") -> "tf.lite.TFLiteConverter":
        return tf.lite.TFLiteConverter.from_keras_model(m)

    def _make_converter_int8(m: "tf.keras.Model") -> "tf.lite.TFLiteConverter":
        c = tf.lite.TFLiteConverter.from_keras_model(m)
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = lambda: representative_dataset(x_calib, representative_samples)
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type  = tf.int8
        c.inference_output_type = tf.int8
        return c

    # FP32
    try:
        fp32_path = exp_dir / "model_fp32.tflite"
        fp32_path.write_bytes(_make_converter(unrolled).convert())
        result["model_fp32_tflite"] = str(fp32_path)
        print(f"    fp32 OK ({fp32_path.stat().st_size // 1024} KB)", flush=True)
    except Exception as exc:
        result["fp32_export_error"] = repr(exc)
        print(f"    fp32 FAIL: {exc}", flush=True)

    # INT8
    try:
        int8_path = exp_dir / "model_int8.tflite"
        int8_path.write_bytes(_make_converter_int8(unrolled).convert())
        result["model_int8_tflite"] = str(int8_path)
        result["model_int8_size_kb"] = round(int8_path.stat().st_size / 1024.0, 3)
        print(f"    int8 OK ({int8_path.stat().st_size // 1024} KB)", flush=True)
    except Exception as exc:
        result["int8_export_error"] = repr(exc)
        print(f"    int8 FAIL: {exc}", flush=True)

    # Patch metrics.json export_paths (only when called standalone, not from train_baseline.py)
    if metrics_path.exists() and os.environ.get("REEXPORT_UPDATE_METRICS", "1") == "1":
        payload = json.loads(metrics_path.read_text())
        payload["export_paths"] = result
        metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"    metrics.json updated", flush=True)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ids", nargs="*", help="Experiment IDs to re-export. Defaults to all completed.")
    parser.add_argument("--representative-samples", type=int, default=256)
    args = parser.parse_args()

    root = Path(args.output_root)
    if args.ids:
        candidates = [root / id_ for id_ in args.ids]
    else:
        candidates = sorted(p for p in root.iterdir() if p.is_dir() and (p / "metrics.json").exists())

    for exp_dir in candidates:
        if not (exp_dir / "metrics.json").exists():
            print(f"{exp_dir.name}: no metrics.json, skip")
            continue
        ep = json.loads((exp_dir / "metrics.json").read_text()).get("export_paths", {})
        if "model_int8_tflite" in ep and Path(ep["model_int8_tflite"]).exists():
            print(f"{exp_dir.name}: int8 already exists, skip")
            continue
        print(f"\n[{exp_dir.name}]")
        reexport(exp_dir, args.representative_samples)

    print("\nDone.")


if __name__ == "__main__":
    main()
