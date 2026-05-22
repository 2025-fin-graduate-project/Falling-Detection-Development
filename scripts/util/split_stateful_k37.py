#!/usr/bin/env python3
"""Conv-GRU 분리 및 Stateful GRU STedgeAI generate PoC.

Conv1D×2 → GRU(stateful=True) 분리 submodel을 만들어 STedgeAI generate 검증.
Keras 3.7 (STedgeAI Python) 에서만 실행.

Usage:
    /home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python \
        scripts/util/split_stateful_k37.py \
        --exp-dir results/phase44_keras37/P44-kp13-w60

출력:
    exp-dir/submodels/conv_submodel.keras
    exp-dir/submodels/gru_stateful.keras
    exp-dir/submodels/split_generate_result.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

STEDGEAI = Path("/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/stedgeai")
PROJ_ROOT = Path(__file__).resolve().parents[2]


def load_model(exp_dir: Path) -> keras.Model:
    """Load the Keras 3.7 trained model (or compat version)."""
    compat = exp_dir / "model_stedgeai_compat.keras"
    orig   = exp_dir / "model.keras"

    if compat.exists():
        print(f"Loading compat model: {compat}", flush=True)
        return keras.models.load_model(str(compat))
    if orig.exists():
        print(f"Loading model: {orig}", flush=True)
        return keras.models.load_model(str(orig))
    raise FileNotFoundError(f"No model.keras found in {exp_dir}")


def split_model(model: keras.Model) -> tuple[keras.Model, list[keras.layers.Layer]]:
    """Split into Conv submodel and list of GRU layers."""
    # Find the boundary: last Conv1D layer → GRU start
    conv_layers = [l for l in model.layers if isinstance(l, keras.layers.Conv1D)]
    gru_layers  = [l for l in model.layers if isinstance(l, keras.layers.GRU)]

    if not conv_layers or not gru_layers:
        raise ValueError(f"Expected Conv1D + GRU layers. Found: {[l.name for l in model.layers]}")

    last_conv = conv_layers[-1]
    # Build conv submodel: input → output of last conv
    conv_model = keras.Model(
        inputs=model.input,
        outputs=last_conv.output,
        name="conv_submodel",
    )
    return conv_model, gru_layers


def build_stateful_gru(gru_layers: list[keras.layers.Layer], conv_out_shape: tuple) -> keras.Model:
    """Build stateful GRU model that processes ONE frame at a time."""
    # conv_out_shape = (T, F) → per-frame shape = (1, F)
    F = conv_out_shape[-1]

    inp = keras.Input(batch_shape=(1, 1, F), name="conv_features")
    x = inp
    for i, gru_src in enumerate(gru_layers):
        return_seq = (i < len(gru_layers) - 1)
        gru_new = keras.layers.GRU(
            gru_src.units,
            return_sequences=return_seq,
            stateful=True,
            unroll=False,  # stateful doesn't need unroll
            reset_after=True,
            name=f"gru_stateful_{i+1}",
        )
        x = gru_new(x)

    # append head layers (Dense, Dropout, output)
    dense_layers = [l for l in gru_src._serialized_attributes if False]  # placeholder

    return keras.Model(inp, x, name="gru_stateful")


def build_stateful_gru_with_head(
    model: keras.Model,
    gru_layers: list[keras.layers.Layer],
    conv_out_shape: tuple,
) -> keras.Model:
    """Build stateful GRU + dense head that processes ONE frame → fall score."""
    F = conv_out_shape[-1]
    inp = keras.Input(batch_shape=(1, 1, F), name="conv_features")
    x = inp

    for i, gru_src in enumerate(gru_layers):
        return_seq = (i < len(gru_layers) - 1)
        gru_new = keras.layers.GRU(
            gru_src.units,
            return_sequences=return_seq,
            stateful=True,
            unroll=False,
            reset_after=True,
            name=f"gru_stateful_{i+1}",
        )(x if i == 0 else x)
        x = gru_new

    # copy dense head layers
    head_layers = [l for l in model.layers
                   if isinstance(l, (keras.layers.Dense, keras.layers.Dropout))
                   and l not in gru_layers]
    for layer in head_layers:
        x = layer(x)

    return keras.Model(inp, x, name="gru_stateful_with_head")


def copy_gru_weights(src_gru: keras.layers.Layer, dst_gru: keras.layers.Layer) -> None:
    """Copy weights from trained (stateless) GRU to new stateful GRU."""
    src_w = src_gru.get_weights()
    dst_gru.set_weights(src_w)


def verify_split_numerically(
    model: keras.Model,
    conv_model: keras.Model,
    gru_stateful_model: keras.Model,
    gru_src_layers: list[keras.layers.Layer],
    n_samples: int = 4,
) -> float:
    """Compare full model output vs split streaming output on random input."""
    T = model.input.shape[1]
    F = model.input.shape[2]

    errors = []
    for _ in range(n_samples):
        x = np.random.randn(1, T, F).astype(np.float32)

        # full model
        full_pred = model.predict(x, verbose=0)  # (1, 2)

        # split: run conv, then feed each frame to stateful GRU
        gru_stateful_model.reset_states()
        conv_out = conv_model.predict(x, verbose=0)  # (1, T, conv_filters)
        for t in range(T):
            frame = conv_out[:, t:t+1, :]  # (1, 1, conv_filters)
            split_pred = gru_stateful_model.predict(frame, verbose=0)

        max_err = float(np.abs(full_pred - split_pred).max())
        errors.append(max_err)

    return float(np.max(errors))


def run_stedgeai_generate(
    model_path: Path,
    out_dir: Path,
    name: str,
    allocate_states: bool = False,
) -> dict:
    """Run STedgeAI generate and return result dict."""
    cmd = [
        str(STEDGEAI), "generate",
        "--target", "stm32n6",
        "--model", str(model_path),
        "--type", "keras",
        "--name", name,
        "--output", str(out_dir),
        "--workspace", str(out_dir / "workspace"),
        "--compression", "lossless",
        "--verbosity", "1",
    ]
    if allocate_states:
        cmd.append("--allocate-states")

    print(f"\n[stedgeai generate] {' '.join(cmd[-8:])}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    ok = result.returncode == 0
    if not ok:
        print(f"  FAILED (rc={result.returncode})", flush=True)
        print(result.stderr[-2000:], flush=True)
    else:
        print(f"  OK", flush=True)
        # try to parse flash/macc from output
        for line in result.stdout.splitlines():
            if "flash" in line.lower() or "macc" in line.lower() or "weights" in line.lower():
                print(f"  {line.strip()}", flush=True)

    return {
        "ok": ok,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:] if not ok else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    sub_dir = exp_dir / "submodels"
    sub_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    print(f"=== Conv-GRU Split PoC: {exp_dir.name} ===", flush=True)
    print(f"Keras version: {keras.__version__}", flush=True)

    # load full model
    model = load_model(exp_dir)
    model.summary()

    # split
    conv_model, gru_layers = split_model(model)
    print(f"\nConv submodel output shape: {conv_model.output.shape}", flush=True)
    print(f"GRU layers: {[f'{l.name}(units={l.units})' for l in gru_layers]}", flush=True)

    # save conv submodel
    conv_path = sub_dir / "conv_submodel.keras"
    conv_model.save(str(conv_path))
    print(f"Saved conv submodel → {conv_path}", flush=True)

    # STedgeAI generate on conv submodel
    results["conv_generate"] = run_stedgeai_generate(
        conv_path, sub_dir / "conv_gen", "conv_submodel"
    )

    # build stateful GRU with head
    conv_out_shape = tuple(conv_model.output.shape[1:])  # (T, F)
    print(f"\nBuilding stateful GRU (conv_out_shape={conv_out_shape}) …", flush=True)

    # Create stateful GRU + head model
    F = conv_out_shape[-1]
    inp = keras.Input(batch_shape=(1, 1, F), name="conv_features")
    x = inp

    gru_new_layers = []
    for i, gru_src in enumerate(gru_layers):
        return_seq = (i < len(gru_layers) - 1)
        gru_new = keras.layers.GRU(
            gru_src.units,
            return_sequences=return_seq,
            stateful=True,
            unroll=False,
            reset_after=True,
            name=f"gru_stateful_{i+1}",
        )
        x = gru_new(x)
        gru_new_layers.append(gru_new)

    # append dense head from original model
    dense_outputs = [l for l in model.layers
                     if isinstance(l, (keras.layers.Dense, keras.layers.Dropout))]
    for layer in dense_outputs:
        x = layer(x)

    gru_stateful_model = keras.Model(inp, x, name="gru_stateful_with_head")
    gru_stateful_model.build((1, 1, F))

    # copy weights from original GRU layers to stateful
    for gru_src, gru_dst in zip(gru_layers, gru_new_layers):
        try:
            copy_gru_weights(gru_src, gru_dst)
            print(f"  Copied weights: {gru_src.name} → {gru_dst.name}", flush=True)
        except Exception as e:
            print(f"  Weight copy failed for {gru_src.name}: {e}", flush=True)

    # GRU-only submodel (no head) for STedgeAI generate
    inp_g = keras.Input(batch_shape=(1, 1, F), name="conv_features")
    x_g = inp_g
    gru_only_layers = []
    for i, gru_src in enumerate(gru_layers):
        return_seq = (i < len(gru_layers) - 1)
        gru_only = keras.layers.GRU(
            gru_src.units,
            return_sequences=return_seq,
            stateful=True,
            unroll=False,
            reset_after=True,
            name=f"gru_stf_{i+1}",
        )
        x_g = gru_only(x_g)
        gru_only_layers.append(gru_only)

    gru_only_model = keras.Model(inp_g, x_g, name="gru_stateful_only")
    gru_only_model.build((1, 1, F))
    for gru_src, gru_dst in zip(gru_layers, gru_only_layers):
        try:
            copy_gru_weights(gru_src, gru_dst)
        except Exception:
            pass

    # save stateful GRU
    gru_path = sub_dir / "gru_stateful.keras"
    gru_only_model.save(str(gru_path))
    print(f"Saved stateful GRU → {gru_path}", flush=True)

    # STedgeAI generate on stateful GRU (with --allocate-states)
    results["gru_stateful_generate"] = run_stedgeai_generate(
        gru_path, sub_dir / "gru_gen", "gru_stateful", allocate_states=True
    )

    # numerical verification
    print("\nNumerical verification …", flush=True)
    try:
        for l in gru_stateful_model.layers:
            if hasattr(l, "reset_states"):
                l.reset_states()
        T = model.input.shape[1]
        Fin = model.input.shape[2]
        x_sample = np.random.randn(1, T, Fin).astype(np.float32)

        full_pred = model.predict(x_sample, verbose=0)

        for l in gru_stateful_model.layers:
            if hasattr(l, "reset_states"):
                l.reset_states()
        conv_out = conv_model.predict(x_sample, verbose=0)
        for t in range(T):
            frame = conv_out[:, t:t+1, :]
            split_pred = gru_stateful_model.predict(frame, verbose=0)

        max_err = float(np.abs(full_pred - split_pred).max())
        print(f"  Max numerical error: {max_err:.2e}", flush=True)
        results["numerical_max_error"] = max_err
        results["numerical_ok"] = max_err < 1e-4
    except Exception as e:
        print(f"  Numerical verification failed: {e}", flush=True)
        results["numerical_ok"] = False
        results["numerical_error"] = str(e)

    # save results
    result_path = sub_dir / "split_generate_result.json"
    result_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {result_path}", flush=True)

    # summary
    print("\n=== Summary ===", flush=True)
    print(f"  Conv generate:          {'OK' if results.get('conv_generate',{}).get('ok') else 'FAIL'}", flush=True)
    print(f"  GRU stateful generate:  {'OK' if results.get('gru_stateful_generate',{}).get('ok') else 'FAIL'}", flush=True)
    print(f"  Numerical OK:           {results.get('numerical_ok', '?')}", flush=True)
    if results.get("numerical_ok"):
        print(f"  → STATEFUL STREAMING PoC PASSED", flush=True)
    else:
        print(f"  → Check logs for failures", flush=True)


if __name__ == "__main__":
    main()
