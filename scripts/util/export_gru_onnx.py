#!/usr/bin/env python3
"""Stateful GRU → ONNX 변환 + STedgeAI generate.

split_stateful_k37.py가 저장한 gru_stateful.keras를 ONNX로 변환 후
STedgeAI generate (--type onnx) 실행.

uv run python 환경에서 실행 (tf2onnx, onnx 필요).

Usage:
    uv run python scripts/util/export_gru_onnx.py \
        --exp-dir results/top3_retrain/M1-kp13-w60
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx
import onnx

STEDGEAI = Path("/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/stedgeai")
ST_PYTHON = Path("/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python")
PROJ_ROOT = Path(__file__).resolve().parents[2]


def split_and_save_submodels(exp_dir: Path) -> Path:
    """STedgeAI Python으로 split_stateful_k37.py 실행 → submodels/ 생성."""
    split_script = PROJ_ROOT / "scripts" / "util" / "split_stateful_k37.py"
    sub_dir = exp_dir / "submodels"
    gru_path = sub_dir / "gru_stateful.keras"

    if gru_path.exists():
        print(f"  gru_stateful.keras 이미 존재: {gru_path}", flush=True)
        return gru_path

    print(f"  split_stateful_k37.py 실행 중 …", flush=True)
    result = subprocess.run(
        [str(ST_PYTHON), str(split_script), "--exp-dir", str(exp_dir)],
        capture_output=False,
        text=True,
    )
    if not gru_path.exists():
        raise RuntimeError(f"split_stateful_k37.py 실행 후 {gru_path} 없음")
    return gru_path


def keras_to_onnx(keras_path: Path, onnx_path: Path, opset: int = 17) -> None:
    """Keras stateful GRU → ONNX 변환 (tf2onnx)."""
    print(f"\n[ONNX 변환] {keras_path.name} → {onnx_path.name}", flush=True)

    # Keras 모델 로드
    model = tf.keras.models.load_model(str(keras_path))
    model.summary(print_fn=lambda x: print(f"  {x}", flush=True))

    input_sig = [tf.TensorSpec(shape=model.input_shape, dtype=tf.float32, name="conv_features")]

    # tf2onnx 변환
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_sig,
        opset=opset,
        output_path=str(onnx_path),
    )

    # 검증
    onnx.checker.check_model(str(onnx_path))
    size_kb = onnx_path.stat().st_size / 1024
    print(f"  → {onnx_path}  ({size_kb:.1f} KiB, opset={opset})", flush=True)


def stedgeai_generate_onnx(onnx_path: Path, gen_dir: Path, name: str) -> dict:
    """STedgeAI generate --type onnx."""
    gen_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(STEDGEAI), "generate",
        "--target", "stm32n6",
        "--model", str(onnx_path),
        "--type", "onnx",
        "--name", name,
        "--output", str(gen_dir),
        "--workspace", str(gen_dir / "workspace"),
        "--compression", "lossless",
        "--allocate-states",
        "--verbosity", "1",
    ]
    print(f"\n[STedgeAI generate] {' '.join(cmd[-10:])}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    ok = result.returncode == 0
    if ok:
        print("  OK", flush=True)
        for line in result.stdout.splitlines():
            if any(k in line.lower() for k in ("flash", "macc", "weights", "activation")):
                print(f"  {line.strip()}", flush=True)
    else:
        print(f"  FAIL (rc={result.returncode})", flush=True)
        print(result.stderr[-3000:], flush=True)

    return {
        "ok": ok,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:] if not ok else "",
    }


def numerical_verify(keras_path: Path, onnx_path: Path, n_samples: int = 4) -> float:
    """Keras vs ONNX 출력 오차 확인."""
    import onnxruntime as ort

    model = tf.keras.models.load_model(str(keras_path))
    T = model.input_shape[1]
    F = model.input_shape[2]

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    errors = []
    for _ in range(n_samples):
        x = np.random.randn(1, T, F).astype(np.float32)

        # Keras 예측 (reset state)
        for l in model.layers:
            if hasattr(l, "reset_states"):
                l.reset_states()
        keras_out = model.predict(x, verbose=0)

        # ONNX 예측
        onnx_out = sess.run(None, {input_name: x})[0]

        errors.append(float(np.abs(keras_out - onnx_out).max()))

    return float(np.max(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-split", action="store_true",
                    help="gru_stateful.keras가 이미 있으면 split 건너뜀")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    sub_dir = exp_dir / "submodels"
    sub_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    print(f"=== GRU ONNX export: {exp_dir.name} ===", flush=True)

    # 1. stateful GRU submodel 생성 (아직 없으면)
    gru_keras_path = sub_dir / "gru_stateful.keras"
    if not gru_keras_path.exists() and not args.skip_split:
        gru_keras_path = split_and_save_submodels(exp_dir)

    if not gru_keras_path.exists():
        print(f"ERROR: {gru_keras_path} 없음. --skip-split 없이 재실행하거나 split_stateful_k37.py 먼저 실행하세요.")
        sys.exit(1)

    # 2. ONNX 변환
    onnx_path = sub_dir / "gru_stateful.onnx"
    try:
        keras_to_onnx(gru_keras_path, onnx_path, opset=args.opset)
        results["onnx_export"] = {"ok": True, "path": str(onnx_path)}
    except Exception as e:
        print(f"  ONNX 변환 실패: {e}", flush=True)
        results["onnx_export"] = {"ok": False, "error": str(e)}
        sys.exit(1)

    # 3. 수치 검증 (onnxruntime 있을 때만)
    try:
        import onnxruntime
        print("\n[수치 검증]", flush=True)
        max_err = numerical_verify(gru_keras_path, onnx_path)
        print(f"  Keras vs ONNX 최대 오차: {max_err:.2e}", flush=True)
        results["numerical_max_error"] = max_err
        results["numerical_ok"] = max_err < 1e-4
    except ImportError:
        print("  onnxruntime 없음, 수치 검증 스킵", flush=True)
        results["numerical_ok"] = None

    # 4. STedgeAI generate --type onnx
    name = exp_dir.name.replace("-", "_")
    gen_dir = sub_dir / "onnx_gen"
    results["stedgeai_onnx_generate"] = stedgeai_generate_onnx(onnx_path, gen_dir, name)

    # 5. 결과 저장
    result_path = sub_dir / "onnx_export_result.json"
    result_path.write_text(json.dumps(results, indent=2))

    print(f"\n=== 결과 ===", flush=True)
    print(f"  ONNX 변환:        {'OK' if results.get('onnx_export',{}).get('ok') else 'FAIL'}", flush=True)
    print(f"  수치 OK:          {results.get('numerical_ok', '?')}", flush=True)
    print(f"  STedgeAI generate: {'OK' if results.get('stedgeai_onnx_generate',{}).get('ok') else 'FAIL'}", flush=True)
    print(f"  결과 → {result_path}", flush=True)


if __name__ == "__main__":
    main()
