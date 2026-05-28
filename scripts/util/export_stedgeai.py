#!/usr/bin/env python3
"""STedgeAI 호환 .keras 파일 생성 + analyze/validate 실행.

STedgeAI 4.0 내부 Python (Keras 3.7) 기준으로 작성.
현재 학습 환경의 Keras 3.13+ 모델에서 quantization_config 등 신규 필드를
제거한 뒤 STedgeAI로 analyze/validate를 수행하고 임시 파일을 삭제한다.

Usage (STedgeAI 내부 Python으로 실행):
    /path/to/stedgeai/python scripts/util/export_stedgeai.py \
        --exp-dir results/gru_phase7_quant/Q7-v01 \
        --val-npy  /tmp/val_windows.npy \
        --target   stm32n6
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

STEDGEAI = Path(__file__).resolve().parents[2] / "scripts" / "util" / "_stedgeai_path.txt"


def _get_stedgeai() -> str:
    """Return stedgeai binary path."""
    candidates = [
        "/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/stedgeai",
        "/home/min/app/ST/STEdgeAI/3.0/Utilities/linux/stedgeai",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError("stedgeai binary not found")


def _strip_compat_fields(obj):
    """Keras 3.10+ 신규 필드를 재귀 제거 (Keras 3.7 호환)."""
    if isinstance(obj, dict):
        for key in ("quantization_config",):
            obj.pop(key, None)
        for v in obj.values():
            _strip_compat_fields(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_compat_fields(item)


def make_compat_keras(keras_path: Path, out_path: Path) -> None:
    """quantization_config 등 제거한 STedgeAI 호환 .keras 파일 생성."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        with zipfile.ZipFile(keras_path, "r") as z:
            z.extractall(tmpdir)

        config_file = tmpdir / "config.json"
        if config_file.exists():
            cfg = json.loads(config_file.read_text())
            _strip_compat_fields(cfg)
            config_file.write_text(json.dumps(cfg))

        if out_path.exists():
            out_path.unlink()
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(tmpdir.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(tmpdir))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_analyze(model_path: Path, name: str, target: str,
                work_dir: Path, out_dir: Path) -> dict:
    """stedgeai analyze 실행 → FLASH/RAM 크기 반환."""
    stedgeai = _get_stedgeai()
    cmd = [
        stedgeai, "analyze",
        "--target", target,
        "--model", str(model_path),
        "--type", "keras",
        "--name", name,
        "--output", str(out_dir),
        "--workspace", str(work_dir),
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout + result.stderr

    info: dict = {"analyze_ok": result.returncode == 0, "raw": out}

    # 핵심 수치 파싱
    for line in out.splitlines():
        line = line.strip()
        if "weights (ro)" in line:
            # e.g. "weights (ro) :   549,396 B (536.52 KiB) ..."
            try:
                parts = line.split()
                kb_idx = next(i for i, p in enumerate(parts) if "KiB" in p)
                info["weights_kib"] = float(parts[kb_idx - 1].strip("("))
                info["weights_b"] = int(parts[-6].replace(",", ""))
            except Exception:
                pass
        if "activations (rw)" in line:
            try:
                parts = line.split()
                kb_idx = next(i for i, p in enumerate(parts) if "KiB" in p)
                info["activations_kib"] = float(parts[kb_idx - 1].strip("("))
            except Exception:
                pass
        if "macc" in line and "weights" in line and "act" in line:
            try:
                for tok in line.split():
                    if tok.startswith("macc="):
                        info["macc"] = int(tok.split("=")[1].replace(",", ""))
            except Exception:
                pass

    print(out, file=sys.stderr)
    return info


def run_validate(model_path: Path, name: str, target: str,
                 work_dir: Path, out_dir: Path,
                 val_npy: Path | None) -> dict:
    """stedgeai validate --mode host 실행 → INT8 예측값 반환.

    val_npy: shape (N, T, F) float32 ndarray 파일 경로.
    반환: {"ok": bool, "outputs_npy": Path | None, "raw": str}
    """
    stedgeai = _get_stedgeai()
    csv_dir = out_dir / "val_csvs"
    csv_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        stedgeai, "validate",
        "--target", target,
        "--model", str(model_path),
        "--type", "keras",
        "--name", name,
        "--output", str(out_dir),
        "--workspace", str(work_dir),
        "--mode", "host",
        "--classifier",
        "--save-csv",
        "--quiet",
    ]
    if val_npy is not None and val_npy.exists():
        # stedgeai 는 .npy 를 valinput으로 받음
        cmd += ["--valinput", str(val_npy)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout + result.stderr
    print(out, file=sys.stderr)

    info: dict = {"validate_ok": result.returncode == 0, "raw": out}

    # 출력 CSV 탐색 (stedgeai는 <name>_val_output_?.csv 로 저장)
    csvs = list(out_dir.glob(f"{name}_val_output*.csv"))
    if csvs:
        info["outputs_csv"] = str(csvs[0])

    return info


def compute_minp_from_csv(outputs_csv: str, labels: np.ndarray,
                           threshold: float = 0.5) -> dict:
    """stedgeai validate 출력 CSV → softmax probs → window-level MinP."""
    import csv

    probs = []
    with open(outputs_csv) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            vals = [float(v) for v in row if v.strip()]
            if len(vals) >= 2:
                probs.append(vals)

    if not probs:
        return {"error": "empty csv"}

    probs = np.array(probs)  # (N, 2)
    preds = (probs[:, 1] >= threshold).astype(int)
    n = len(labels)
    if n != len(preds):
        return {"error": f"length mismatch: labels={n} preds={len(preds)}"}

    pos_mask = labels == 1
    neg_mask = labels == 0
    fall_p = preds[pos_mask].mean() if pos_mask.any() else float("nan")
    nfall_p = (1 - preds[neg_mask]).mean() if neg_mask.any() else float("nan")
    minp = min(fall_p, nfall_p)
    return {
        "fall_precision": float(fall_p),
        "nfall_precision": float(nfall_p),
        "min_precision": float(minp),
        "n_samples": int(n),
    }


def run_stedgeai_pipeline(
    exp_dir: Path,
    val_npy: Path | None = None,
    val_labels: np.ndarray | None = None,
    target: str = "stm32n6",
    threshold: float = 0.5,
    keep_compat_keras: bool = False,
    model_name: str = "model.keras",
) -> dict:
    """전체 파이프라인: compat .keras 생성 → analyze → validate → 정리."""
    keras_path = exp_dir / model_name
    if not keras_path.exists():
        keras_path = exp_dir / "model.keras"
    if not keras_path.exists():
        return {"error": "model.keras not found"}

    compat_path = exp_dir / "model_stedgeai_compat.keras"
    work_dir = Path(tempfile.mkdtemp(prefix="stedgeai_ws_"))
    out_dir  = Path(tempfile.mkdtemp(prefix="stedgeai_out_"))

    result: dict = {}

    try:
        print(f"[stedgeai] making compat .keras for {exp_dir.name}", flush=True)
        make_compat_keras(keras_path, compat_path)

        name = exp_dir.name.replace("-", "_").lower()

        # ── analyze ─────────────────────────────────────────────────────────
        print(f"[stedgeai] analyze ...", flush=True)
        analyze = run_analyze(compat_path, name, target, work_dir, out_dir)
        result["analyze"] = {k: v for k, v in analyze.items() if k != "raw"}
        if not analyze["analyze_ok"]:
            result["error"] = "analyze failed"
            return result

        # ── validate ────────────────────────────────────────────────────────
        if val_npy is not None:
            print(f"[stedgeai] validate --mode host ...", flush=True)
            validate = run_validate(compat_path, name, target, work_dir, out_dir, val_npy)
            result["validate"] = {k: v for k, v in validate.items() if k not in ("raw", "outputs_csv")}

            if validate.get("validate_ok") and "outputs_csv" in validate and val_labels is not None:
                metrics = compute_minp_from_csv(
                    validate["outputs_csv"], val_labels, threshold
                )
                result["int8_window_metrics"] = metrics
                print(f"[stedgeai] INT8 window MinP={metrics.get('min_precision', '?'):.4f}", flush=True)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        if not keep_compat_keras and compat_path.exists():
            compat_path.unlink()

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True, help="실험 디렉토리 (model.keras 포함)")
    parser.add_argument("--val-npy", default=None, help="검증 윈도우 npy (N,T,F)")
    parser.add_argument("--val-labels-npy", default=None, help="검증 레이블 npy (N,)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target", default="stm32n6")
    parser.add_argument("--keep-compat", action="store_true",
                        help="compat .keras 파일 삭제하지 않음")
    parser.add_argument("--model", default="model.keras",
                        help="사용할 모델 파일명 (기본: model.keras)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    val_npy = Path(args.val_npy) if args.val_npy else None
    val_labels = np.load(args.val_labels_npy) if args.val_labels_npy else None

    result = run_stedgeai_pipeline(
        exp_dir=exp_dir,
        val_npy=val_npy,
        val_labels=val_labels,
        target=args.target,
        threshold=args.threshold,
        keep_compat_keras=args.keep_compat,
        model_name=args.model,
    )

    # metrics.json 에 stedgeai 결과 추가
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        m["stedgeai"] = result
        metrics_path.write_text(json.dumps(m, indent=2, ensure_ascii=False))
        print(f"[stedgeai] metrics.json updated", flush=True)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
