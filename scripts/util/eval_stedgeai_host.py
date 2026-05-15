#!/usr/bin/env python3
"""STedgeAI --mode host evaluation with full video-level MinP metric.

학습 시와 동일한 평가 기준 적용:
  - test CSV → 윈도우 추출 (학습 normalization 사용)
  - stedgeai validate --mode host → per-window INT8 예측
  - threshold × min_consecutive sweep → video-level 집계
  - MinP = min(FallPrecision, NFallPrecision)

Usage:
    # 학습된 threshold 사용 (기본)
    uv run python scripts/util/eval_stedgeai_host.py \
        --exp-dir results/gru_phase7_quant/Q7-v01

    # val set으로 INT8 최적 threshold 재탐색
    uv run python scripts/util/eval_stedgeai_host.py \
        --exp-dir results/gru_phase7_quant/Q7-v01 \
        --reselect-threshold

    # 여러 실험 한 번에
    uv run python scripts/util/eval_stedgeai_host.py \
        --exp-dir results/gru_phase7_quant/Q7-v01 results/gru_phase7_quant/Q7-v02
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

STEDGE_PY = Path("/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python")
STEDGEAI  = Path("/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/stedgeai")

PROJ_ROOT = Path(__file__).resolve().parents[2]


# ── 피처 선택 (train_baseline.py 동일 로직) ─────────────────────────────────

_KP_RAW = [f"kp{i}_{ax}" for i in range(17) for ax in ("x", "y", "conf")]  # 51개

_FEATURE_SETS: dict[str, list[str]] = {
    "kp7":  ([f"kp{i}_{ax}" for i in range(7)  for ax in ("x","y","conf")]
             + ["HSSC_y","HSSC_x","RWHC","VHSSC"]),
    "kp12": ([f"kp{i}_{ax}" for i in range(13) for ax in ("x","y","conf")]
             + ["HSSC_y","HSSC_x","RWHC","VHSSC"]),
    "kp8":  ([f"kp{i}_{ax}" for i in [0,5,6,7,8,11,12,13] for ax in ("x","y","conf")]
             + ["HSSC_y","HSSC_x","RWHC","VHSSC"]),
    "all":  (_KP_RAW + ["HSSC_y","HSSC_x","RWHC","VHSSC"]),
    "minimal": (["HSSC_y","HSSC_x","RWHC","VHSSC","AHSSC","AHSSC_x"]
                + [f"kp{i}_{ax}" for i in [0,5,6,11,12] for ax in ("x","y","conf")]),
}
_FILTERED_EXTRA = ["AHSSC", "AHSSC_x"]


def feature_columns(df_cols: list[str], feature_set: str, preprocessing: str) -> list[str]:
    base = _FEATURE_SETS.get(feature_set, _FEATURE_SETS["kp12"])
    if preprocessing == "filtered":
        base = base + [c for c in _FILTERED_EXTRA if c not in base]
    return [c for c in base if c in df_cols]


# ── 윈도우 레이블 ─────────────────────────────────────────────────────────────

def window_label(labels: np.ndarray) -> int:
    return 1 if labels.max() > 0 else 0


# ── 윈도우 추출 ──────────────────────────────────────────────────────────────

def build_test_windows(
    csv_path: Path,
    feat_cols: list[str],
    norm_min: np.ndarray,
    norm_scale: np.ndarray,
    target_steps: int,
    window_start_sec: float,
    window_end_sec: float,
    label_column: str = "label",
    eval_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (windows, labels, groups) for the test split.

    windows : (N, T, F) float32 — normalized
    labels  : (N,) int32        — binary (0/1)
    groups  : (N,) str          — video_id per window
    """
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in CSV: {missing[:5]}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found")

    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df[feat_cols] = df[feat_cols].fillna(0.0)
    df["video_id"] = df["video_id"].astype(str)
    df["eval_label"] = (df[label_column] > 0).astype(np.int32)

    windows, labels, groups = [], [], []

    for video_id, grp in df.groupby("video_id", sort=False):
        seg = grp[(grp["time_sec"] >= window_start_sec) & (grp["time_sec"] < window_end_sec)]
        if len(seg) < target_steps:
            continue
        vals = seg[feat_cols].to_numpy(dtype=np.float32)
        ylabs = seg["eval_label"].to_numpy(dtype=np.int32)
        for start in range(0, len(seg) - target_steps + 1, eval_stride):
            chunk = vals[start: start + target_steps]
            lbl   = window_label(ylabs[start: start + target_steps])
            windows.append(chunk)
            labels.append(lbl)
            groups.append(str(video_id))

    if not windows:
        raise ValueError("No windows extracted from test CSV. Check time range / target_steps.")

    x = np.stack(windows, axis=0).astype(np.float32)
    # minmax normalization using training stats
    x = (x - norm_min) / norm_scale
    x = np.clip(x, 0.0, 1.0)

    return x, np.array(labels, dtype=np.int32), np.array(groups, dtype=str)


# ── STedgeAI compat .keras 준비 ───────────────────────────────────────────────

def ensure_compat_keras(exp_dir: Path) -> Path:
    """Keras 3.7 호환 .keras 반환 (없으면 생성)."""
    compat = exp_dir / "model_stedgeai_compat.keras"
    if compat.exists():
        return compat
    src = exp_dir / "model.keras"
    if not src.exists():
        raise FileNotFoundError(f"model.keras not found in {exp_dir}")
    # STedgeAI 내부 Python으로 stripping 실행
    script = PROJ_ROOT / "scripts" / "util" / "export_stedgeai.py"
    result = subprocess.run(
        [str(STEDGE_PY), str(script), "--exp-dir", str(exp_dir),
         "--target", "stm32n6", "--keep-compat"],
        capture_output=True, text=True
    )
    if not compat.exists():
        raise RuntimeError(f"compat .keras 생성 실패:\n{result.stderr[-2000:]}")
    return compat


# ── STedgeAI validate --mode host ─────────────────────────────────────────────

def run_validate_host(
    model_path: Path,
    name: str,
    windows_npy: Path,
    out_dir: Path,
    work_dir: Path,
    host_target: str = "stm32h7",
) -> Path | None:
    """stedgeai validate --mode host 실행 → 예측 CSV 경로 반환.

    Note: stm32n6 (Cortex-M55) 은 --mode host 미지원.
    stm32h7 (Cortex-M7) 타겟으로 대체 — INT8 channel-wise 양자화는 동일,
    타겟 아키텍처 차이만 있으므로 MinP 근사값으로 사용.
    """
    cmd = [
        str(STEDGEAI), "validate",
        "--target", host_target,
        "--model", str(model_path),
        "--type", "keras",
        "--name", name,
        "--output", str(out_dir),
        "--workspace", str(work_dir),
        "--mode", "host",
        "--classifier",
        "--save-csv",
        "--valinput", str(windows_npy),
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[stedgeai] validate failed:\n{result.stderr[-3000:]}", file=sys.stderr)
        return None

    csvs = list(out_dir.glob(f"{name}_val_output*.csv"))
    return csvs[0] if csvs else None


# ── 예측 CSV 파싱 ─────────────────────────────────────────────────────────────

def load_predictions(csv_path: Path) -> np.ndarray:
    """stedgeai validate 출력 CSV → (N, 2) float32 softmax 확률."""
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            vals = [float(v) for v in row if v.strip()]
            if len(vals) >= 2:
                rows.append(vals[:2])
    if not rows:
        raise ValueError(f"Empty predictions CSV: {csv_path}")
    return np.array(rows, dtype=np.float32)


# ── video-level 평가 (train_baseline.py 동일) ─────────────────────────────────

def apply_consecutive_rule(binary: np.ndarray, min_consec: int) -> np.ndarray:
    if min_consec <= 1:
        return binary
    out = np.zeros_like(binary)
    count = 0
    for i, v in enumerate(binary):
        count = count + 1 if v else 0
        if count >= min_consec:
            out[i] = 1
    return out


def video_level_eval(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    min_consecutive: int,
) -> tuple[np.ndarray, np.ndarray]:
    video_ids = np.unique(groups)
    v_true = np.empty(len(video_ids), dtype=np.int32)
    v_pred = np.empty(len(video_ids), dtype=np.int32)
    for i, vid in enumerate(video_ids):
        mask = groups == vid
        v_true[i] = int(labels[mask].max())
        binary = (scores[mask] >= threshold).astype(np.int32)
        filtered = apply_consecutive_rule(binary, min_consecutive)
        v_pred[i] = int(filtered.max()) if len(filtered) > 0 else 0
    return v_true, v_pred


def compute_metrics(v_true: np.ndarray, v_pred: np.ndarray) -> dict:
    tp = int(((v_true == 1) & (v_pred == 1)).sum())
    tn = int(((v_true == 0) & (v_pred == 0)).sum())
    fp = int(((v_true == 0) & (v_pred == 1)).sum())
    fn = int(((v_true == 1) & (v_pred == 0)).sum())
    fall_p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    nfall_p = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1      = 2 * fall_p * recall / (fall_p + recall) if (fall_p + recall) > 0 else 0.0
    return {
        "fall_precision":  round(fall_p, 6),
        "nfall_precision": round(nfall_p, 6),
        "min_precision":   round(min(fall_p, nfall_p), 6),
        "recall":          round(recall, 6),
        "f1":              round(f1, 6),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_videos": int(len(v_true)),
    }


def threshold_sweep(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    min_consecutive_values: list[int],
    min_precision: float = 0.90,
) -> dict:
    """val set 기준 최적 (threshold, min_consecutive) 선택."""
    best = None
    for thr in np.linspace(0.05, 0.95, 37):
        for mc in min_consecutive_values:
            v_true, v_pred = video_level_eval(labels, scores, groups, float(thr), mc)
            m = compute_metrics(v_true, v_pred)
            if m["fall_precision"] >= min_precision and m["nfall_precision"] >= min_precision:
                if best is None or m["f1"] > best["f1"] or (m["f1"] == best["f1"] and m["recall"] > best["recall"]):
                    best = {"threshold": round(float(thr), 4), "min_consecutive": mc, **m}
    return best or {"threshold": 0.5, "min_consecutive": 1}


# ── 메인 평가 루프 ─────────────────────────────────────────────────────────────

def evaluate_one(exp_dir: Path, reselect_threshold: bool = False) -> dict | None:
    exp_dir = exp_dir.resolve()
    print(f"\n{'='*60}")
    print(f"[{exp_dir.name}]")

    # config / metrics 로드
    cfg_path = exp_dir / "run_config.resolved.json"
    met_path = exp_dir / "metrics.json"
    norm_path = exp_dir / "normalization.json"

    if not cfg_path.exists():
        print(f"  SKIP — run_config.resolved.json not found"); return None
    if not met_path.exists():
        print(f"  SKIP — metrics.json not found"); return None
    if not norm_path.exists():
        print(f"  SKIP — normalization.json not found"); return None

    cfg  = json.loads(cfg_path.read_text())
    mets = json.loads(met_path.read_text())
    norm = json.loads(norm_path.read_text())

    norm_min   = np.array(norm["min"],   dtype=np.float32).reshape(1, 1, -1)
    norm_scale = np.array(norm["scale"], dtype=np.float32).reshape(1, 1, -1)

    feature_set  = cfg.get("feature_set", "kp12")
    preprocessing= cfg.get("preprocessing", "raw")
    target_steps = int(cfg.get("target_steps", 40))
    win_start    = float(cfg.get("window_start_sec", 3.0))
    win_end      = float(cfg.get("window_end_sec", 9.0))
    label_col    = cfg.get("label_column", "label")
    min_consec_vals = cfg.get("min_consecutive_values", [1, 3, 5])
    test_csv     = Path(cfg.get("test_csv", ""))
    val_csv      = Path(cfg.get("val_csv", ""))

    if not test_csv.is_absolute():
        test_csv = PROJ_ROOT / test_csv
    if not val_csv.is_absolute():
        val_csv = PROJ_ROOT / val_csv

    if not test_csv.exists():
        print(f"  SKIP — test CSV not found: {test_csv}"); return None

    # 기존 threshold (학습 시 val 기준으로 선택된 값)
    trained_thr  = mets.get("threshold_selection", {}).get("threshold", 0.5)
    trained_mc   = mets.get("threshold_selection", {}).get("min_consecutive", 1)

    # 이미 평가됐는지 확인
    if not reselect_threshold and mets.get("stedgeai_host_eval", {}).get("eval_ok"):
        print(f"  SKIP — already evaluated (use --reselect-threshold to re-run)")
        _print_result(exp_dir.name, mets["stedgeai_host_eval"])
        return mets["stedgeai_host_eval"]

    # feature columns — feature_columns.json 우선 사용 (hardcoded set보다 신뢰)
    feat_col_path = exp_dir / "feature_columns.json"
    if feat_col_path.exists():
        feat_cols = json.loads(feat_col_path.read_text())
    else:
        dummy_df = pd.read_csv(test_csv, nrows=1, low_memory=False)
        feat_cols = feature_columns(dummy_df.columns.tolist(), feature_set, preprocessing)
    print(f"  features={len(feat_cols)}  steps={target_steps}  "
          f"window={win_start}~{win_end}s")

    # 테스트 윈도우 추출
    print(f"  Extracting test windows from {test_csv.name} ...")
    x_test, y_test, g_test = build_test_windows(
        test_csv, feat_cols, norm_min, norm_scale,
        target_steps, win_start, win_end, label_col,
    )
    print(f"  test windows={len(y_test):,}  positive={y_test.sum():,}  "
          f"videos={len(np.unique(g_test)):,}")

    # val 윈도우 (threshold 재탐색 시)
    x_val = y_val = g_val = None
    if reselect_threshold and val_csv.exists():
        print(f"  Extracting val windows for threshold reselect ...")
        x_val, y_val, g_val = build_test_windows(
            val_csv, feat_cols, norm_min, norm_scale,
            target_steps, win_start, win_end, label_col,
        )
        print(f"  val windows={len(y_val):,}")

    # compat .keras 준비
    print(f"  Preparing compat .keras ...")
    try:
        compat_keras = ensure_compat_keras(exp_dir)
    except Exception as e:
        print(f"  FAIL — compat .keras: {e}"); return None

    name = exp_dir.name.replace("-", "_").lower()
    work_dir = Path(tempfile.mkdtemp(prefix="stedgeai_eval_ws_"))
    out_dir  = Path(tempfile.mkdtemp(prefix="stedgeai_eval_out_"))
    result: dict = {"eval_ok": False}

    try:
        # test set validate
        print(f"  stedgeai validate --mode host (test, {len(y_test):,} windows) ...")
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            test_npy = Path(f.name)
        np.save(test_npy, x_test)

        pred_csv = run_validate_host(compat_keras, name, test_npy, out_dir, work_dir)
        test_npy.unlink(missing_ok=True)

        if pred_csv is None:
            print(f"  FAIL — validate returned no CSV"); return None

        probs = load_predictions(pred_csv)
        if len(probs) != len(y_test):
            print(f"  FAIL — prediction count mismatch ({len(probs)} vs {len(y_test)})"); return None

        scores_test = probs[:, 1]  # fall class probability

        # val set threshold 재탐색
        if reselect_threshold and x_val is not None:
            print(f"  stedgeai validate --mode host (val, {len(y_val):,} windows) ...")
            val_out = Path(tempfile.mkdtemp(prefix="stedgeai_val_out_"))
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
                val_npy = Path(f.name)
            np.save(val_npy, x_val)
            val_pred_csv = run_validate_host(compat_keras, name + "_val", val_npy, val_out, work_dir)
            val_npy.unlink(missing_ok=True)
            shutil.rmtree(val_out, ignore_errors=True)

            if val_pred_csv:
                val_probs = load_predictions(val_pred_csv)
                scores_val = val_probs[:, 1]
                sel = threshold_sweep(y_val, scores_val, g_val, min_consec_vals)
                use_thr = sel["threshold"]
                use_mc  = sel["min_consecutive"]
                print(f"  INT8 threshold reselected: thr={use_thr:.2f} mc={use_mc}")
                result["int8_threshold_selection"] = {"threshold": use_thr, "min_consecutive": use_mc}
            else:
                use_thr, use_mc = trained_thr, trained_mc
        else:
            use_thr, use_mc = trained_thr, trained_mc

        # 최종 평가
        v_true, v_pred = video_level_eval(y_test, scores_test, g_test, use_thr, use_mc)
        metrics = compute_metrics(v_true, v_pred)

        result.update({
            "eval_ok": True,
            "threshold": use_thr,
            "min_consecutive": use_mc,
            "threshold_source": "int8_reselected" if reselect_threshold else "training",
            **metrics,
        })

        _print_result(exp_dir.name, result)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(out_dir,  ignore_errors=True)
        # compat .keras는 유지 (재사용)

    # metrics.json 업데이트
    mets["stedgeai_host_eval"] = result
    met_path.write_text(json.dumps(mets, indent=2, ensure_ascii=False))
    print(f"  metrics.json updated → stedgeai_host_eval")

    return result


def _print_result(name: str, r: dict) -> None:
    minp = r.get("min_precision", 0)
    fp   = r.get("fall_precision", 0)
    nfp  = r.get("nfall_precision", 0)
    rec  = r.get("recall", 0)
    f1   = r.get("f1", 0)
    thr  = r.get("threshold", "?")
    mc   = r.get("min_consecutive", "?")
    src  = r.get("threshold_source", "?")
    print(f"\n  ── {name} STedgeAI INT8 (host mode) ──")
    print(f"  MinP={minp:.4f}  FallP={fp:.4f}  NFallP={nfp:.4f}")
    print(f"  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"  threshold={thr:.3f}  min_consecutive={mc}  (source: {src})")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="STedgeAI --mode host evaluation with video-level MinP"
    )
    parser.add_argument("--exp-dir", nargs="+", required=True,
                        help="실험 디렉토리 (여러 개 가능)")
    parser.add_argument("--reselect-threshold", action="store_true",
                        help="val set INT8 예측으로 threshold 재탐색 (기본: 학습 시 선택값 사용)")
    args = parser.parse_args()

    results = {}
    for path in args.exp_dir:
        exp_dir = Path(path)
        r = evaluate_one(exp_dir, reselect_threshold=args.reselect_threshold)
        if r:
            results[exp_dir.name] = r

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"{'ID':<10} {'MinP':>7} {'FallP':>7} {'NFallP':>8} {'Recall':>8} {'F1':>7}")
        print("-" * 55)
        for name, r in results.items():
            print(f"{name:<10} {r.get('min_precision',0):>7.4f} "
                  f"{r.get('fall_precision',0):>7.4f} "
                  f"{r.get('nfall_precision',0):>8.4f} "
                  f"{r.get('recall',0):>8.4f} "
                  f"{r.get('f1',0):>7.4f}")


if __name__ == "__main__":
    main()
