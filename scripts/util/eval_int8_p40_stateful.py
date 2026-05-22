#!/usr/bin/env python3
"""P40 stateful fine-tuning 모델 INT8 평가 스크립트.

Phase 40 stateful 모델의 weights를 stateless proxy로 이식 후
STedgeAI --mode host로 channel-wise INT8 예측을 수행하고,
학습 시와 동일한 event_vote 로직으로 MinPR 계산.

Float 평가 (train_stateful_finetune.py)와 동일 기준:
  threshold = 학습 시 최적값 (0.75), vote_window=5, vote_k=3

Usage:
    uv run python scripts/eval_int8_p40.py
    uv run python scripts/eval_int8_p40.py --reselect-threshold
    uv run python scripts/eval_int8_p40.py --eval-stride 5
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

PROJ_ROOT  = Path(__file__).resolve().parents[1]
STEDGE_PY  = Path("/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python")
STEDGEAI   = Path("/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/stedgeai")

P40_DIR    = PROJ_ROOT / "results/phase40_stateful/P40-nv-a65-stateful"
P38_DIR    = PROJ_ROOT / "results/phase36_window_ablation/P38-nv-a65-gru"

WINDOW_SIZE  = 40
VOTE_WINDOW  = 5
VOTE_K       = 3
FLOAT_THR    = 0.75   # 학습 시 선택된 threshold

KP13_COLS = [
    "kp0_y","kp0_x","kp0_s",
    "kp5_y","kp5_x","kp5_s","kp6_y","kp6_x","kp6_s",
    "kp7_y","kp7_x","kp7_s","kp8_y","kp8_x","kp8_s",
    "kp9_y","kp9_x","kp9_s","kp10_y","kp10_x","kp10_s",
    "kp11_y","kp11_x","kp11_s","kp12_y","kp12_x","kp12_s",
    "kp13_y","kp13_x","kp13_s","kp14_y","kp14_x","kp14_s",
    "kp15_y","kp15_x","kp15_s","kp16_y","kp16_x","kp16_s",
    "HSSC_y","HSSC_x","RWHC","VHSSC","AHSSC","AHSSC_x",
]  # 45 features


# ── proxy 모델 빌드 (train_stateful_finetune.py 동일 로직) ────────────────────

def build_eval_proxy(stateful_model, window_size: int, n_feat: int):
    import tensorflow as tf
    sf_convs  = [l for l in stateful_model.layers if "sf_conv" in l.name]
    sf_grus   = [l for l in stateful_model.layers if "sf_gru"  in l.name]
    sf_denses = [l for l in stateful_model.layers
                 if l.name.startswith("sf_dense") or l.name == "sf_out"]

    inp = tf.keras.Input(shape=(window_size, n_feat))
    x   = inp

    proxy_convs, proxy_grus, proxy_denses = [], [], []

    for cl in sf_convs:
        new_l = tf.keras.layers.Conv1D(
            cl.filters, cl.kernel_size[0], padding="causal", activation="relu")
        x = new_l(x)
        proxy_convs.append(new_l)

    for i, gl in enumerate(sf_grus):
        is_last = (i == len(sf_grus) - 1)
        new_l = tf.keras.layers.GRU(
            gl.units, return_sequences=not is_last,
            dropout=gl.dropout, recurrent_dropout=0.0, reset_after=True)
        x = new_l(x)
        proxy_grus.append(new_l)

    for dl in sf_denses:
        act = "softmax" if dl.name == "sf_out" else "relu"
        new_l = tf.keras.layers.Dense(dl.units, activation=act)
        x = new_l(x)
        proxy_denses.append(new_l)

    proxy = tf.keras.Model(inp, x)
    for src, dst in zip(sf_convs,  proxy_convs):  dst.set_weights(src.get_weights())
    for src, dst in zip(sf_grus,   proxy_grus):   dst.set_weights(src.get_weights())
    for src, dst in zip(sf_denses, proxy_denses): dst.set_weights(src.get_weights())
    return proxy


# ── STedgeAI compat .keras 생성 ──────────────────────────────────────────────

def _strip_compat_fields(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for v in obj.values():
            _strip_compat_fields(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_compat_fields(item)


def make_compat_keras(src_keras: Path, dst_keras: Path) -> None:
    tmpdir = Path(tempfile.mkdtemp())
    try:
        with zipfile.ZipFile(src_keras, "r") as z:
            z.extractall(tmpdir)
        for cfg_file in tmpdir.rglob("config.json"):
            cfg = json.loads(cfg_file.read_text())
            _strip_compat_fields(cfg)
            cfg_file.write_text(json.dumps(cfg))
        with zipfile.ZipFile(dst_keras, "w", zipfile.ZIP_DEFLATED) as z:
            for f in tmpdir.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(tmpdir))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── 비디오 데이터 로드 ────────────────────────────────────────────────────────

def load_video_frames(csv_path: Path, feat_cols: list[str], norm_min, norm_scale):
    data = defaultdict(lambda: {"feat": [], "label": [], "frame": []})
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            try:
                lbl  = int(row["label"])
                fidx = int(row.get("frame", 0))
                feat = [float(row[c]) if row.get(c) else 0.0 for c in feat_cols]
            except (KeyError, ValueError):
                continue
            data[vid]["feat"].append((fidx, feat))
            data[vid]["label"].append((fidx, lbl))

    result = {}
    for vid, d in data.items():
        if not d["feat"]:
            continue
        d["feat"].sort(key=lambda x: x[0])
        d["label"].sort(key=lambda x: x[0])
        frames = np.array([f for _, f in d["feat"]], np.float32)
        labels = np.array([l for _, l in d["label"]], np.int32)
        # normalize
        frames = np.clip((frames - norm_min) / norm_scale, 0.0, 1.0)
        result[vid] = (frames, labels)
    return result


# ── 윈도우 추출 (eval용, stride 적용) ────────────────────────────────────────

def extract_windows_for_stedgeai(video_data: dict, window_size: int, eval_stride: int):
    """STedgeAI validate 입력용 윈도우 배열 + 레이블 + 그룹 반환."""
    windows, labels, groups = [], [], []
    for vid, (frames, lbs) in video_data.items():
        n = len(frames)
        for start in range(0, n - window_size + 1, eval_stride):
            chunk = frames[start:start + window_size]
            lbl   = int(lbs[start:start + window_size].max())
            windows.append(chunk)
            labels.append(lbl)
            groups.append(str(vid))
    x = np.stack(windows, 0).astype(np.float32) if windows else np.empty((0, window_size, len(norm_min_placeholder)), np.float32)
    return x, np.array(labels, np.int32), np.array(groups, str)


# ── STedgeAI validate --mode host ────────────────────────────────────────────

def run_stedgeai_validate(model_path: Path, name: str, windows_npy: Path,
                          out_dir: Path, work_dir: Path) -> Path | None:
    cmd = [
        str(STEDGEAI), "validate",
        "--target", "stm32h7",
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
        print(f"[STedgeAI] validate FAILED:\n{result.stderr[-3000:]}", file=sys.stderr)
        return None
    csvs = list(out_dir.glob(f"{name}_val_c_outputs_*.csv"))
    if not csvs:
        csvs = list(out_dir.glob(f"{name}_val_output*.csv"))
    return csvs[0] if csvs else None


def load_predictions(csv_path: Path) -> np.ndarray:
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                vals = [float(v) for v in row if v.strip()]
            except ValueError:
                continue
            if len(vals) >= 2:
                rows.append(vals[:2])
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, 2), np.float32)


# ── event_vote 평가 ────────────────────────────────────────────────────────────

def apply_event_vote(groups: np.ndarray, scores: np.ndarray,
                     labels: np.ndarray, threshold: float,
                     vote_window: int = 5, vote_k: int = 3) -> dict:
    # 비디오별 스코어 집계
    from collections import defaultdict as dd
    vid_scores = dd(list)
    vid_label  = {}
    for g, s, l in zip(groups, scores, labels):
        vid_scores[g].append(float(s))
        vid_label[g] = int(vid_label.get(g, 0) or l)

    tp = fp = fn = tn = 0
    for vid, scs in vid_scores.items():
        has_fall = bool(vid_label[vid])
        buf  = np.zeros(vote_window, dtype=np.int32)
        head = 0
        vsum = 0
        detected = False
        for s in scs:
            this_v = 1 if s >= threshold else 0
            old_v  = int(buf[head])
            buf[head] = this_v
            head = (head + 1) % vote_window
            vsum += this_v - old_v
            if not detected and vsum >= vote_k:
                detected = True

        if   detected and has_fall:      tp += 1
        elif detected and not has_fall:  fp += 1
        elif not detected and has_fall:  fn += 1
        else:                            tn += 1

    if tp + fp == 0 or tp + fn == 0:
        return {"min_pr": 0.0, "fall_pr": 0.0, "nfall_pr": 0.0,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    fall_pr  = tp / (tp + fp)
    nfall_pr = tn / (tn + fn) if tn + fn > 0 else 0.0
    return {
        "min_pr":    round(min(fall_pr, nfall_pr), 6),
        "fall_pr":   round(fall_pr, 6),
        "nfall_pr":  round(nfall_pr, 6),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def threshold_sweep(groups, scores, labels, vote_window=5, vote_k=3):
    best_thr, best_minpr = FLOAT_THR, 0.0
    for thr in np.arange(0.40, 0.96, 0.025):
        r = apply_event_vote(groups, scores, labels, float(thr), vote_window, vote_k)
        if r["min_pr"] > best_minpr:
            best_minpr = r["min_pr"]
            best_thr   = float(thr)
    return round(best_thr, 4), best_minpr


# ── STedgeAI analyze ─────────────────────────────────────────────────────────

def run_stedgeai_analyze(model_path: Path, work_dir: Path) -> dict:
    out_dir = work_dir / "analyze_out"
    out_dir.mkdir(exist_ok=True)
    cmd = [
        str(STEDGEAI), "analyze",
        "--target", "stm32n6",
        "--model", str(model_path),
        "--type", "keras",
        "--output", str(out_dir),
        "--workspace", str(work_dir),
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[STedgeAI] analyze FAILED:\n{result.stderr[-2000:]}", file=sys.stderr)
        return {"analyze_ok": False}

    # metrics.json 파싱
    for mf in out_dir.rglob("metrics.json"):
        try:
            data = json.loads(mf.read_text())
            info = data.get("stedgeai", data)
            an   = info.get("analyze", info)
            return {
                "analyze_ok":    True,
                "weights_kib":   an.get("weights_kib") or an.get("weights", {}).get("kib"),
                "activations_kib": an.get("activations_kib") or an.get("activations", {}).get("kib"),
                "macc":          an.get("macc"),
            }
        except Exception:
            continue

    # 대안: 로그에서 파싱
    for line in result.stdout.splitlines():
        if "weights" in line.lower() and "kib" in line.lower():
            print(f"  [analyze log] {line.strip()}")
    return {"analyze_ok": True, "raw": result.stdout[-500:]}


# ── 메인 ─────────────────────────────────────────────────────────────────────

norm_min_placeholder = []  # hack for type hint in extract_windows_for_stedgeai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reselect-threshold", action="store_true",
                        help="val INT8 스코어로 threshold 재탐색")
    parser.add_argument("--eval-stride", type=int, default=5,
                        help="윈도우 샘플링 stride (기본 5)")
    parser.add_argument("--test-csv",
                        default=str(PROJ_ROOT / "dataset/splits_v2_class_balanced_filtered/test.csv"))
    parser.add_argument("--val-csv",
                        default=str(PROJ_ROOT / "dataset/splits_v2_class_balanced_filtered/val.csv"))
    args = parser.parse_args()

    test_csv = Path(args.test_csv)
    val_csv  = Path(args.val_csv)

    # ── 이미 평가됐는지 확인 ──────────────────────────────────────────────────
    met_path = P40_DIR / "metrics.json"
    mets     = json.loads(met_path.read_text())
    if not args.reselect_threshold and mets.get("int8_eval", {}).get("eval_ok"):
        print("이미 INT8 평가 완료. --reselect-threshold로 재실행 가능.")
        r = mets["int8_eval"]
        print(f"  MinPR={r['min_pr']:.4f}  FallPR={r['fall_pr']:.4f}  "
              f"NFallPR={r['nfall_pr']:.4f}  FP={r['fp']}  FN={r['fn']}")
        return

    # ── 정규화 파라미터 로드 ──────────────────────────────────────────────────
    norm = json.loads((P40_DIR / "normalization.json").read_text())
    mn   = np.array(norm["min"],   dtype=np.float32)
    sc   = np.array(norm["scale"], dtype=np.float32)
    n_feat = len(mn)
    print(f"정규화 파라미터: {n_feat}개 피처")

    # ── stateful 모델 → proxy 빌드 ──────────────────────────────────────────
    print("Stateful 모델 로드 및 proxy 빌드 중...")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    stateful_model = tf.keras.models.load_model(
        str(P40_DIR / "model_stateful.keras"), compile=False)
    proxy = build_eval_proxy(stateful_model, WINDOW_SIZE, n_feat)
    proxy.summary(line_length=80)

    proxy_keras = P40_DIR / "model_proxy.keras"
    proxy.save(str(proxy_keras))
    print(f"Proxy 모델 저장: {proxy_keras}")

    # ── STedgeAI compat .keras ────────────────────────────────────────────────
    compat_keras = P40_DIR / "model_proxy_compat.keras"
    print("STedgeAI compat .keras 생성 중...")
    make_compat_keras(proxy_keras, compat_keras)
    print(f"Compat .keras: {compat_keras}")

    # ── STedgeAI analyze ─────────────────────────────────────────────────────
    work_tmp = Path(tempfile.mkdtemp(prefix="p40_analyze_"))
    print("STedgeAI analyze (stm32n6) ...")
    analyze_res = run_stedgeai_analyze(compat_keras, work_tmp)
    shutil.rmtree(work_tmp, ignore_errors=True)
    print(f"  analyze: {analyze_res}")

    # ── 데이터 로드 & 윈도우 추출 ─────────────────────────────────────────────
    print(f"Test 데이터 로드: {test_csv.name} (stride={args.eval_stride}) ...")
    test_data = load_video_frames(test_csv, KP13_COLS, mn, sc)
    x_test, y_test, g_test = _extract_windows(test_data, WINDOW_SIZE, args.eval_stride)
    print(f"  test windows={len(y_test):,}  fall={y_test.sum():,}  "
          f"videos={len(np.unique(g_test)):,}")

    if args.reselect_threshold:
        print(f"Val 데이터 로드: {val_csv.name} ...")
        val_data = load_video_frames(val_csv, KP13_COLS, mn, sc)
        x_val, y_val, g_val = _extract_windows(val_data, WINDOW_SIZE, args.eval_stride)
        print(f"  val windows={len(y_val):,}")
    else:
        x_val = y_val = g_val = None

    # ── STedgeAI validate: test ───────────────────────────────────────────────
    name = "p40_nv_a65_proxy"
    out_dir  = Path(tempfile.mkdtemp(prefix="p40_out_"))
    work_dir = Path(tempfile.mkdtemp(prefix="p40_ws_"))

    try:
        test_npy = Path(tempfile.mktemp(suffix=".npy"))
        np.save(test_npy, x_test)
        print(f"STedgeAI validate --mode host (test, {len(y_test):,} windows) ...")
        pred_csv = run_stedgeai_validate(compat_keras, name, test_npy, out_dir, work_dir)
        test_npy.unlink(missing_ok=True)

        if pred_csv is None:
            print("FAIL — STedgeAI validate 실패", file=sys.stderr)
            return

        probs_test = load_predictions(pred_csv)
        if len(probs_test) != len(y_test):
            print(f"FAIL — 예측 수 불일치: {len(probs_test)} vs {len(y_test)}", file=sys.stderr)
            return
        scores_test = probs_test[:, 1]

        # ── Val threshold 재탐색 ──────────────────────────────────────────────
        use_thr = FLOAT_THR
        if args.reselect_threshold and x_val is not None:
            val_out = Path(tempfile.mkdtemp(prefix="p40_val_out_"))
            val_npy = Path(tempfile.mktemp(suffix=".npy"))
            np.save(val_npy, x_val)
            print(f"STedgeAI validate --mode host (val, {len(y_val):,} windows) ...")
            val_pred_csv = run_stedgeai_validate(compat_keras, name + "_val",
                                                  val_npy, val_out, work_dir)
            val_npy.unlink(missing_ok=True)
            shutil.rmtree(val_out, ignore_errors=True)
            if val_pred_csv:
                probs_val = load_predictions(val_pred_csv)
                scores_val = probs_val[:, 1]
                use_thr, best_minpr = threshold_sweep(g_val, scores_val, y_val)
                print(f"  INT8 val threshold 재탐색: thr={use_thr:.4f} minpr={best_minpr:.4f}")

        # ── 최종 평가 ─────────────────────────────────────────────────────────
        print(f"event_vote 평가 (thr={use_thr:.4f}, vote={VOTE_K}/{VOTE_WINDOW}) ...")
        result = apply_event_vote(g_test, scores_test, y_test, use_thr, VOTE_WINDOW, VOTE_K)

        print(f"\n{'='*55}")
        print(f"P40-nv-a65-stateful  INT8 (STedgeAI host mode)")
        print(f"{'='*55}")
        print(f"  MinPR   = {result['min_pr']:.4f}")
        print(f"  FallPR  = {result['fall_pr']:.4f}  NFallPR = {result['nfall_pr']:.4f}")
        print(f"  TP={result['tp']}  TN={result['tn']}  FP={result['fp']}  FN={result['fn']}")
        print(f"  threshold={use_thr:.4f}  vote={VOTE_K}/{VOTE_WINDOW}")
        print(f"{'='*55}")
        print(f"  [참고] Float  MinPR = 0.9543  FP=19  FN=10")
        if analyze_res.get("analyze_ok"):
            print(f"  [참고] Flash={analyze_res.get('weights_kib','?')} KiB  "
                  f"RAM={analyze_res.get('activations_kib','?')} KiB  "
                  f"MACC={analyze_res.get('macc','?')}")

        # ── metrics.json 업데이트 ─────────────────────────────────────────────
        mets["int8_eval"] = {
            "eval_ok": True,
            "threshold": use_thr,
            "vote_window": VOTE_WINDOW,
            "vote_k": VOTE_K,
            "threshold_source": "int8_reselected" if args.reselect_threshold else "float_training",
            **result,
        }
        mets["stedgeai_proxy"] = analyze_res
        met_path.write_text(json.dumps(mets, indent=2, ensure_ascii=False))
        print(f"\nmetrics.json 업데이트 완료 → int8_eval, stedgeai_proxy")

    finally:
        shutil.rmtree(out_dir,  ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)


def _extract_windows(video_data: dict, window_size: int, stride: int):
    windows, labels, groups = [], [], []
    for vid, (frames, lbs) in video_data.items():
        n = len(frames)
        for start in range(0, n - window_size + 1, stride):
            windows.append(frames[start:start + window_size])
            labels.append(int(lbs[start:start + window_size].max()))
            groups.append(str(vid))
    if not windows:
        return np.empty((0, window_size, frames.shape[1] if video_data else 45), np.float32), \
               np.empty(0, np.int32), np.empty(0, str)
    return np.stack(windows, 0).astype(np.float32), \
           np.array(labels, np.int32), \
           np.array(groups, str)


if __name__ == "__main__":
    main()
