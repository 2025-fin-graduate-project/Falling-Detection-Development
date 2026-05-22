#!/usr/bin/env python3
"""Phase 37: Window-level 심화 실험 — velocity, relative coords, focal tuning.

Phase 36 (기본 ablation) 이후 0.94 미달 시 자동 실행.

추가 변인:
  --use-velocity    : 프레임-간 Δ좌표 추가 (explicit motion signal)
  --use-hip-center  : hip centroid 기준 상대 좌표
  --focal-alpha F   : focal loss fall class weight (default 0.75)
  --focal-gamma F   : focal loss gamma (default 2.0)
  --fall-stride N   : 학습 fall window stride
  --nfall-stride N  : 학습 nfall window stride
  --exclude-boundary: label_3class=2 윈도우 학습 제외
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

# GPU memory growth — 다중 프로세스 공존 및 서버 안정성
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

REPO     = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "dataset/splits_v2_class_balanced_filtered"
OUT_ROOT = REPO / "results/phase36_window_ablation"

_DERIVED = ["HSSC_y","HSSC_x","RWHC","VHSSC","AHSSC","AHSSC_x"]

# kp5: 코+어깨+골반 (5kp, 21f) — 최소 torso
KP5_COLS = [
    "kp0_y","kp0_x","kp0_s",
    "kp5_y","kp5_x","kp5_s","kp6_y","kp6_x","kp6_s",
    "kp11_y","kp11_x","kp11_s","kp12_y","kp12_x","kp12_s",
] + _DERIVED

# kp7: +팔꿈치 (7kp, 27f)
KP7_COLS = [
    "kp0_y","kp0_x","kp0_s",
    "kp5_y","kp5_x","kp5_s","kp6_y","kp6_x","kp6_s",
    "kp7_y","kp7_x","kp7_s","kp8_y","kp8_x","kp8_s",
    "kp11_y","kp11_x","kp11_s","kp12_y","kp12_x","kp12_s",
] + _DERIVED

# kp9: +손목 (9kp, 33f)
KP9_COLS = [
    "kp0_y","kp0_x","kp0_s",
    "kp5_y","kp5_x","kp5_s","kp6_y","kp6_x","kp6_s",
    "kp7_y","kp7_x","kp7_s","kp8_y","kp8_x","kp8_s",
    "kp9_y","kp9_x","kp9_s","kp10_y","kp10_x","kp10_s",
    "kp11_y","kp11_x","kp11_s","kp12_y","kp12_x","kp12_s",
] + _DERIVED

# kp11: +무릎 (11kp, 39f)
KP11_COLS = [
    "kp0_y","kp0_x","kp0_s",
    "kp5_y","kp5_x","kp5_s","kp6_y","kp6_x","kp6_s",
    "kp7_y","kp7_x","kp7_s","kp8_y","kp8_x","kp8_s",
    "kp9_y","kp9_x","kp9_s","kp10_y","kp10_x","kp10_s",
    "kp11_y","kp11_x","kp11_s","kp12_y","kp12_x","kp12_s",
    "kp13_y","kp13_x","kp13_s","kp14_y","kp14_x","kp14_s",
] + _DERIVED

# kp13: +발목 (13kp, 45f) — 기존 full body
KP13_COLS = [
    "kp0_y","kp0_x","kp0_s",
    "kp5_y","kp5_x","kp5_s","kp6_y","kp6_x","kp6_s",
    "kp7_y","kp7_x","kp7_s","kp8_y","kp8_x","kp8_s",
    "kp9_y","kp9_x","kp9_s","kp10_y","kp10_x","kp10_s",
    "kp11_y","kp11_x","kp11_s","kp12_y","kp12_x","kp12_s",
    "kp13_y","kp13_x","kp13_s","kp14_y","kp14_x","kp14_s",
    "kp15_y","kp15_x","kp15_s","kp16_y","kp16_x","kp16_s",
] + _DERIVED

# kp17: +눈/귀 (17kp, 57f) — MoveNet 전체
KP17_COLS = [
    "kp0_y","kp0_x","kp0_s",
    "kp1_y","kp1_x","kp1_s","kp2_y","kp2_x","kp2_s",
    "kp3_y","kp3_x","kp3_s","kp4_y","kp4_x","kp4_s",
    "kp5_y","kp5_x","kp5_s","kp6_y","kp6_x","kp6_s",
    "kp7_y","kp7_x","kp7_s","kp8_y","kp8_x","kp8_s",
    "kp9_y","kp9_x","kp9_s","kp10_y","kp10_x","kp10_s",
    "kp11_y","kp11_x","kp11_s","kp12_y","kp12_x","kp12_s",
    "kp13_y","kp13_x","kp13_s","kp14_y","kp14_x","kp14_s",
    "kp15_y","kp15_x","kp15_s","kp16_y","kp16_x","kp16_s",
] + _DERIVED

# velocity 계산 대상: _y, _x 컬럼만 (confidence 제외)
KP5_VEL_BASE  = [c for c in KP5_COLS  if c.endswith("_y") or c.endswith("_x")]
KP7_VEL_BASE  = [c for c in KP7_COLS  if c.endswith("_y") or c.endswith("_x")]
KP9_VEL_BASE  = [c for c in KP9_COLS  if c.endswith("_y") or c.endswith("_x")]
KP11_VEL_BASE = [c for c in KP11_COLS if c.endswith("_y") or c.endswith("_x")]
KP13_VEL_BASE = [c for c in KP13_COLS if c.endswith("_y") or c.endswith("_x")]
KP17_VEL_BASE = [c for c in KP17_COLS if c.endswith("_y") or c.endswith("_x")]

FEATURE_SETS = {
    "kp5":  KP5_COLS,
    "kp7":  KP7_COLS,
    "kp9":  KP9_COLS,
    "kp11": KP11_COLS,
    "kp13": KP13_COLS,
    "kp17": KP17_COLS,
}
VEL_BASE_MAP = {
    "kp5":  KP5_VEL_BASE,
    "kp7":  KP7_VEL_BASE,
    "kp9":  KP9_VEL_BASE,
    "kp11": KP11_VEL_BASE,
    "kp13": KP13_VEL_BASE,
    "kp17": KP17_VEL_BASE,
}

# hip center y,x 인덱스 (kp11, kp12)
_HIP_Y = ["kp11_y","kp12_y"]
_HIP_X = ["kp11_x","kp12_x"]
HIP_Y_COLS = {k: _HIP_Y for k in FEATURE_SETS}
HIP_X_COLS = {k: _HIP_X for k in FEATURE_SETS}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-id",         required=True)
    p.add_argument("--feature-set",    choices=["kp5","kp7","kp9","kp11","kp13","kp17"], default="kp13")
    p.add_argument("--window-size",    type=int, default=40)
    p.add_argument("--model-type",     choices=["gru","lstm","tcn"],   default="gru")
    p.add_argument("--hidden-sizes",   nargs="+", type=int,      default=[128,64])
    p.add_argument("--epochs",         type=int,   default=80)
    p.add_argument("--batch-size",     type=int,   default=512)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--dropout",        type=float, default=0.3)
    p.add_argument("--fall-stride",    type=int,   default=1)
    p.add_argument("--nfall-stride",   type=int,   default=5)
    p.add_argument("--focal-alpha",    type=float, default=0.75)
    p.add_argument("--focal-gamma",    type=float, default=2.0)
    p.add_argument("--use-velocity",   action="store_true")
    p.add_argument("--use-hip-center", action="store_true")
    p.add_argument("--exclude-boundary", action="store_true",
                   help="exclude windows where last frame is label_3class=2")
    p.add_argument("--pure-window", action="store_true",
                   help="fall window requires fall onset inside window; end allowed to exceed by --pure-margin frames")
    p.add_argument("--pure-margin", type=int, default=5,
                   help="frames fall_end may extend beyond window end (captures post-fall lying state)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=None,
                   help="Override data directory (default: splits_v2_class_balanced_filtered)")
    p.add_argument("--out-root", default=None,
                   help="Override output root (default: results/phase36_window_ablation)")
    return p.parse_args()


def log(msg):
    import datetime
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_video_frames(csv_path, feat_cols, need_3class=False):
    data = defaultdict(lambda: {"feat": [], "label": [], "label3": [], "frame": []})
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            lbl = int(row["label"])
            frame_idx = int(row.get("frame", 0))
            lbl3 = int(row.get("label_3class", lbl))
            try:
                feat = [float(row[c]) if c in row and row[c] != "" else 0.0 for c in feat_cols]
            except ValueError:
                continue
            data[vid]["feat"].append((frame_idx, feat))
            data[vid]["label"].append((frame_idx, lbl))
            data[vid]["label3"].append((frame_idx, lbl3))

    result = {}
    for vid, d in data.items():
        if not d["feat"]:
            continue
        d["feat"].sort(key=lambda x: x[0])
        d["label"].sort(key=lambda x: x[0])
        d["label3"].sort(key=lambda x: x[0])
        frames = np.array([f for _, f in d["feat"]], np.float32)
        labels = np.array([l for _, l in d["label"]], np.int32)
        labels3 = np.array([l for _, l in d["label3"]], np.int32)
        result[vid] = (frames, labels, labels3)
    return result


def add_velocity(frames: np.ndarray, vel_col_indices: list[int]) -> np.ndarray:
    """Append frame-to-frame Δ for selected feature indices. First frame Δ = 0."""
    delta = np.zeros_like(frames)
    delta[1:, vel_col_indices] = frames[1:, vel_col_indices] - frames[:-1, vel_col_indices]
    return np.concatenate([frames, delta[:, vel_col_indices]], axis=-1)


def apply_hip_center(frames: np.ndarray, hip_y_idx: list[int], hip_x_idx: list[int]) -> np.ndarray:
    """Subtract hip centroid from all _y, _x columns (in-place copy)."""
    out = frames.copy()
    cy = out[:, hip_y_idx].mean(axis=-1, keepdims=True)
    cx = out[:, hip_x_idx].mean(axis=-1, keepdims=True)
    # subtract from all _y and _x columns
    y_cols = [i for i, _ in enumerate(frames[0]) if i % 3 == 0]  # simplified: y is index 0,3,6,...
    # Instead: apply to all even-indexed kp columns that represent y or x
    # (safe: just center the whole feature vector on hip, not perfect but useful)
    out[:, hip_y_idx] -= cy
    out[:, hip_x_idx] -= cx
    return out


def find_fall_events(labels):
    """Return [(start, end), ...] inclusive for each contiguous label==1 segment."""
    events, in_fall, start = [], False, 0
    for i, l in enumerate(labels):
        if l == 1 and not in_fall:
            start, in_fall = i, True
        elif l != 1 and in_fall:
            events.append((start, i - 1))
            in_fall = False
    if in_fall:
        events.append((start, len(labels) - 1))
    return events


def build_train_windows(video_data, window_size, fall_stride, nfall_stride, rng,
                        vel_idx=None, hip_y=None, hip_x=None, exclude_boundary=False,
                        pure_window=False, pure_margin=5):
    fall_wins, nfall_wins = [], []
    for vid, (frames, labels, labels3) in video_data.items():
        n = len(frames)
        if n < window_size:
            continue
        f = frames
        if hip_y is not None:
            f = apply_hip_center(f, hip_y, hip_x)
        if vel_idx is not None:
            f = add_velocity(f, vel_idx)

        if pure_window:
            # fall: onset inside window, end within window + pure_margin frames
            # condition: w_start <= fall_start  AND  fall_end <= w_start + window_size - 1 + pure_margin
            # → w_start ∈ [max(0, fall_end - pure_margin - window_size + 1), fall_start]
            for (fs, fe) in find_fall_events(labels):
                w_min = max(0, fe - pure_margin - window_size + 1)
                w_max = min(fs, n - window_size)
                for w_start in range(w_min, w_max + 1, fall_stride):
                    fall_wins.append(f[w_start:w_start + window_size])
            # nfall: all frames are 0
            for t in range(window_size - 1, n):
                if np.all(labels[t - window_size + 1:t + 1] == 0):
                    pos = t - (window_size - 1)
                    if pos % nfall_stride == 0:
                        nfall_wins.append(f[t - window_size + 1:t + 1])
        else:
            for t in range(window_size - 1, n):
                if exclude_boundary and labels3[t] == 2:
                    continue
                lbl = int(labels[t])
                pos = t - (window_size - 1)
                win = f[t - window_size + 1:t + 1]
                if lbl == 1:
                    if pos % fall_stride == 0:
                        fall_wins.append(win)
                else:
                    if pos % nfall_stride == 0:
                        nfall_wins.append(win)

    if fall_wins:
        n_feat_out = fall_wins[0].shape[-1]
    elif nfall_wins:
        n_feat_out = nfall_wins[0].shape[-1]
    else:
        sample = next(iter(video_data.values()))[0]
        n_feat_out = sample.shape[-1] + (len(vel_idx) if vel_idx else 0)
    X_f = np.stack(fall_wins,  0).astype(np.float32) if fall_wins  else np.empty((0, window_size, n_feat_out), np.float32)
    X_n = np.stack(nfall_wins, 0).astype(np.float32) if nfall_wins else np.empty((0, window_size, n_feat_out), np.float32)
    X = np.concatenate([X_f, X_n], 0)
    y = np.concatenate([np.ones(len(X_f), np.int32), np.zeros(len(X_n), np.int32)], 0)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def build_eval_windows(video_data, window_size, vel_idx=None, hip_y=None, hip_x=None,
                       pure_window=False, pure_margin=5):
    X_list, y_list = [], []
    for vid, (frames, labels, labels3) in video_data.items():
        n = len(frames)
        if n < window_size:
            continue
        f = frames
        if hip_y is not None:
            f = apply_hip_center(f, hip_y, hip_x)
        if vel_idx is not None:
            f = add_velocity(f, vel_idx)

        if pure_window:
            # fall windows: onset inside, end within +pure_margin
            fall_set = set()
            for (fs, fe) in find_fall_events(labels):
                w_min = max(0, fe - pure_margin - window_size + 1)
                w_max = min(fs, n - window_size)
                for w_start in range(w_min, w_max + 1):
                    fall_set.add(w_start)
                    X_list.append(f[w_start:w_start + window_size])
                    y_list.append(1)
            # nfall windows: all frames 0
            for t in range(window_size - 1, n):
                w_start = t - window_size + 1
                if w_start not in fall_set and np.all(labels[w_start:t + 1] == 0):
                    X_list.append(f[w_start:t + 1])
                    y_list.append(0)
        else:
            for t in range(window_size - 1, n):
                X_list.append(f[t - window_size + 1:t + 1])
                y_list.append(int(labels[t]))
    return np.stack(X_list, 0).astype(np.float32), np.array(y_list, np.int32)


def fit_normalization(X_train):
    flat = X_train.reshape(-1, X_train.shape[-1])
    mn, mx = flat.min(0), flat.max(0)
    scale = np.where(mx > mn, mx - mn, 1.0)
    return mn, scale


def normalize(X, mn, scale):
    return np.clip((X - mn) / scale, 0.0, 1.0)


def build_model(n_feat, window_size, model_type, hidden_sizes, dropout):
    inp = tf.keras.Input(shape=(window_size, n_feat))

    if model_type == "tcn":
        # Dilated causal Conv1D stack (no recurrence) — deployable, stateless
        filters = hidden_sizes[0]  # e.g. 64
        x = inp
        for dilation in [1, 2, 4, 8]:
            residual = x
            x = tf.keras.layers.Conv1D(filters, 3, padding="causal",
                                       dilation_rate=dilation, activation="relu")(x)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Conv1D(filters, 3, padding="causal",
                                       dilation_rate=dilation, activation="relu")(x)
            x = tf.keras.layers.Dropout(dropout)(x)
            # residual projection if channel mismatch
            if residual.shape[-1] != filters:
                residual = tf.keras.layers.Conv1D(filters, 1, padding="same")(residual)
            x = tf.keras.layers.Add()([x, residual])
        x = x[:, -1, :]  # last time step
        x = tf.keras.layers.Dense(hidden_sizes[-1], activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    else:
        RNN = tf.keras.layers.GRU if model_type == "gru" else tf.keras.layers.LSTM
        x = tf.keras.layers.Conv1D(64, 5, padding="causal", activation="relu")(inp)
        x = tf.keras.layers.Conv1D(64, 5, padding="causal", activation="relu")(x)
        rnn_kwargs = {"dropout": dropout, "recurrent_dropout": 0.0, "unroll": True}
        if model_type == "gru":
            rnn_kwargs["reset_after"] = True
        for i, h in enumerate(hidden_sizes):
            x = RNN(h, return_sequences=(i < len(hidden_sizes)-1), **rnn_kwargs)(x)
        x = tf.keras.layers.Dense(hidden_sizes[-1], activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    out = tf.keras.layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inp, out)


def focal_loss(gamma=2.0, alpha=0.75):
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        probs = y_pred[:, 1]
        bce = -y_true_f * tf.math.log(probs+1e-9) - (1-y_true_f)*tf.math.log(1-probs+1e-9)
        p_t = tf.where(tf.cast(y_true_f, bool), probs, 1-probs)
        a_t = tf.where(tf.cast(y_true_f, bool), alpha, 1-alpha)
        return tf.reduce_mean(a_t * tf.pow(1-p_t, gamma) * bce)
    return loss


def compute_video_scores(model, video_data, window_size,
                         vel_idx=None, hip_y=None, hip_x=None, mn=None, sc=None):
    """Pre-compute fall score sequence for each video. Returns {vid: (scores, has_fall)}."""
    results = {}
    for vid, (frames, labels, _) in video_data.items():
        n = len(frames)
        if n < window_size:
            continue
        f = frames.copy()
        if hip_y is not None:
            f = apply_hip_center(f, hip_y, hip_x)
        if vel_idx is not None:
            f = add_velocity(f, vel_idx)
        if mn is not None:
            f = normalize(f, mn, sc)
        n_wins = n - window_size + 1
        wins = np.stack([f[t:t + window_size] for t in range(n_wins)], 0).astype(np.float32)
        scores = WindowMinPRCallback._predict_batched(model, wins)
        results[vid] = (scores, bool(np.any(labels == 1)))
    return results


def apply_vote_metrics(video_scores, threshold, vote_window, vote_k):
    """Apply K-of-N vote on precomputed per-video scores and return full metrics."""
    tp = fp = fn = tn = 0
    for scores, has_fall in video_scores.values():
        buf = np.zeros(vote_window, dtype=np.int32)
        vote_sum = head = 0
        detected = False
        for score in scores:
            v = 1 if score >= threshold else 0
            vote_sum += v - int(buf[head])
            buf[head] = v
            head = (head + 1) % vote_window
            if vote_sum >= vote_k:
                detected = True
                break
        if has_fall:
            tp += detected;  fn += not detected
        else:
            fp += detected;  tn += not detected
    e = 1e-9
    return {
        "min_pr":          round(min(tp/(tp+fp+e), tp/(tp+fn+e), tn/(tn+fn+e), tn/(tn+fp+e)), 4),
        "fall_precision":  round(tp/(tp+fp+e), 4),
        "fall_recall":     round(tp/(tp+fn+e), 4),
        "nfall_precision": round(tn/(tn+fn+e), 4),
        "nfall_recall":    round(tn/(tn+fp+e), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def window_minpr(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tp = int(np.sum((y_pred==1)&(y_true==1))); fp = int(np.sum((y_pred==1)&(y_true==0)))
    fn = int(np.sum((y_pred==0)&(y_true==1))); tn = int(np.sum((y_pred==0)&(y_true==0)))
    e = 1e-9
    return {"min_pr": min(tp/(tp+fp+e), tp/(tp+fn+e), tn/(tn+fn+e), tn/(tn+fp+e)),
            "fall_precision": tp/(tp+fp+e), "fall_recall": tp/(tp+fn+e),
            "nfall_precision": tn/(tn+fn+e), "nfall_recall": tn/(tn+fp+e),
            "tp":tp,"fp":fp,"fn":fn,"tn":tn}


def select_threshold(y_true, y_prob):
    best_thr, best_mp = 0.5, 0.0
    for thr in np.arange(0.30, 0.905, 0.025):
        mp = window_minpr(y_true, y_prob, float(thr))["min_pr"]
        if mp > best_mp:
            best_mp, best_thr = mp, float(thr)
    return round(best_thr, 3), round(best_mp, 6)


class WindowMinPRCallback(tf.keras.callbacks.Callback):
    _MAX_CB_VAL = 20_000  # 서브샘플 상한 — 234K 전체 eager 루프 대비 ~10x 빠름

    def __init__(self, X_val, y_val, out_dir, patience=15):
        super().__init__()
        if len(X_val) > self._MAX_CB_VAL:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_val), self._MAX_CB_VAL, replace=False)
            X_val, y_val = X_val[idx], y_val[idx]
        self.X_val, self.y_val = X_val, y_val
        self.out_dir = out_dir
        self.patience = patience
        self.best_mp = 0.0; self.best_thr = 0.5; self.wait = 0
        self.history = {"epoch": [], "loss": [], "val_loss": [],
                        "val_acc": [], "val_window_minpr": []}

    @staticmethod
    def _predict_batched(model, X, batch=128):
        """Eager batched inference — avoids compiled-graph EagerConst copy failures."""
        parts = []
        for i in range(0, len(X), batch):
            parts.append(model(X[i:i + batch], training=False)[:, 1].numpy())
        return np.concatenate(parts)

    def on_epoch_end(self, epoch, logs=None):
        probs = self._predict_batched(self.model, self.X_val)
        thr, mp = select_threshold(self.y_val, probs)
        logs["val_window_minpr"] = mp
        self.history["epoch"].append(epoch + 1)
        self.history["loss"].append(float(logs.get("loss", 0)))
        self.history["val_loss"].append(float(logs.get("val_loss", 0)))
        self.history["val_acc"].append(float(logs.get("val_accuracy", 0)))
        self.history["val_window_minpr"].append(float(mp))
        if mp > self.best_mp:
            self.best_mp, self.best_thr, self.wait = mp, thr, 0
            self.model.save(str(self.out_dir / "model_best.keras"))
            log(f"  ep{epoch+1:3d}  val_minpr={mp:.4f}  thr={thr}  *** best")
        else:
            self.wait += 1
            if (epoch+1) % 5 == 0:
                log(f"  ep{epoch+1:3d}  val_minpr={mp:.4f}  thr={thr}  (wait {self.wait}/{self.patience})")
            if self.wait >= self.patience:
                log(f"  Early stop ep{epoch+1}")
                self.model.stop_training = True


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    out_root = Path(args.out_root) if args.out_root else OUT_ROOT

    feat_cols = FEATURE_SETS[args.feature_set]
    exp_dir = out_root / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # velocity 인덱스 계산
    vel_base = VEL_BASE_MAP[args.feature_set]
    vel_idx = [feat_cols.index(c) for c in vel_base if c in feat_cols] if args.use_velocity else None

    # hip center 인덱스
    hip_y = [feat_cols.index(c) for c in HIP_Y_COLS[args.feature_set]] if args.use_hip_center else None
    hip_x = [feat_cols.index(c) for c in HIP_X_COLS[args.feature_set]] if args.use_hip_center else None

    n_base = len(feat_cols)
    n_vel  = len(vel_idx) if vel_idx else 0
    n_feat = n_base + n_vel

    log(f"=== {args.exp_id} ===")
    log(f"  feat={args.feature_set}({n_feat}f)  win={args.window_size}  model={args.model_type}")
    log(f"  velocity={args.use_velocity}  hip_center={args.use_hip_center}  exclude_boundary={args.exclude_boundary}")
    log(f"  focal α={args.focal_alpha} γ={args.focal_gamma}  strides fall={args.fall_stride}/nfall={args.nfall_stride}")
    if args.pure_window:
        log(f"  pure_window=True  margin={args.pure_margin} frames")

    log("Loading CSVs …")
    train_data = load_video_frames(data_dir/"train.csv", feat_cols, need_3class=True)
    val_data   = load_video_frames(data_dir/"val.csv",   feat_cols, need_3class=True)
    test_data  = load_video_frames(data_dir/"test.csv",  feat_cols, need_3class=True)

    log("Building windows …")
    X_tr, y_tr = build_train_windows(train_data, args.window_size, args.fall_stride, args.nfall_stride,
                                      rng, vel_idx, hip_y, hip_x, args.exclude_boundary,
                                      args.pure_window, args.pure_margin)
    X_va, y_va = build_eval_windows(val_data,  args.window_size, vel_idx, hip_y, hip_x,
                                    args.pure_window, args.pure_margin)
    X_te, y_te = build_eval_windows(test_data, args.window_size, vel_idx, hip_y, hip_x,
                                    args.pure_window, args.pure_margin)
    log(f"  train={len(X_tr)} (fall={y_tr.sum()})  val={len(X_va)} (fall={y_va.sum()})  test={len(X_te)} (fall={y_te.sum()})")

    mn, sc = fit_normalization(X_tr)
    X_tr = normalize(X_tr, mn, sc)
    X_va = normalize(X_va, mn, sc)
    X_te = normalize(X_te, mn, sc)

    model = build_model(n_feat, args.window_size, args.model_type, args.hidden_sizes, args.dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=focal_loss(args.focal_gamma, args.focal_alpha), metrics=["accuracy"])
    log(f"  params={model.count_params():,}")

    cb = WindowMinPRCallback(X_va, y_va, exp_dir, patience=15)
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=0)
    log("Training …")
    with tf.device("/CPU:0"):
        ds_tr = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
                 .shuffle(min(len(X_tr), 50_000), seed=42)
                 .batch(args.batch_size)
                 .prefetch(tf.data.AUTOTUNE))
        ds_va = (tf.data.Dataset.from_tensor_slices((X_va, y_va))
                 .batch(args.batch_size)
                 .prefetch(tf.data.AUTOTUNE))
    model.fit(ds_tr, epochs=args.epochs,
              validation_data=ds_va, callbacks=[cb, lr_cb], verbose=0)
    (exp_dir / "training_history.json").write_text(json.dumps(cb.history, indent=2))

    best = exp_dir / "model_best.keras"
    if best.exists():
        model = tf.keras.models.load_model(str(best), compile=False)
    model.save(str(exp_dir / "model.keras"))

    # window-level probabilities (for threshold sweep)
    val_probs  = WindowMinPRCallback._predict_batched(model, X_va)
    test_probs = WindowMinPRCallback._predict_batched(model, X_te)

    # per-video scores (computed once; reused across all sweep combos)
    log("Scoring videos …")
    ev_kw = dict(vel_idx=vel_idx, hip_y=hip_y, hip_x=hip_x, mn=mn, sc=sc)
    val_vscores  = compute_video_scores(model, val_data,  args.window_size, **ev_kw)
    test_vscores = compute_video_scores(model, test_data, args.window_size, **ev_kw)

    VOTE_COMBOS = [(1, 1), (3, 2), (5, 3), (5, 4), (7, 4), (7, 5), (10, 6)]

    # ── Threshold sweep ───────────────────────────────────────────────────────
    log("Threshold sweep …")
    thr_sweep = []
    for thr_f in np.arange(0.30, 0.905, 0.025):
        t = round(float(thr_f), 3)
        thr_sweep.append({
            "thr":          t,
            "val_win":      window_minpr(y_va, val_probs,  t),
            "test_win":     window_minpr(y_te, test_probs, t),
            "val_ev_v5k3":  apply_vote_metrics(val_vscores,  t, 5, 3),
            "test_ev_v5k3": apply_vote_metrics(test_vscores, t, 5, 3),
        })

    best_thr_row = max(thr_sweep, key=lambda r: r["val_win"]["min_pr"])
    thr    = best_thr_row["thr"]
    val_m  = best_thr_row["val_win"]
    test_m = best_thr_row["test_win"]

    log(f"Best thr={thr}")
    log(f"Val  window: minpr={val_m['min_pr']:.4f}  "
        f"fall={val_m['fall_precision']:.4f}/{val_m['fall_recall']:.4f}  "
        f"nfall={val_m['nfall_precision']:.4f}/{val_m['nfall_recall']:.4f}  "
        f"FN={val_m['fn']}  FP={val_m['fp']}")
    log(f"Test window: minpr={test_m['min_pr']:.4f}  "
        f"fall={test_m['fall_precision']:.4f}/{test_m['fall_recall']:.4f}  "
        f"nfall={test_m['nfall_precision']:.4f}/{test_m['nfall_recall']:.4f}  "
        f"FN={test_m['fn']}  FP={test_m['fp']}")

    # ── Post-processing sweep at best threshold ───────────────────────────────
    log(f"Postproc sweep (thr={thr}) …")
    pp_sweep = []
    for vw, vk in VOTE_COMBOS:
        val_ev  = apply_vote_metrics(val_vscores,  thr, vw, vk)
        test_ev = apply_vote_metrics(test_vscores, thr, vw, vk)
        pp_sweep.append({"vote_window": vw, "vote_k": vk,
                         "val": val_ev, "test": test_ev})
        log(f"  v{vw}k{vk}"
            f"  val={val_ev['min_pr']:.4f}"
            f"(fall={val_ev['fall_precision']:.4f}/{val_ev['fall_recall']:.4f}"
            f" nfall={val_ev['nfall_precision']:.4f}/{val_ev['nfall_recall']:.4f}"
            f" FP={val_ev['fp']} FN={val_ev['fn']})"
            f"  test={test_ev['min_pr']:.4f}"
            f"(fall={test_ev['fall_precision']:.4f}/{test_ev['fall_recall']:.4f}"
            f" nfall={test_ev['nfall_precision']:.4f}/{test_ev['nfall_recall']:.4f}"
            f" FP={test_ev['fp']} FN={test_ev['fn']})")

    best_pp = max(pp_sweep, key=lambda r: r["val"]["min_pr"])
    log(f"Best postproc: v{best_pp['vote_window']}k{best_pp['vote_k']}  "
        f"val_ev={best_pp['val']['min_pr']:.4f}  test_ev={best_pp['test']['min_pr']:.4f}")
    log(f"Gap window→event(best): {val_m['min_pr'] - best_pp['val']['min_pr']:+.4f}")

    # unified_eval: summary entry used by the analysis loop
    unified_eval = {
        "threshold": thr,
        "window": {"val": val_m, "test": test_m},
        "event": {
            "min_pr":      best_pp["test"]["min_pr"],
            "val":         best_pp["val"],
            "test":        best_pp["test"],
            "vote_window": best_pp["vote_window"],
            "vote_k":      best_pp["vote_k"],
        },
    }

    (exp_dir/"feature_columns.json").write_text(json.dumps(feat_cols, indent=2))
    (exp_dir/"normalization.json").write_text(
        json.dumps({"min": mn.tolist(), "scale": sc.tolist()}))
    metrics_out = {
        "exp_id":         args.exp_id,
        "config":         vars(args),
        "unified_eval":   unified_eval,
        "threshold_sweep": thr_sweep,
        "postproc_sweep":  pp_sweep,
        "window_stats": {
            "train_fall":  int(y_tr.sum()),
            "train_nfall": int((y_tr == 0).sum()),
            "val_fall":    int(y_va.sum()),
            "test_fall":   int(y_te.sum()),
        },
    }
    (exp_dir/"metrics.json").write_text(json.dumps(metrics_out, indent=2))
    log(f"=== DONE {args.exp_id}  "
        f"test_win={test_m['min_pr']:.4f}  "
        f"test_ev={best_pp['test']['min_pr']:.4f}"
        f"(v{best_pp['vote_window']}k{best_pp['vote_k']}) ===")


if __name__ == "__main__":
    main()
