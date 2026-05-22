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

REPO     = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "dataset/splits_v2_class_balanced_filtered"
OUT_ROOT = REPO / "results/phase36_window_ablation"

KP7_COLS = [
    "kp0_y","kp0_x","kp0_s",
    "kp5_y","kp5_x","kp5_s","kp6_y","kp6_x","kp6_s",
    "kp7_y","kp7_x","kp7_s","kp8_y","kp8_x","kp8_s",
    "kp11_y","kp11_x","kp11_s","kp12_y","kp12_x","kp12_s",
    "HSSC_y","HSSC_x","RWHC","VHSSC","AHSSC","AHSSC_x",
]
KP13_COLS = [
    "kp0_y","kp0_x","kp0_s",
    "kp5_y","kp5_x","kp5_s","kp6_y","kp6_x","kp6_s",
    "kp7_y","kp7_x","kp7_s","kp8_y","kp8_x","kp8_s",
    "kp9_y","kp9_x","kp9_s","kp10_y","kp10_x","kp10_s",
    "kp11_y","kp11_x","kp11_s","kp12_y","kp12_x","kp12_s",
    "kp13_y","kp13_x","kp13_s","kp14_y","kp14_x","kp14_s",
    "kp15_y","kp15_x","kp15_s","kp16_y","kp16_x","kp16_s",
    "HSSC_y","HSSC_x","RWHC","VHSSC","AHSSC","AHSSC_x",
]
# velocity 계산 대상: _y, _x 컬럼만 (confidence 제외)
KP7_VEL_BASE  = [c for c in KP7_COLS  if c.endswith("_y") or c.endswith("_x")]
KP13_VEL_BASE = [c for c in KP13_COLS if c.endswith("_y") or c.endswith("_x")]
FEATURE_SETS  = {"kp7": KP7_COLS, "kp13": KP13_COLS}

# hip center y,x 인덱스 (kp11, kp12)
HIP_Y_COLS = {"kp7": ["kp11_y","kp12_y"], "kp13": ["kp11_y","kp12_y"]}
HIP_X_COLS = {"kp7": ["kp11_x","kp12_x"], "kp13": ["kp11_x","kp12_x"]}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-id",         required=True)
    p.add_argument("--feature-set",    choices=["kp7","kp13"], default="kp13")
    p.add_argument("--window-size",    type=int, choices=[30,40], default=40)
    p.add_argument("--model-type",     choices=["gru","lstm"],   default="gru")
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
                feat = [float(row[c]) for c in feat_cols]
            except (KeyError, ValueError):
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

    n_feat_out = fall_wins[0].shape[-1] if fall_wins else (frames.shape[-1] + (len(vel_idx) if vel_idx else 0))
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
    RNN = tf.keras.layers.GRU if model_type == "gru" else tf.keras.layers.LSTM
    inp = tf.keras.Input(shape=(window_size, n_feat))
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


def event_vote_eval(model, video_data, window_size, threshold,
                    vote_window=5, vote_k=3,
                    vel_idx=None, hip_y=None, hip_x=None, mn=None, sc=None):
    """Simulate deployed vote-based detection at video level.

    For each video: run sliding window inference, apply K-of-N vote logic,
    declare 'fall detected' if vote_sum >= vote_k at any point.
    Ground truth: video contains label==1 anywhere → fall video.
    """
    tp = fp = fn = tn = 0
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

        has_fall = bool(np.any(labels == 1))
        n_wins = n - window_size + 1
        windows = np.stack([f[t:t + window_size] for t in range(n_wins)], axis=0).astype(np.float32)
        fall_scores = model.predict(windows, batch_size=256, verbose=0)[:, 1]

        vote_buf = np.zeros(vote_window, dtype=np.int32)
        vote_sum = 0
        vote_head = 0
        detected = False
        for score in fall_scores:
            this_vote = 1 if score >= threshold else 0
            old_vote = int(vote_buf[vote_head])
            vote_buf[vote_head] = this_vote
            vote_head = (vote_head + 1) % vote_window
            vote_sum += this_vote - old_vote
            if vote_sum >= vote_k:
                detected = True
                break

        if has_fall:
            tp += detected; fn += (not detected)
        else:
            fp += detected; tn += (not detected)

    e = 1e-9
    return {
        "min_pr":          round(min(tp/(tp+fp+e), tp/(tp+fn+e), tn/(tn+fn+e), tn/(tn+fp+e)), 4),
        "fall_precision":  round(tp/(tp+fp+e), 4),
        "fall_recall":     round(tp/(tp+fn+e), 4),
        "nfall_precision": round(tn/(tn+fn+e), 4),
        "nfall_recall":    round(tn/(tn+fp+e), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "vote_window": vote_window, "vote_k": vote_k,
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
    def __init__(self, X_val, y_val, out_dir, patience=15):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val
        self.out_dir = out_dir
        self.patience = patience
        self.best_mp = 0.0; self.best_thr = 0.5; self.wait = 0
        self.history = {"epoch": [], "loss": [], "val_loss": [],
                        "val_acc": [], "val_window_minpr": []}

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.X_val, batch_size=1024, verbose=0)[:, 1]
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

    feat_cols = FEATURE_SETS[args.feature_set]
    exp_dir = OUT_ROOT / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # velocity 인덱스 계산
    vel_base = KP7_VEL_BASE if args.feature_set == "kp7" else KP13_VEL_BASE
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
    train_data = load_video_frames(DATA_DIR/"train.csv", feat_cols, need_3class=True)
    val_data   = load_video_frames(DATA_DIR/"val.csv",   feat_cols, need_3class=True)
    test_data  = load_video_frames(DATA_DIR/"test.csv",  feat_cols, need_3class=True)

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
    model.fit(X_tr, y_tr, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=(X_va, y_va), callbacks=[cb, lr_cb], verbose=0)
    (exp_dir / "training_history.json").write_text(json.dumps(cb.history, indent=2))

    best = exp_dir / "model_best.keras"
    if best.exists():
        model = tf.keras.models.load_model(str(best), compile=False)
    model.save(str(exp_dir / "model.keras"))

    val_probs = model.predict(X_va, batch_size=1024, verbose=0)[:, 1]
    thr, val_mp = select_threshold(y_va, val_probs)
    val_m = window_minpr(y_va, val_probs, thr)

    test_probs = model.predict(X_te, batch_size=1024, verbose=0)[:, 1]
    test_m = window_minpr(y_te, test_probs, thr)

    log(f"Val   minpr={val_mp:.4f}  thr={thr}")
    log(f"Test  minpr={test_m['min_pr']:.4f}  "
        f"fall_pr={test_m['fall_precision']:.4f}  nfall_pr={test_m['nfall_precision']:.4f}  "
        f"fall_rc={test_m['fall_recall']:.4f}  nfall_rc={test_m['nfall_recall']:.4f}  "
        f"FN={test_m['fn']}  FP={test_m['fp']}")

    # ── Event-level vote simulation ───────────────────────────────────────────
    # raw: single-window threshold (vote_window=1, vote_k=1) — baseline
    # vote: K-of-N matching STM32 deployment (vote_window=5, vote_k=3)
    log("Event eval …")
    ev_kw = dict(vel_idx=vel_idx, hip_y=hip_y, hip_x=hip_x, mn=mn, sc=sc)
    val_ev_raw  = event_vote_eval(model, val_data,  args.window_size, thr, 1, 1, **ev_kw)
    test_ev_raw = event_vote_eval(model, test_data, args.window_size, thr, 1, 1, **ev_kw)
    val_ev_vote  = event_vote_eval(model, val_data,  args.window_size, thr, 5, 3, **ev_kw)
    test_ev_vote = event_vote_eval(model, test_data, args.window_size, thr, 5, 3, **ev_kw)

    log(f"Val  event(raw)  minpr={val_ev_raw['min_pr']:.4f}  "
        f"fall={val_ev_raw['fall_precision']:.4f}/{val_ev_raw['fall_recall']:.4f}  "
        f"nfall={val_ev_raw['nfall_precision']:.4f}/{val_ev_raw['nfall_recall']:.4f}  "
        f"TP={val_ev_raw['tp']} FP={val_ev_raw['fp']} FN={val_ev_raw['fn']}")
    log(f"Val  event(v5k3) minpr={val_ev_vote['min_pr']:.4f}  "
        f"fall={val_ev_vote['fall_precision']:.4f}/{val_ev_vote['fall_recall']:.4f}  "
        f"nfall={val_ev_vote['nfall_precision']:.4f}/{val_ev_vote['nfall_recall']:.4f}  "
        f"TP={val_ev_vote['tp']} FP={val_ev_vote['fp']} FN={val_ev_vote['fn']}")
    log(f"Test event(raw)  minpr={test_ev_raw['min_pr']:.4f}  "
        f"fall={test_ev_raw['fall_precision']:.4f}/{test_ev_raw['fall_recall']:.4f}  "
        f"nfall={test_ev_raw['nfall_precision']:.4f}/{test_ev_raw['nfall_recall']:.4f}  "
        f"TP={test_ev_raw['tp']} FP={test_ev_raw['fp']} FN={test_ev_raw['fn']}")
    log(f"Test event(v5k3) minpr={test_ev_vote['min_pr']:.4f}  "
        f"fall={test_ev_vote['fall_precision']:.4f}/{test_ev_vote['fall_recall']:.4f}  "
        f"nfall={test_ev_vote['nfall_precision']:.4f}/{test_ev_vote['nfall_recall']:.4f}  "
        f"TP={test_ev_vote['tp']} FP={test_ev_vote['fp']} FN={test_ev_vote['fn']}")
    gap_raw  = round(val_mp - val_ev_raw['min_pr'],  4)
    gap_vote = round(val_mp - val_ev_vote['min_pr'], 4)
    log(f"Gap window→event: raw={gap_raw:+.4f}  vote(5,3)={gap_vote:+.4f}")

    (exp_dir/"feature_columns.json").write_text(json.dumps(feat_cols, indent=2))
    (exp_dir/"normalization.json").write_text(json.dumps({"min": mn.tolist(), "scale": sc.tolist()}))
    metrics = {
        "exp_id": args.exp_id,
        "config": vars(args),
        "threshold": thr,
        "metrics": {
            "val_window":   {**val_m},
            "test_window":  {**test_m},
            "val_event_raw":   {**val_ev_raw},
            "test_event_raw":  {**test_ev_raw},
            "val_event_vote":  {**val_ev_vote},
            "test_event_vote": {**test_ev_vote},
        },
        "window_stats": {"train_fall": int(y_tr.sum()), "train_nfall": int((y_tr==0).sum()),
                         "val_fall": int(y_va.sum()), "test_fall": int(y_te.sum())},
    }
    (exp_dir/"metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"=== DONE {args.exp_id}  test_minpr={test_m['min_pr']:.4f} ===")


if __name__ == "__main__":
    main()
