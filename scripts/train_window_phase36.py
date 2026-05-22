#!/usr/bin/env python3
"""Phase 36: Window-level GRU/LSTM training.

Train & evaluate at WINDOW level (not event/video level).
This directly aligns training objective with embedded deployment:
  GRU maintains state frame-by-frame → output at frame T reflects last W frames.
  Window label = label of the LAST frame in the window.

Ablation variables (run in order):
  1. feature_set : kp7 (27-feat) vs kp13 (45-feat, adds wrists/knees/ankles)
  2. window_size : 40 vs 30
  3. model_type  : gru vs lstm

Usage:
  cd /path/to/Falling-Model-Development
  uv run python scripts/train_window_phase36.py \\
      --exp-id P36-kp7-w40-gru \\
      --feature-set kp7 --window-size 40 --model-type gru
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

# ── paths ─────────────────────────────────────────────────────────────────────
REPO     = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "dataset/splits_v2_class_balanced_filtered"
OUT_ROOT = REPO / "results/phase36_window_ablation"

# ── feature sets ──────────────────────────────────────────────────────────────
KP7_COLS = [
    "kp0_y","kp0_x","kp0_s",
    "kp5_y","kp5_x","kp5_s",
    "kp6_y","kp6_x","kp6_s",
    "kp7_y","kp7_x","kp7_s",
    "kp8_y","kp8_x","kp8_s",
    "kp11_y","kp11_x","kp11_s",
    "kp12_y","kp12_x","kp12_s",
    "HSSC_y","HSSC_x","RWHC","VHSSC","AHSSC","AHSSC_x",
]  # 27 features — current production config

KP13_COLS = [
    "kp0_y","kp0_x","kp0_s",
    "kp5_y","kp5_x","kp5_s",
    "kp6_y","kp6_x","kp6_s",
    "kp7_y","kp7_x","kp7_s",
    "kp8_y","kp8_x","kp8_s",
    "kp9_y","kp9_x","kp9_s",    # wrists (new)
    "kp10_y","kp10_x","kp10_s",
    "kp11_y","kp11_x","kp11_s",
    "kp12_y","kp12_x","kp12_s",
    "kp13_y","kp13_x","kp13_s", # knees (new)
    "kp14_y","kp14_x","kp14_s",
    "kp15_y","kp15_x","kp15_s", # ankles (new)
    "kp16_y","kp16_x","kp16_s",
    "HSSC_y","HSSC_x","RWHC","VHSSC","AHSSC","AHSSC_x",
]  # 45 features

FEATURE_SETS = {"kp7": KP7_COLS, "kp13": KP13_COLS}


# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-id",       required=True,  help="e.g. P36-kp7-w40-gru")
    p.add_argument("--feature-set",  choices=["kp7","kp13"], default="kp7")
    p.add_argument("--window-size",  type=int, choices=[30,40], default=40)
    p.add_argument("--model-type",   choices=["gru","lstm"],   default="gru")
    p.add_argument("--hidden-sizes", nargs="+", type=int,      default=[128, 64])
    p.add_argument("--epochs",       type=int, default=80)
    p.add_argument("--batch-size",   type=int, default=512)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--fall-stride",  type=int,  default=1,
                   help="sliding stride for fall windows during training")
    p.add_argument("--nfall-stride", type=int,  default=5,
                   help="sliding stride for non-fall windows during training")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


# ── logging ───────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    import datetime
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ── data loading ──────────────────────────────────────────────────────────────
def load_video_frames(csv_path: Path, feat_cols: list[str]):
    """Return dict[video_id] -> (frames: float32 array, labels: int32 array)."""
    data: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    missing_cols: set[str] = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            lbl = int(row["label"])
            frame_idx = int(row.get("frame", 0))
            try:
                feat = [float(row[c]) for c in feat_cols]
            except KeyError as e:
                missing_cols.add(str(e))
                continue
            except ValueError:
                continue
            data[vid][0].append((frame_idx, feat))
            data[vid][1].append((frame_idx, lbl))

    if missing_cols:
        print(f"WARNING: missing columns: {missing_cols}", file=sys.stderr)

    result = {}
    for vid, (frame_feat, frame_lbl) in data.items():
        if not frame_feat:
            continue
        frame_feat.sort(key=lambda x: x[0])
        frame_lbl.sort(key=lambda x: x[0])
        frames = np.array([f for _, f in frame_feat], dtype=np.float32)
        labels = np.array([l for _, l in frame_lbl], dtype=np.int32)
        result[vid] = (frames, labels)
    return result


# ── windowing ─────────────────────────────────────────────────────────────────
def build_train_windows(
    video_data: dict,
    window_size: int,
    fall_stride: int,
    nfall_stride: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows, last-frame label, strided sampling for balance."""
    fall_wins, nfall_wins = [], []

    for vid, (frames, labels) in video_data.items():
        n = len(frames)
        if n < window_size:
            continue
        for t in range(window_size - 1, n):
            lbl = int(labels[t])
            pos = t - (window_size - 1)
            if lbl == 1:
                if pos % fall_stride == 0:
                    fall_wins.append(frames[t - window_size + 1 : t + 1])
            else:
                if pos % nfall_stride == 0:
                    nfall_wins.append(frames[t - window_size + 1 : t + 1])

    X_f = np.stack(fall_wins,  axis=0).astype(np.float32) if fall_wins  else np.empty((0, window_size, frames.shape[1]), np.float32)
    X_n = np.stack(nfall_wins, axis=0).astype(np.float32) if nfall_wins else np.empty((0, window_size, frames.shape[1]), np.float32)

    X = np.concatenate([X_f, X_n], axis=0)
    y = np.concatenate([np.ones(len(X_f), np.int32), np.zeros(len(X_n), np.int32)], axis=0)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def build_eval_windows(
    video_data: dict,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """All windows (stride=1) for evaluation — directly reflects deployment."""
    X_list, y_list = [], []
    for vid, (frames, labels) in video_data.items():
        n = len(frames)
        if n < window_size:
            continue
        for t in range(window_size - 1, n):
            X_list.append(frames[t - window_size + 1 : t + 1])
            y_list.append(int(labels[t]))
    return (np.stack(X_list, axis=0).astype(np.float32),
            np.array(y_list, np.int32))


# ── normalization ─────────────────────────────────────────────────────────────
def fit_normalization(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min-max per feature, fitted on flattened training windows."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    mn   = flat.min(axis=0)
    mx   = flat.max(axis=0)
    scale = np.where(mx > mn, mx - mn, 1.0)
    return mn, scale


def normalize(X: np.ndarray, mn: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((X - mn) / scale, 0.0, 1.0)


# ── model ─────────────────────────────────────────────────────────────────────
def build_model(
    n_feat: int,
    window_size: int,
    model_type: str,
    hidden_sizes: list[int],
    dropout: float,
) -> tf.keras.Model:
    RNN = tf.keras.layers.GRU if model_type == "gru" else tf.keras.layers.LSTM

    inp = tf.keras.Input(shape=(window_size, n_feat), name="pose_window")
    x = inp
    x = tf.keras.layers.Conv1D(64, 5, padding="causal", activation="relu", name="conv1")(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="causal", activation="relu", name="conv2")(x)

    rnn_kwargs = {"unroll": True, "dropout": dropout, "recurrent_dropout": 0.0}
    if model_type == "gru":
        rnn_kwargs["reset_after"] = True

    for i, h in enumerate(hidden_sizes):
        return_seq = (i < len(hidden_sizes) - 1)
        x = RNN(
            h,
            return_sequences=return_seq,
            name=f"{model_type}_{i+1}",
            **rnn_kwargs,
        )(x)

    x = tf.keras.layers.Dense(hidden_sizes[-1], activation="relu", name="head")(x)
    x = tf.keras.layers.Dropout(dropout, name="drop")(x)
    out = tf.keras.layers.Dense(2, activation="softmax", name="prob")(x)
    return tf.keras.Model(inp, out)


# ── metrics ───────────────────────────────────────────────────────────────────
def window_minpr(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    eps = 1e-9
    fall_pr  = tp / (tp + fp + eps)
    fall_rc  = tp / (tp + fn + eps)
    nfall_pr = tn / (tn + fn + eps)
    nfall_rc = tn / (tn + fp + eps)
    return {
        "min_pr":          min(fall_pr, fall_rc, nfall_pr, nfall_rc),
        "fall_precision":  fall_pr,
        "fall_recall":     fall_rc,
        "nfall_precision": nfall_pr,
        "nfall_recall":    nfall_rc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_thr, best_mp = 0.5, 0.0
    for thr in np.arange(0.30, 0.905, 0.025):
        mp = window_minpr(y_true, y_prob, float(thr))["min_pr"]
        if mp > best_mp:
            best_mp, best_thr = mp, float(thr)
    return round(best_thr, 3), round(best_mp, 6)


# ── focal loss ────────────────────────────────────────────────────────────────
def focal_loss(gamma: float = 2.0, alpha: float = 0.75):
    """Focal loss weighted toward the fall class (alpha > 0.5 = more weight on fall)."""
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        probs = y_pred[:, 1]
        bce = -y_true_f * tf.math.log(probs + 1e-9) \
              - (1 - y_true_f) * tf.math.log(1 - probs + 1e-9)
        p_t   = tf.where(tf.cast(y_true_f, bool), probs, 1 - probs)
        alpha_t = tf.where(tf.cast(y_true_f, bool), alpha, 1 - alpha)
        fl = alpha_t * tf.pow(1 - p_t, gamma) * bce
        return tf.reduce_mean(fl)
    return loss


# ── window-level MinPR Keras callback ─────────────────────────────────────────
class WindowMinPRCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, out_dir: Path, patience: int = 12):
        super().__init__()
        self.X_val   = X_val
        self.y_val   = y_val
        self.out_dir = out_dir
        self.patience = patience
        self.best_mp  = 0.0
        self.best_thr = 0.5
        self.wait     = 0

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.X_val, batch_size=1024, verbose=0)[:, 1]
        thr, mp = select_threshold(self.y_val, probs)
        logs["val_window_minpr"] = mp
        if mp > self.best_mp:
            self.best_mp  = mp
            self.best_thr = thr
            self.wait     = 0
            self.model.save(str(self.out_dir / "model_best.keras"))
            log(f"  epoch {epoch+1:3d}  val_window_minpr={mp:.4f}  thr={thr}  *** new best")
        else:
            self.wait += 1
            if (epoch + 1) % 5 == 0:
                log(f"  epoch {epoch+1:3d}  val_window_minpr={mp:.4f}  thr={thr}  (wait {self.wait}/{self.patience})")
            if self.wait >= self.patience:
                log(f"  Early stopping at epoch {epoch+1}")
                self.model.stop_training = True


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    feat_cols   = FEATURE_SETS[args.feature_set]
    n_feat      = len(feat_cols)
    window_size = args.window_size
    exp_dir     = OUT_ROOT / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    log(f"=== {args.exp_id} ===")
    log(f"  feature_set={args.feature_set} ({n_feat} feat), window={window_size}, model={args.model_type}")

    # ── load data ──────────────────────────────────────────────────────────────
    log("Loading frame-level CSVs …")
    train_data = load_video_frames(DATA_DIR / "train.csv", feat_cols)
    val_data   = load_video_frames(DATA_DIR / "val.csv",   feat_cols)
    test_data  = load_video_frames(DATA_DIR / "test.csv",  feat_cols)
    log(f"  train={len(train_data)} videos, val={len(val_data)}, test={len(test_data)}")

    # ── build windows ──────────────────────────────────────────────────────────
    log("Building windows …")
    X_train, y_train = build_train_windows(train_data, window_size, args.fall_stride, args.nfall_stride, rng)
    X_val,   y_val   = build_eval_windows(val_data,   window_size)
    X_test,  y_test  = build_eval_windows(test_data,  window_size)
    log(f"  train: {len(X_train)} windows  (fall={y_train.sum()}, nfall={(y_train==0).sum()})")
    log(f"  val:   {len(X_val)}   windows  (fall={y_val.sum()}, nfall={(y_val==0).sum()})")
    log(f"  test:  {len(X_test)}  windows  (fall={y_test.sum()}, nfall={(y_test==0).sum()})")

    # ── normalization ──────────────────────────────────────────────────────────
    mn, scale = fit_normalization(X_train)
    X_train = normalize(X_train, mn, scale)
    X_val   = normalize(X_val,   mn, scale)
    X_test  = normalize(X_test,  mn, scale)

    # ── build model ────────────────────────────────────────────────────────────
    model = build_model(n_feat, window_size, args.model_type, args.hidden_sizes, args.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda x: None)
    log(f"  params={model.count_params():,}")

    # ── train ──────────────────────────────────────────────────────────────────
    minpr_cb = WindowMinPRCallback(X_val, y_val, exp_dir, patience=15)
    lr_cb    = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=0,
    )

    log("Training …")
    model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=[minpr_cb, lr_cb],
        verbose=0,
    )

    # ── reload best model ──────────────────────────────────────────────────────
    best_path = exp_dir / "model_best.keras"
    if best_path.exists():
        model = tf.keras.models.load_model(str(best_path), compile=False)
        log(f"Loaded best model (val_window_minpr={minpr_cb.best_mp:.4f}, thr={minpr_cb.best_thr})")
    model.save(str(exp_dir / "model.keras"))

    # ── val evaluation ─────────────────────────────────────────────────────────
    val_probs = model.predict(X_val, batch_size=1024, verbose=0)[:, 1]
    thr, val_mp = select_threshold(y_val, val_probs)
    val_m = window_minpr(y_val, val_probs, thr)
    log(f"Val   window MinPR={val_mp:.4f}  thr={thr}")

    # ── test evaluation ────────────────────────────────────────────────────────
    test_probs = model.predict(X_test, batch_size=1024, verbose=0)[:, 1]
    test_m = window_minpr(y_test, test_probs, thr)
    log(f"Test  window MinPR={test_m['min_pr']:.4f}  "
        f"fall_pr={test_m['fall_precision']:.4f}  "
        f"nfall_pr={test_m['nfall_precision']:.4f}  "
        f"fall_rc={test_m['fall_recall']:.4f}  "
        f"nfall_rc={test_m['nfall_recall']:.4f}  "
        f"FN={test_m['fn']}  FP={test_m['fp']}")

    # ── save artifacts ─────────────────────────────────────────────────────────
    (exp_dir / "feature_columns.json").write_text(json.dumps(feat_cols, indent=2))
    (exp_dir / "normalization.json").write_text(json.dumps({
        "min": mn.tolist(), "scale": scale.tolist(),
    }))

    metrics = {
        "exp_id": args.exp_id,
        "config": {
            "feature_set": args.feature_set,
            "n_features":  n_feat,
            "window_size": window_size,
            "model_type":  args.model_type,
            "hidden_sizes":args.hidden_sizes,
            "epochs":      args.epochs,
            "fall_stride": args.fall_stride,
            "nfall_stride":args.nfall_stride,
        },
        "threshold": thr,
        "metrics": {
            "val_window": {
                "min_pr":          val_m["min_pr"],
                "fall_precision":  val_m["fall_precision"],
                "fall_recall":     val_m["fall_recall"],
                "nfall_precision": val_m["nfall_precision"],
                "nfall_recall":    val_m["nfall_recall"],
                "fn": val_m["fn"], "fp": val_m["fp"],
            },
            "test_window": {
                "min_pr":          test_m["min_pr"],
                "fall_precision":  test_m["fall_precision"],
                "fall_recall":     test_m["fall_recall"],
                "nfall_precision": test_m["nfall_precision"],
                "nfall_recall":    test_m["nfall_recall"],
                "fn": test_m["fn"], "fp": test_m["fp"],
            },
        },
        "window_stats": {
            "train_fall":  int(y_train.sum()),
            "train_nfall": int((y_train==0).sum()),
            "val_fall":    int(y_val.sum()),
            "val_nfall":   int((y_val==0).sum()),
            "test_fall":   int(y_test.sum()),
            "test_nfall":  int((y_test==0).sum()),
        },
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log(f"=== DONE: {args.exp_id}  test_window_minpr={test_m['min_pr']:.4f} ===")
    return test_m["min_pr"]


if __name__ == "__main__":
    main()
