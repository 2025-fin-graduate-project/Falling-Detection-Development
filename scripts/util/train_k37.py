#!/usr/bin/env python3
"""Train Conv-GRU model with Keras 3.7 (STedgeAI Python, no pandas/sklearn).

Loads pre-built numpy arrays from build_train_npy.py, trains, saves Keras 3.7
format model.keras, and writes metrics.json.

Usage (STedgeAI Python = Keras 3.7):
    /home/min/app/ST/STEdgeAI/4.0/Utilities/linux/python \
        scripts/util/train_k37.py \
        --npy-dir /tmp/p44_kp13_w60_npy \
        --output-dir results/phase44_keras37/P44-kp13-w60 \
        --gru-units 64,32 \
        --conv-filters 64 --conv-kernel 5 --conv-layers 2 \
        --focal-alpha 0.25 --focal-gamma 2.0 \
        --dropout-rate 0.3 --noise-std 0.02 \
        --batch-size 512 --epochs 100 --patience 15 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

# ── reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ── focal loss ───────────────────────────────────────────────────────────────

class SparseFocalLoss(keras.losses.Loss):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, **kw):
        super().__init__(**kw)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        idx = tf.stack([tf.range(tf.shape(y_true)[0]), y_true], axis=1)
        p_t = tf.gather_nd(y_pred, idx)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * tf.pow(1.0 - p_t, self.gamma)
        return focal_weight * (-tf.math.log(p_t))

    def get_config(self):
        return {**super().get_config(), "alpha": self.alpha, "gamma": self.gamma}


# ── model ────────────────────────────────────────────────────────────────────

def build_model(
    input_shape: tuple[int, int],
    gru_units: list[int],
    conv_filters: int,
    conv_kernel: int,
    conv_layers: int,
    dropout_rate: float,
) -> keras.Model:
    inp = keras.Input(shape=input_shape, name="pose_sequence")
    x = inp
    for i in range(conv_layers):
        x = keras.layers.Conv1D(
            conv_filters, conv_kernel, padding="causal", activation="relu", name=f"conv{i+1}"
        )(x)
    for i, units in enumerate(gru_units):
        return_seq = (i < len(gru_units) - 1)
        x = keras.layers.GRU(
            units,
            return_sequences=return_seq,
            unroll=True,
            reset_after=True,
            name=f"gru{i+1}",
        )(x)
    x = keras.layers.Dense(32, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(dropout_rate, name="drop")(x)
    x = keras.layers.Dense(2, activation="softmax", name="output")(x)
    return keras.Model(inp, x)


# ── evaluation helpers (no pandas/sklearn) ───────────────────────────────────

def apply_consecutive_rule(binary: np.ndarray, min_consec: int) -> np.ndarray:
    if min_consec <= 1:
        return binary.astype(np.int32)
    out = np.zeros_like(binary, dtype=np.int32)
    start = None
    for idx, v in enumerate(binary):
        if v == 1 and start is None:
            start = idx
        elif v == 0 and start is not None:
            if idx - start >= min_consec:
                out[start:idx] = 1
            start = None
    if start is not None and len(binary) - start >= min_consec:
        out[start:] = 1
    return out


def video_minpr(
    y_win: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    min_consec: int,
) -> tuple[float, float, float, float, float]:
    """Returns (minpr, fall_pr, nfall_pr, fall_rc, nfall_rc)."""
    video_ids = np.unique(groups)
    v_true = np.empty(len(video_ids), dtype=np.int32)
    v_pred = np.empty(len(video_ids), dtype=np.int32)
    for i, vid in enumerate(video_ids):
        mask = groups == vid
        v_true[i] = int(y_win[mask].max())
        binary = (scores[mask] >= threshold).astype(np.int32)
        filtered = apply_consecutive_rule(binary, min_consec)
        v_pred[i] = int(filtered.max()) if len(filtered) > 0 else 0

    tp = int(((v_true == 1) & (v_pred == 1)).sum())
    fp = int(((v_true == 0) & (v_pred == 1)).sum())
    fn = int(((v_true == 1) & (v_pred == 0)).sum())
    tn = int(((v_true == 0) & (v_pred == 0)).sum())

    fall_pr  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    nfall_pr = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fall_rc  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    nfall_rc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    minpr = min(fall_pr, nfall_pr, fall_rc, nfall_rc)
    return minpr, fall_pr, nfall_pr, fall_rc, nfall_rc


def threshold_sweep(
    y_win: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    thresholds: list[float],
    min_consec_values: list[int],
) -> tuple[float, float, int, float]:
    """Returns (best_minpr, best_thr, best_mc, best_fall_pr)."""
    best = 0.0
    best_thr = 0.5
    best_mc = 1
    best_fall_pr = 0.0
    for thr in thresholds:
        for mc in min_consec_values:
            minpr, fall_pr, *_ = video_minpr(y_win, scores, groups, thr, mc)
            if minpr > best:
                best = minpr
                best_thr = thr
                best_mc = mc
                best_fall_pr = fall_pr
    return best, best_thr, best_mc, best_fall_pr


# ── MinPR callback with early stopping ───────────────────────────────────────

class MinPRCallback(keras.callbacks.Callback):
    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        groups_val: np.ndarray,
        patience: int,
        batch_size: int,
        thresholds: list[float],
        min_consec_values: list[int],
        quiet: bool = False,
    ):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.groups_val = groups_val
        self.patience = patience
        self.batch_size = batch_size
        self.thresholds = thresholds
        self.min_consec_values = min_consec_values
        self.quiet = quiet
        self.best_min_pr: float = -1.0
        self.best_weights = None
        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.history: list[dict] = []

    def on_epoch_end(self, epoch: int, logs=None):
        raw = self.model.predict(self.X_val, batch_size=self.batch_size, verbose=0)
        scores = raw[:, 1]
        best, best_thr, best_mc, _ = threshold_sweep(
            self.y_val, scores, self.groups_val,
            self.thresholds, self.min_consec_values,
        )
        self.history.append({"epoch": epoch + 1, "val_minpr": best, "thr": best_thr, "mc": best_mc})

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        if best > self.best_min_pr:
            self.best_min_pr = best
            self.best_weights = self.model.get_weights()
            self.wait = 0
            if not self.quiet:
                print(f"[{ts}] ep{epoch+1}: val_minpr={best:.4f} ↑ (thr={best_thr:.3f} mc={best_mc}) — saved", flush=True)
        else:
            self.wait += 1
            if not self.quiet:
                print(f"[{ts}] ep{epoch+1}: val_minpr={best:.4f} (best={self.best_min_pr:.4f} wait={self.wait}/{self.patience})", flush=True)
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"[MinPRCallback] restored best weights (val_minpr={self.best_min_pr:.4f})", flush=True)


# ── noise augmentation (applied per batch via tf.data) ────────────────────────

def add_gaussian_noise(X: np.ndarray, noise_std: float) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(X)

    def _add_noise(x):
        return x + tf.random.normal(tf.shape(x), stddev=noise_std)

    return ds.map(_add_noise, num_parallel_calls=tf.data.AUTOTUNE)


# ── class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    n = len(y)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        return {0: 1.0, 1: 1.0}
    w0 = n / (2 * n0)
    w1 = n / (2 * n1)
    return {0: w0, 1: w1}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-dir",     required=True,  help="Directory from build_train_npy.py")
    ap.add_argument("--output-dir",  required=True,  help="Where to write model.keras + metrics.json")
    ap.add_argument("--gru-units",   default="64,32")
    ap.add_argument("--conv-filters",type=int, default=64)
    ap.add_argument("--conv-kernel", type=int, default=5)
    ap.add_argument("--conv-layers", type=int, default=2)
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--dropout-rate",type=float, default=0.3)
    ap.add_argument("--noise-std",   type=float, default=0.02)
    ap.add_argument("--batch-size",  type=int,   default=512)
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--patience",    type=int,   default=15)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--quiet",       action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    npy = Path(args.npy_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # skip if already done
    if (out / "metrics.json").exists():
        print(f"[SKIP] {out} already has metrics.json", flush=True)
        sys.exit(0)

    # load arrays
    print(f"Loading numpy arrays from {npy} …", flush=True)
    X_tr = np.load(npy / "X_train.npy")
    y_tr = np.load(npy / "y_train.npy")
    X_va = np.load(npy / "X_val.npy")
    y_va = np.load(npy / "y_val.npy")
    g_va = np.load(npy / "groups_val.npy",  allow_pickle=True)
    X_te = np.load(npy / "X_test.npy")
    y_te = np.load(npy / "y_test.npy")
    g_te = np.load(npy / "groups_test.npy", allow_pickle=True)
    data_cfg = json.loads((npy / "data_config.json").read_text())

    gru_units = [int(u) for u in args.gru_units.split(",")]
    input_shape = (data_cfg["target_steps"], data_cfg["num_features"])
    print(f"Input shape: {input_shape}  GRU units: {gru_units}", flush=True)

    # build model
    model = build_model(input_shape, gru_units, args.conv_filters, args.conv_kernel, args.conv_layers, args.dropout_rate)
    model.summary()

    # class weights
    cw = None if args.no_class_weight else compute_class_weights(y_tr)
    if cw:
        print(f"Class weights: {cw}", flush=True)

    # compile
    loss = SparseFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )

    # thresholds for sweep
    thresholds = list(np.linspace(0.10, 0.90, 33))
    min_consec_values = [1, 3, 5]

    minpr_cb = MinPRCallback(
        X_va, y_va, g_va,
        patience=args.patience,
        batch_size=args.batch_size,
        thresholds=thresholds,
        min_consec_values=min_consec_values,
        quiet=args.quiet,
    )

    # training dataset with noise augmentation
    def make_ds(X, y, shuffle=False, noise=0.0):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(X), 10000), seed=args.seed)
        if noise > 0:
            def add_noise(x, lbl):
                return x + tf.random.normal(tf.shape(x), stddev=noise), lbl
            ds = ds.map(add_noise, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(X_tr, y_tr, shuffle=True, noise=args.noise_std)

    print(f"\n=== Training {out.name} ===", flush=True)
    model.fit(
        train_ds,
        epochs=args.epochs,
        callbacks=[minpr_cb],
        class_weight=cw,
        verbose=0,
    )

    # final threshold selection on val
    print("\nFinal threshold sweep on val …", flush=True)
    raw_val = model.predict(X_va, batch_size=args.batch_size, verbose=0)
    s_val = raw_val[:, 1]
    best_minpr, best_thr, best_mc, best_fall_pr = threshold_sweep(
        y_va, s_val, g_va, thresholds, min_consec_values
    )
    print(f"  val_minpr={best_minpr:.4f}  thr={best_thr:.3f}  mc={best_mc}", flush=True)

    # test evaluation
    print("Evaluating on test set …", flush=True)
    raw_te = model.predict(X_te, batch_size=args.batch_size, verbose=0)
    s_te = raw_te[:, 1]
    test_minpr, t_fall_pr, t_nfall_pr, t_fall_rc, t_nfall_rc = video_minpr(
        y_te, s_te, g_te, best_thr, best_mc
    )
    print(f"  test_minpr={test_minpr:.4f}  fall_pr={t_fall_pr:.4f}  nfall_pr={t_nfall_pr:.4f}", flush=True)
    print(f"  fall_rc={t_fall_rc:.4f}  nfall_rc={t_nfall_rc:.4f}", flush=True)

    # save model
    model_path = out / "model.keras"
    model.save(str(model_path))
    print(f"Saved model → {model_path}", flush=True)

    # save metrics.json (compatible with eval_stedgeai_host.py)
    run_cfg = {
        "feature_set": data_cfg["feature_set"],
        "preprocessing": data_cfg["preprocessing"],
        "target_steps": data_cfg["target_steps"],
        "window_start_sec": data_cfg["window_start_sec"],
        "window_end_sec": data_cfg["window_end_sec"],
        "label_column": data_cfg["label_column"],
        "gru_units": gru_units,
        "conv_pre_filters": args.conv_filters,
        "conv_pre_kernel": args.conv_kernel,
        "conv_pre_layers": args.conv_layers,
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
        "dropout_rate": args.dropout_rate,
        "noise_std": args.noise_std,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "keras_version": keras.__version__,
        "npy_dir": str(npy),
    }
    (out / "run_config.resolved.json").write_text(json.dumps(run_cfg, indent=2))

    metrics = {
        "threshold_selection": {
            "threshold": best_thr,
            "min_consecutive": best_mc,
        },
        "metrics": {
            "test_video": {
                "min_precision": test_minpr,
                "precision": t_fall_pr,
                "nfall_precision": t_nfall_pr,
                "recall": t_fall_rc,
                "nfall_recall": t_nfall_rc,
            }
        },
        "training_history": minpr_cb.history,
        "best_val_minpr": float(minpr_cb.best_min_pr),
        "stopped_epoch": minpr_cb.stopped_epoch,
        "keras_version": keras.__version__,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics.json → {out}/metrics.json", flush=True)

    print(f"\n[DONE] {out.name}:  test_minpr={test_minpr:.4f}", flush=True)


if __name__ == "__main__":
    main()
