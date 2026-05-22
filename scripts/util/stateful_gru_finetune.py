#!/usr/bin/env python3
"""Stateful GRU fine-tuning — CNN 분리 + 1-step Stateful GRU.

기학습된 CNN-GRU 모델의 가중치를 가져와:
  - Conv1D × 2  : 9-frame 링 버퍼 기반 (수용 범위 = (5-1)×2+1 = 9), 동결 선택 가능
  - GRU(stateful=True) : 비디오 내 상태 유지, 비디오 경계에서 reset_states()

학습 방식:
  - 각 비디오를 시간 순서대로 처리 (윈도우 순서 보장)
  - 비디오 시작마다 reset_states()
  - 윈도우 레벨 레이블 유지 (fall/not-fall)
  - focal loss, 매우 낮은 LR (기본 5e-5)

Usage:
  uv run python scripts/train_stateful_finetune.py \\
      --base-model results/phase36_window_ablation/P38-nv-a65-gru/model_best.keras \\
      --exp-id P40-nv-a65-stateful \\
      --freeze-conv
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
OUT_ROOT = REPO / "results/phase40_stateful"

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
CONV_RECEPTIVE = 9   # (5-1)*2 + 1 = 9 for Conv1D(k=5) × 2 causal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model",  required=True,
                   help="기학습 model_best.keras 경로")
    p.add_argument("--base-dir",    default=None,
                   help="베이스 모델 실험 디렉토리 (feature_columns.json + normalization.json 자동 로드)")
    p.add_argument("--exp-id",      required=True)
    p.add_argument("--out-root",    default=str(OUT_ROOT),
                   help="실험 결과 저장 루트 디렉토리 (기본: results/phase40_stateful)")
    p.add_argument("--train-csv",   default=str(REPO / "dataset/splits_v2_class_balanced_filtered/train.csv"))
    p.add_argument("--val-csv",     default=str(REPO / "dataset/splits_v2_class_balanced_filtered/val.csv"))
    p.add_argument("--test-csv",    default=str(REPO / "dataset/splits_v2_class_balanced_filtered/test.csv"))
    p.add_argument("--window-size", type=int, default=40)
    p.add_argument("--epochs",      type=int, default=15)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--focal-alpha", type=float, default=0.65)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--pure-window", action="store_true", default=True)
    p.add_argument("--pure-margin", type=int, default=5)
    p.add_argument("--nfall-stride",type=int, default=5)
    p.add_argument("--freeze-conv", action="store_true", default=True,
                   help="Conv1D 레이어 동결 (기본 True — GRU만 적응)")
    p.add_argument("--vote-window", type=int, default=5)
    p.add_argument("--vote-k",      type=int, default=3)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def log(msg):
    import datetime
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_video_frames(csv_path, feat_cols):
    data = defaultdict(lambda: {"feat": [], "label": [], "frame": []})
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            lbl = int(row["label"])
            frame_idx = int(row.get("frame", 0))
            try:
                feat = [float(row[c]) for c in feat_cols]
            except (KeyError, ValueError):
                continue
            data[vid]["feat"].append((frame_idx, feat))
            data[vid]["label"].append((frame_idx, lbl))

    result = {}
    for vid, d in data.items():
        if not d["feat"]:
            continue
        d["feat"].sort(key=lambda x: x[0])
        d["label"].sort(key=lambda x: x[0])
        frames = np.array([f for _, f in d["feat"]], np.float32)
        labels = np.array([l for _, l in d["label"]], np.int32)
        result[vid] = (frames, labels)
    return result


def normalize(X, mn, scale):
    return np.clip((X - mn) / scale, 0.0, 1.0)


def find_fall_events(labels):
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


def build_stateful_model(base_model, window_size, n_feat, freeze_conv):
    """기학습 모델에서 구조와 가중치를 추출해 stateful 버전으로 재구성."""

    # 레이어 이름으로 위치 찾기
    conv_layers, gru_layers, dense_layers = [], [], []
    for layer in base_model.layers:
        cname = layer.__class__.__name__
        if cname == "Conv1D":
            conv_layers.append(layer)
        elif cname in ("GRU", "LSTM"):
            gru_layers.append(layer)
        elif cname == "Dense":
            dense_layers.append(layer)

    assert len(conv_layers) == 2, f"Conv1D 레이어 2개 기대, {len(conv_layers)}개 발견"
    assert len(gru_layers)  >= 2, f"GRU 레이어 2개 기대, {len(gru_layers)}개 발견"

    # 원본 GRU 설정 읽기
    gru_units = [l.units for l in gru_layers]
    dropout   = gru_layers[0].dropout

    # --- stateful 모델 구성 (batch_size=1 고정) ---
    inp = tf.keras.Input(batch_shape=(1, window_size, n_feat))
    x   = inp

    # Conv1D — 가중치 복사 후 동결 선택
    for cl in conv_layers:
        new_conv = tf.keras.layers.Conv1D(
            cl.filters, cl.kernel_size[0],
            padding="causal", activation="relu",
            name=f"sf_{cl.name}"
        )
        x = new_conv(x)

    # GRU — stateful=True, unroll=False (stateful과 unroll=True 충돌)
    for i, gl in enumerate(gru_layers):
        is_last = (i == len(gru_layers) - 1)
        x = tf.keras.layers.GRU(
            gl.units,
            stateful=True,
            return_sequences=not is_last,
            dropout=dropout,
            recurrent_dropout=0.0,
            reset_after=True,
            name=f"sf_gru_{i}"
        )(x)

    # Dense layers
    for i, dl in enumerate(dense_layers):
        if dl.activation.__class__.__name__ in ("relu", "ReLU") or \
           hasattr(dl.activation, "__name__") and dl.activation.__name__ == "relu":
            x = tf.keras.layers.Dense(dl.units, activation="relu",
                                       name=f"sf_dense_{i}")(x)
        elif dl.activation.__class__.__name__ in ("softmax", "Softmax") or \
             hasattr(dl.activation, "__name__") and dl.activation.__name__ == "softmax":
            x = tf.keras.layers.Dense(dl.units, activation="softmax",
                                       name=f"sf_out")(x)
        else:
            act = getattr(dl.activation, "__name__", "relu")
            x = tf.keras.layers.Dense(dl.units, activation=act,
                                       name=f"sf_dense_{i}")(x)

    model = tf.keras.Model(inp, x)

    # 가중치 복사
    sf_convs  = [l for l in model.layers if "sf_conv" in l.name]
    sf_grus   = [l for l in model.layers if "sf_gru"  in l.name]
    sf_denses = [l for l in model.layers if l.name.startswith("sf_dense") or l.name == "sf_out"]

    for src, dst in zip(conv_layers, sf_convs):
        dst.set_weights(src.get_weights())
        dst.trainable = not freeze_conv

    for src, dst in zip(gru_layers, sf_grus):
        dst.set_weights(src.get_weights())

    for src, dst in zip(dense_layers, sf_denses):
        dst.set_weights(src.get_weights())

    trainable_count = sum(np.prod(w.shape) for w in model.trainable_weights)
    frozen_count    = sum(np.prod(w.shape) for w in model.non_trainable_weights)
    log(f"Stateful 모델: trainable={trainable_count:,}  frozen={frozen_count:,}")
    if freeze_conv:
        log("Conv1D 동결 — GRU + Dense만 파인튜닝")

    return model


def focal_loss(gamma=2.0, alpha=0.65):
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        probs = y_pred[:, 1]
        bce = -y_true_f * tf.math.log(probs + 1e-9) \
              - (1 - y_true_f) * tf.math.log(1 - probs + 1e-9)
        p_t = tf.where(tf.cast(y_true_f, bool), probs, 1 - probs)
        a_t = tf.where(tf.cast(y_true_f, bool),
                       tf.constant(alpha), tf.constant(1 - alpha))
        return tf.reduce_mean(a_t * tf.pow(1 - p_t, gamma) * bce)
    return loss


@tf.function
def _forward_and_grads(model, x, y_label, loss_fn):
    """단일 윈도우 forward + gradient 계산 — compiled."""
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y_label, pred)
    return tape.gradient(loss, model.trainable_variables), loss


def make_sequential_windows(video_data, window_size, mn, scale,
                             pure_window, pure_margin, nfall_stride):
    """비디오별 시간순 윈도우 목록 생성.

    Returns:
        list of (video_id, [(window_array, label), ...]) — 비디오 내 시간 순서 보장
    """
    result = []
    for vid, (frames, labels) in video_data.items():
        n = len(frames)
        if n < window_size:
            continue
        f = normalize(frames, mn, scale)

        wins = []
        if pure_window:
            fall_set = set()
            for (fs, fe) in find_fall_events(labels):
                w_min = max(0, fe - pure_margin - window_size + 1)
                w_max = min(fs, n - window_size)
                for w_start in range(w_min, w_max + 1):
                    fall_set.add(w_start)
                    wins.append((w_start, f[w_start:w_start + window_size], 1))
            for t in range(window_size - 1, n):
                w_start = t - window_size + 1
                if w_start not in fall_set and np.all(labels[w_start:t + 1] == 0):
                    if w_start % nfall_stride == 0:
                        wins.append((w_start, f[w_start:t + 1], 0))
        else:
            for t in range(window_size - 1, n):
                w_start = t - window_size + 1
                lbl = int(labels[t])
                if lbl == 0 and w_start % nfall_stride != 0:
                    continue
                wins.append((w_start, f[w_start:t + 1], lbl))

        if wins:
            wins.sort(key=lambda x: x[0])  # 시간 순 정렬
            result.append((vid, [(w, l) for _, w, l in wins]))

    return result


def train_one_epoch(model, video_windows, optimizer, loss_fn, rng):
    """비디오 단위 순차 학습.

    최적화:
      - @tf.function compiled forward+grad (_forward_and_grads)
      - gradient accumulation: 비디오 내 모든 윈도우 grad 누산 후 1회 apply
        (optimizer step N→1/video, 오버헤드 대폭 감소)
    """
    vid_order = list(range(len(video_windows)))
    rng.shuffle(vid_order)

    total_loss = 0.0
    total_wins = 0
    tvars = model.trainable_variables

    for vi in vid_order:
        vid_id, wins = video_windows[vi]
        reset_model_states(model)  # stateful: 비디오 경계 리셋

        accum = [tf.zeros_like(v) for v in tvars]
        vid_loss = 0.0

        for win_arr, label in wins:
            x = tf.constant(win_arr[np.newaxis], dtype=tf.float32)
            y = tf.constant([label], dtype=tf.int32)
            grads, loss_val = _forward_and_grads(model, x, y, loss_fn)
            accum = [a + (g if g is not None else tf.zeros_like(v))
                     for a, g, v in zip(accum, grads, tvars)]
            vid_loss += float(loss_val)

        # 비디오당 1회 optimizer step
        n = len(wins)
        optimizer.apply_gradients(zip([g / n for g in accum], tvars))

        total_loss += vid_loss
        total_wins += n

    return total_loss / max(total_wins, 1)


def build_eval_proxy(stateful_model, window_size, n_feat):
    """파인튜닝된 가중치를 stateless 모델에 이식 — 배치 평가용.

    eval 중 stateful 1-step 루프 대신 batch predict를 사용해 10× 속도 향상.
    가중치는 동일하므로 window-level 정확도 측정에 유효하다.
    """
    inp = tf.keras.Input(shape=(window_size, n_feat))
    x   = inp
    # stateful 모델 레이어에서 Conv1D, GRU, Dense 순서로 재구성
    sf_convs  = [l for l in stateful_model.layers if "sf_conv" in l.name]
    sf_grus   = [l for l in stateful_model.layers if "sf_gru"  in l.name]
    sf_denses = [l for l in stateful_model.layers
                 if l.name.startswith("sf_dense") or l.name == "sf_out"]

    proxy_convs  = []
    proxy_grus   = []
    proxy_denses = []

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

    for i, dl in enumerate(sf_denses):
        act = "softmax" if dl.name == "sf_out" else "relu"
        new_l = tf.keras.layers.Dense(dl.units, activation=act)
        x = new_l(x)
        proxy_denses.append(new_l)

    proxy = tf.keras.Model(inp, x)

    # 가중치 복사
    for src, dst in zip(sf_convs,  proxy_convs):  dst.set_weights(src.get_weights())
    for src, dst in zip(sf_grus,   proxy_grus):   dst.set_weights(src.get_weights())
    for src, dst in zip(sf_denses, proxy_denses): dst.set_weights(src.get_weights())

    return proxy


def reset_model_states(model):
    """Functional 모델의 모든 stateful 레이어 리셋."""
    for layer in model.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()


def precompute_scores(proxy_model, video_data, window_size, mn, scale):
    """val/test 비디오 스코어를 비디오당 1회 predict로 캐시.

    threshold sweep 시 재사용 — 23× 호출 → 1× 호출로 단축.
    큰 concat 대신 비디오별 predict로 메모리 안전.
    """
    cached = {}
    for vid, (frames, labels) in video_data.items():
        n = len(frames)
        if n < window_size:
            continue
        f = normalize(frames, mn, scale)
        n_wins = n - window_size + 1
        windows = np.stack(
            [f[t:t + window_size] for t in range(n_wins)], 0
        ).astype(np.float32)
        scores   = proxy_model.predict(windows, batch_size=512, verbose=0)[:, 1]
        has_fall = bool(np.any(labels == 1))
        cached[vid] = (scores, has_fall)
    return cached


def _apply_vote(cached_scores, threshold, vote_window=5, vote_k=3):
    """캐시된 스코어에 threshold + vote 로직 적용 — 순수 numpy, 매우 빠름."""
    tp = fp = fn = tn = 0
    for vid, (scores, has_fall) in cached_scores.items():
        vote_buf  = np.zeros(vote_window, dtype=np.int32)
        vote_head = 0
        vote_sum  = 0
        detected  = False
        for score in scores:
            this_vote = 1 if score >= threshold else 0
            old_vote  = int(vote_buf[vote_head])
            vote_buf[vote_head] = this_vote
            vote_head = (vote_head + 1) % vote_window
            vote_sum += this_vote - old_vote
            if not detected and vote_sum >= vote_k:
                detected = True
        if detected and has_fall:      tp += 1
        elif detected and not has_fall: fp += 1
        elif not detected and has_fall: fn += 1
        else:                           tn += 1

    fall_prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fall_rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    nfall_prec = tn / (tn + fn) if (tn + fn) > 0 else 1.0  # TN/(TN+FN)
    nfall_rec  = tn / (tn + fp) if (tn + fp) > 0 else 1.0  # TN/(TN+FP)
    min_pr = min(fall_prec, fall_rec, nfall_prec, nfall_rec)
    return {
        "min_pr": min_pr,
        "fall_precision": fall_prec, "fall_recall": fall_rec,
        "nfall_precision": nfall_prec, "nfall_recall": nfall_rec,
        "fall_pr": fall_prec, "nfall_pr": nfall_prec,  # backward compat
        "fp": fp, "fn": fn, "tp": tp, "tn": tn,
    }


def event_vote_eval_batch(proxy_model, video_data, window_size, threshold, mn, scale,
                          vote_window=5, vote_k=3):
    cached = precompute_scores(proxy_model, video_data, window_size, mn, scale)
    return _apply_vote(cached, threshold, vote_window, vote_k)


def threshold_sweep_batch(proxy_model, video_data, window_size, mn, scale,
                          vote_window=5, vote_k=3):
    """스코어 1회 precompute 후 23개 threshold를 numpy로 즉시 sweep."""
    cached = precompute_scores(proxy_model, video_data, window_size, mn, scale)
    best_thr, best_minpr = 0.5, 0.0
    for thr in np.arange(0.40, 0.96, 0.025):
        r = _apply_vote(cached, float(thr), vote_window, vote_k)
        if r["min_pr"] > best_minpr:
            best_minpr = r["min_pr"]
            best_thr   = float(thr)
    return best_thr, best_minpr


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    out_dir = Path(args.out_root) / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 베이스 모델 로드 ──────────────────────────────────────────────────────
    log(f"베이스 모델 로드: {args.base_model}")
    base_model = tf.keras.models.load_model(args.base_model, compile=False)

    # base-dir: feature_columns.json + normalization.json 자동 로드
    base_dir = Path(args.base_dir) if args.base_dir else Path(args.base_model).parent
    feat_cols_path = base_dir / "feature_columns.json"
    if feat_cols_path.exists():
        feat_cols = json.load(open(feat_cols_path))
        log(f"feature_columns.json 로드: {len(feat_cols)}개 피처")
    else:
        feat_cols = KP13_COLS
        log(f"feature_columns.json 없음 → KP13_COLS({len(KP13_COLS)}개) 사용")

    # 정규화 파라미터 로드
    norm_path = base_dir / "normalization.json"
    if not norm_path.exists():
        norm_path = Path(args.base_model).parent / "normalization.json"
    norm      = json.load(open(norm_path))
    mn        = np.array(norm["min"],   dtype=np.float32)
    scale     = np.array(norm["scale"], dtype=np.float32)
    n_feat    = len(mn)
    log(f"정규화 파라미터 로드: {n_feat}개 피처")

    # ── Stateful 모델 구성 ────────────────────────────────────────────────────
    model = build_stateful_model(
        base_model, args.window_size, n_feat, args.freeze_conv
    )
    model.summary(print_fn=log)

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    log("데이터 로드 중...")
    train_data = load_video_frames(args.train_csv, feat_cols)
    val_data   = load_video_frames(args.val_csv,   feat_cols)
    test_data  = load_video_frames(args.test_csv,  feat_cols)
    log(f"  train: {len(train_data)} videos / val: {len(val_data)} / test: {len(test_data)}")

    # 학습용 순차 윈도우 구성
    log("학습 윈도우 구성 중 (비디오별 시간순)...")
    train_vwins = make_sequential_windows(
        train_data, args.window_size, mn, scale,
        args.pure_window, args.pure_margin, args.nfall_stride
    )
    total_wins = sum(len(w) for _, w in train_vwins)
    fall_wins  = sum(sum(1 for _, l in w if l == 1) for _, w in train_vwins)
    log(f"  총 {total_wins}개 윈도우 (fall={fall_wins}, nfall={total_wins-fall_wins}) / {len(train_vwins)} 비디오")

    # ── 학습 설정 ─────────────────────────────────────────────────────────────
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn   = focal_loss(args.focal_gamma, args.focal_alpha)

    # 베이스라인 ev_vote 확인 (배치 평가, 고속)
    log("베이스라인 평가 (가중치 복사 직후)...")
    proxy = build_eval_proxy(model, args.window_size, n_feat)
    base_thr, base_val_minpr = threshold_sweep_batch(
        proxy, val_data, args.window_size, mn, scale, args.vote_window, args.vote_k)
    base_val = event_vote_eval_batch(proxy, val_data, args.window_size,
                                     base_thr, mn, scale, args.vote_window, args.vote_k)
    log(f"  val ev_vote = {base_val_minpr:.4f}  thr={base_thr:.3f}  "
        f"FP={base_val['fp']}  FN={base_val['fn']}")

    # ── 파인튜닝 루프 ─────────────────────────────────────────────────────────
    best_val_minpr = base_val_minpr
    best_thr       = base_thr
    best_weights   = model.get_weights()
    history        = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_vwins, optimizer, loss_fn, rng)

        # val threshold sweep (5 epoch마다, 배치 평가로 고속)
        if epoch % 5 == 0 or epoch == args.epochs:
            proxy = build_eval_proxy(model, args.window_size, n_feat)
            thr, val_minpr = threshold_sweep_batch(
                proxy, val_data, args.window_size, mn, scale,
                args.vote_window, args.vote_k)
            val_r = event_vote_eval_batch(proxy, val_data, args.window_size,
                                          thr, mn, scale, args.vote_window, args.vote_k)
            log(f"Epoch {epoch:3d}  loss={train_loss:.4f}  "
                f"val_ev={val_minpr:.4f}  thr={thr:.3f}  FP={val_r['fp']}  FN={val_r['fn']}")

            history.append({
                "epoch": epoch, "train_loss": train_loss,
                "val_ev": val_minpr, "threshold": thr
            })

            if val_minpr > best_val_minpr:
                best_val_minpr = val_minpr
                best_thr       = thr
                best_weights   = model.get_weights()
                log(f"  ★ NEW BEST val_ev = {best_val_minpr:.4f}  thr={best_thr:.3f}")
        else:
            log(f"Epoch {epoch:3d}  loss={train_loss:.4f}")
            history.append({"epoch": epoch, "train_loss": train_loss})

    # ── 최고 가중치로 복원 후 테스트 ──────────────────────────────────────────
    log(f"\n최고 가중치 복원 (val_ev={best_val_minpr:.4f}, thr={best_thr:.3f})")
    model.set_weights(best_weights)

    proxy = build_eval_proxy(model, args.window_size, n_feat)
    test_r = event_vote_eval_batch(proxy, test_data, args.window_size,
                                   best_thr, mn, scale, args.vote_window, args.vote_k)
    log(f"test ev_vote = {test_r['min_pr']:.4f}  "
        f"FallPR={test_r['fall_pr']:.4f}  NFallPR={test_r['nfall_pr']:.4f}  "
        f"FP={test_r['fp']}  FN={test_r['fn']}")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    model.save(out_dir / "model_stateful.keras")
    log(f"모델 저장: {out_dir}/model_stateful.keras")

    metrics = {
        "base_model":    args.base_model,
        "threshold":     best_thr,
        "freeze_conv":   args.freeze_conv,
        "epochs":        args.epochs,
        "lr":            args.lr,
        "baseline_val_ev": base_val["min_pr"],
        "metrics": {
            "val_event_vote":  {"min_pr": best_val_minpr},
            "test_event_vote": test_r,
        },
        "history": history,
    }
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)
    json.dump(norm,    open(out_dir / "normalization.json", "w"), indent=2)
    log(f"완료: {out_dir}")
    log(f"결과: baseline_val={base_val['min_pr']:.4f} → best_val={best_val_minpr:.4f} → test={test_r['min_pr']:.4f}")


if __name__ == "__main__":
    main()
