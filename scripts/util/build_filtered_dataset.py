#!/usr/bin/env python3
"""
build_filtered_dataset.py  —  Pipeline D (최적 전처리)

final_dataset.csv → final_dataset_filtered.csv

처리 단계:
  1. 좌표 클리핑          — kp*_y/x → [0, 1]  (OOR 100% 제거)
  2. 저신뢰도 마스킹      — score < conf_thr → NaN → 선형보간
  3. One-Euro Filter       — min_cutoff=0.5, beta=0.3
  4. EMA Filter            — kp*_s (신뢰도 스무딩, alpha=0.5)
  5. 파생 피처 재계산      — HSSC_y/x, RWHC, VHSSC
  6. VHSSC EMA 스무딩      — alpha=0.4  (AHSSC 잡음 -71.9%)
  7. 가속도 계산           — AHSSC, AHSSC_x

분석 근거: docs/keypoint-data-analysis-and-filter-strategy.md
  - VHSSC Cohen's d: 0.59 → 0.96 (+62.4%)
  - AHSSC noise std: 3.35 → 0.94 (-71.9%)
  - VHSSC SNR:       0.90 → 1.66 (+84.7%)
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ─────────────────────────────────────────────
# Filter implementations
# ─────────────────────────────────────────────

class OneEuroFilter:
    """
    One-Euro Filter (Casiez et al., CHI 2012).
    기본값: min_cutoff=0.5 Hz, beta=0.3
      - 정지 상태 alpha=0.136 (이전값 86.4% 유지)
      - 낙상 속도(3/s)에서 cutoff=1.4 Hz — 낙상 신호 추종 가능
    """
    def __init__(self, t0: float, x0: float,
                 min_cutoff: float = 0.5,
                 beta: float = 0.3,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

    @staticmethod
    def _alpha(t_e: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def __call__(self, t: float, x: float) -> float:
        if math.isnan(x):
            return self.x_prev
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return self.x_prev
        dx = (x - self.x_prev) / t_e
        a_d = self._alpha(t_e, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class EMAFilter:
    def __init__(self, x0: float, alpha: float = 0.5):
        self.alpha = alpha
        self.x_prev = x0 if not math.isnan(x0) else 0.0

    def __call__(self, x: float) -> float:
        if math.isnan(x):
            return self.x_prev
        x_hat = self.alpha * x + (1.0 - self.alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat


# ─────────────────────────────────────────────
# Column helpers
# ─────────────────────────────────────────────

N_KP = 17
KP_Y_COLS   = [f"kp{i}_y" for i in range(N_KP)]
KP_X_COLS   = [f"kp{i}_x" for i in range(N_KP)]
KP_S_COLS   = [f"kp{i}_s" for i in range(N_KP)]
HSSC_KP_IDX = list(range(7))
HSSC_Y_COLS = [f"kp{i}_y" for i in HSSC_KP_IDX]
HSSC_X_COLS = [f"kp{i}_x" for i in HSSC_KP_IDX]


# ─────────────────────────────────────────────
# Per-video processing  (Pipeline D)
# ─────────────────────────────────────────────

def _process_video(group: pd.DataFrame,
                   min_cutoff: float,
                   beta: float,
                   d_cutoff: float,
                   conf_alpha: float,
                   conf_thr: float,
                   ema_deriv_alpha: float) -> pd.DataFrame:
    g = group.sort_values("time_sec").copy()
    if g.empty:
        return g

    times = g["time_sec"].to_numpy()

    # ── Step 1 & 2: 저신뢰도 마스킹 → 선형보간 → 클리핑 ─────────────
    for i in range(N_KP):
        s_col = f"kp{i}_s"
        if s_col not in g.columns:
            continue
        scores = g[s_col].to_numpy(dtype=float)
        low = scores < conf_thr

        for coord in [f"kp{i}_y", f"kp{i}_x"]:
            if coord not in g.columns:
                continue
            v = g[coord].to_numpy(dtype=float).copy()
            # 저신뢰도 마스킹
            v[low] = np.nan
            # 선형 보간
            nans = np.isnan(v)
            if nans.any() and (~nans).any():
                idx = np.arange(len(v))
                v[nans] = np.interp(idx[nans], idx[~nans], v[~nans])
            # 좌표 클리핑
            v = np.clip(v, 0.0, 1.0)
            g[coord] = v

    # ── Step 3: One-Euro Filter (좌표) ───────────────────────────────
    for col in KP_Y_COLS + KP_X_COLS:
        if col not in g.columns:
            continue
        v = g[col].to_numpy(dtype=float).copy()
        first = int(np.argmax(~np.isnan(v))) if np.any(~np.isnan(v)) else None
        if first is None:
            continue
        f = OneEuroFilter(times[first], v[first], min_cutoff, beta, d_cutoff)
        for idx in range(first, len(v)):
            v[idx] = f(times[idx], v[idx])
        g[col] = v

    # ── Step 4: EMA Filter (신뢰도) ──────────────────────────────────
    for col in KP_S_COLS:
        if col not in g.columns:
            continue
        v = g[col].to_numpy(dtype=float).copy()
        x0 = 0.0 if np.isnan(v[0]) else v[0]
        f = EMAFilter(x0, alpha=conf_alpha)
        for idx in range(len(v)):
            v[idx] = f(v[idx])
        g[col] = v

    # ── Step 5: 파생 피처 재계산 ─────────────────────────────────────
    y_cols_h = [c for c in HSSC_Y_COLS if c in g.columns]
    x_cols_h = [c for c in HSSC_X_COLS if c in g.columns]
    if y_cols_h:
        g["HSSC_y"] = g[y_cols_h].mean(axis=1)
    if x_cols_h:
        g["HSSC_x"] = g[x_cols_h].mean(axis=1)

    y_all = [c for c in KP_Y_COLS if c in g.columns]
    x_all = [c for c in KP_X_COLS if c in g.columns]
    if y_all and x_all:
        width  = g[x_all].max(axis=1) - g[x_all].min(axis=1)
        height = g[y_all].max(axis=1) - g[y_all].min(axis=1)
        g["RWHC"] = width / height.replace(0.0, 1e-4)

    if "HSSC_y" in g.columns:
        g["VHSSC"] = np.gradient(g["HSSC_y"].values, times)

    # ── Step 6: VHSSC EMA 스무딩 ─────────────────────────────────────
    if "VHSSC" in g.columns and ema_deriv_alpha > 0:
        g["VHSSC"] = g["VHSSC"].ewm(alpha=ema_deriv_alpha, adjust=False).mean()

    # ── Step 7: 가속도 계산 ───────────────────────────────────────────
    if "VHSSC" in g.columns:
        g["AHSSC"] = np.gradient(g["VHSSC"].values, times)

    if "HSSC_x" in g.columns:
        vhssc_x = np.gradient(g["HSSC_x"].values, times)
        g["AHSSC_x"] = np.gradient(vhssc_x, times)

    return g


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        sys.exit(f"[ERROR] 입력 파일 없음: {input_path}")

    print(f"[1/4] 로딩: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"      rows={len(df):,}  cols={len(df.columns)}")

    required = {"video_id", "time_sec"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] 필수 컬럼 누락: {missing}")

    print(f"[2/4] Pipeline D 적용 (video 수={df['video_id'].nunique():,})")
    print(f"      OneEuro: min_cutoff={args.min_cutoff}, beta={args.beta}")
    print(f"      conf_thr={args.conf_thr}, ema_deriv_alpha={args.ema_deriv_alpha}")

    groups  = df.groupby("video_id", sort=False)
    results = []
    iterator = tqdm(groups, total=groups.ngroups, unit="video") if HAS_TQDM else groups

    for vid, grp in iterator:
        results.append(_process_video(
            grp,
            min_cutoff=args.min_cutoff,
            beta=args.beta,
            d_cutoff=args.d_cutoff,
            conf_alpha=args.conf_alpha,
            conf_thr=args.conf_thr,
            ema_deriv_alpha=args.ema_deriv_alpha,
        ))

    print("[3/4] 결합 및 정렬 중...")
    df_out = pd.concat(results, ignore_index=True)
    df_out = df_out.sort_values(["video_id", "time_sec"]).reset_index(drop=True)

    # label_3class 생성: 0=normal, 1=falling, 2=fallen
    # 각 비디오에서 마지막 label=1 프레임 이후를 2(fallen)로 표시
    if "label" in df_out.columns:
        df_out["label_3class"] = df_out["label"].copy()
        for vid, grp in df_out.groupby("video_id"):
            fall_idx = grp.index[grp["label"] == 1]
            if len(fall_idx) > 0:
                last_fall = fall_idx.max()
                after_fall = grp.index[grp.index > last_fall]
                df_out.loc[after_fall, "label_3class"] = 2

    # 컬럼 순서 정리
    meta   = [c for c in ["video_id", "frame", "time_sec"] if c in df_out.columns]
    kp     = [c for c in df_out.columns if c.startswith("kp")]
    feat   = [c for c in ["HSSC_y", "HSSC_x", "RWHC", "VHSSC", "AHSSC", "AHSSC_x"] if c in df_out.columns]
    labels = [c for c in df_out.columns if c.startswith("label") or c == "label"]
    other  = [c for c in df_out.columns if c not in meta + kp + feat + labels]
    df_out = df_out[meta + kp + feat + other + labels]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[4/4] 저장: {output_path}")
    df_out.to_csv(output_path, index=False)

    # 요약 통계
    new_feats = [c for c in ["AHSSC", "AHSSC_x"] if c in df_out.columns]
    print("\n─── 완료 ──────────────────────────────────────────────────")
    print(f"  출력 rows : {len(df_out):,}")
    print(f"  출력 cols : {len(df_out.columns)}  (입력 {len(df.columns)} + 신규 {len(new_feats)}: {new_feats})")
    if "label" in df_out.columns:
        print(f"  label 분포: {df_out['label'].value_counts().sort_index().to_dict()}")
    if "label_3class" in df_out.columns:
        print(f"  label_3class: {df_out['label_3class'].value_counts().sort_index().to_dict()}")

    # 품질 지표 출력
    if "VHSSC" in df_out.columns and "label" in df_out.columns:
        v0 = df_out[df_out["label"]==0]["VHSSC"]
        v1 = df_out[df_out["label"]==1]["VHSSC"]
        pooled = ((v0.var() + v1.var()) / 2) ** 0.5
        cohens_d = abs(v1.mean() - v0.mean()) / pooled if pooled > 0 else 0
        print(f"  VHSSC Cohen's d  : {cohens_d:.4f}")

    if "AHSSC" in df_out.columns:
        a_std = df_out[df_out["label"]==0]["AHSSC"].std() if "label" in df_out.columns else df_out["AHSSC"].std()
        print(f"  AHSSC noise std  : {a_std:.4f}")
        print(f"  AHSSC range      : [{df_out['AHSSC'].min():.2f}, {df_out['AHSSC'].max():.2f}]")
    print("─────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline D: 키포인트 정제 + 가속도 피처 → final_dataset_filtered.csv"
    )
    p.add_argument("--input",  "-i", default="dataset/final_dataset.csv")
    p.add_argument("--output", "-o", default="dataset/final_dataset_filtered.csv")
    # One-Euro
    p.add_argument("--min-cutoff",  type=float, default=0.5,
                   help="One-Euro min_cutoff [Hz] (default: 0.5)")
    p.add_argument("--beta",        type=float, default=0.3,
                   help="One-Euro beta (default: 0.3)")
    p.add_argument("--d-cutoff",    type=float, default=1.0,
                   help="One-Euro d_cutoff [Hz] (default: 1.0)")
    # EMA
    p.add_argument("--conf-alpha",  type=float, default=0.5,
                   help="EMA alpha for confidence (default: 0.5)")
    # 저신뢰도 마스킹
    p.add_argument("--conf-thr",    type=float, default=0.15,
                   help="Confidence threshold for masking (default: 0.15)")
    # 미분 스무딩
    p.add_argument("--ema-deriv-alpha", type=float, default=0.4,
                   help="EMA alpha for VHSSC smoothing before AHSSC (default: 0.4, 0=off)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
