#!/usr/bin/env python3
"""
analyze_and_split_dataset.py

final_dataset.csv 기반 분석 + 정제 + 분할 파이프라인

Steps:
  1. 방향별(BY/FY/SY/N) 키포인트 신뢰도 분석
  2. 비디오 단위 이상치(outlier) 제거 → 목표 유지율 ~85%
  3. 키포인트 선정 3티어 분석 (전체 문서화만, CSV는 원본 유지)
  4. train / val / test 분할 (7:2:1, stratified by direction)
  5. 시각화 및 리포트 생성
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────
def _setup_korean_font():
    candidates = ["NanumGothic", "NanumBarunGothic", "Baekmuk Batang", "Baekmuk Gulim", "UnDotum"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    import os
    for path in [
        "/usr/share/fonts/truetype/baekmuk/batang.ttf",
        "/usr/share/fonts/truetype/baekmuk/gulim.ttf",
        "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
    ]:
        if os.path.exists(path):
            fe = fm.FontEntry(fname=path, name="KoreanFont")
            fm.fontManager.ttflist.insert(0, fe)
            matplotlib.rcParams["font.family"] = "KoreanFont"
            matplotlib.rcParams["axes.unicode_minus"] = False
            return path
    return None

_setup_korean_font()

# ── 키포인트 메타 (MoveNet 17-point) ─────────────────────────────────────────
KP_NAMES = {
    0:  "코",        1:  "왼눈",    2:  "오른눈",
    3:  "왼귀",      4:  "오른귀",  5:  "왼어깨",
    6:  "오른어깨",  7:  "왼팔꿈치", 8:  "오른팔꿈치",
    9:  "왼손목",    10: "오른손목", 11: "왼골반",
    12: "오른골반",  13: "왼무릎",  14: "오른무릎",
    15: "왼발목",    16: "오른발목",
}
KP_EN = {
    0: "nose",       1: "L-eye",     2: "R-eye",
    3: "L-ear",      4: "R-ear",     5: "L-shoulder",
    6: "R-shoulder", 7: "L-elbow",   8: "R-elbow",
    9: "L-wrist",    10: "R-wrist",  11: "L-hip",
    12: "R-hip",     13: "L-knee",   14: "R-knee",
    15: "L-ankle",   16: "R-ankle",
}
N_KP = 17
KP_S_COLS = [f"kp{i}_s" for i in range(N_KP)]

DIRECTION_LABELS = {
    "BY": "후면(Back)", "FY": "전면(Front)", "SY": "측면(Side)", "N": "비낙상(Normal)"
}
PALETTE = {"BY": "#e74c3c", "FY": "#3498db", "SY": "#2ecc71", "N": "#95a5a6"}

# ── 키포인트 3-티어 정의 ────────────────────────────────────────────────────
# MoveNet은 항상 17개를 출력 → 학습 시 어느 컬럼을 입력에 포함할지만 결정
# 파생 피처(HSSC/RWHC/VHSSC)는 CSV에 이미 계산된 별도 컬럼이므로
# 원시 kp 컬럼을 제거해도 파생 피처는 독립적으로 사용 가능

KP_TIERS = {
    "minimal": {
        "kps": [0, 5, 6, 11, 12],
        "label": "최소 세트 (5개)",
        "desc": (
            "코·양어깨·양골반만 사용. 낙상의 핵심 신호인 머리 높이(VHSSC의 원천)와 "
            "어깨→골반 기울기 벡터를 포착. HSSC/VHSSC/AHSSC가 이 5점을 이미 집약하므로 "
            "파생 피처와 함께 쓸 때 가장 경량."
        ),
        "remove": [1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16],
        "remove_reason": {
            "1,2 (눈)":    "코와 위치 거의 동일, HSSC 기여분 미미, 후면·측면 신뢰도 낮음",
            "3,4 (귀)":    "눈과 동일 이유, 낙상 방향 식별에 불필요",
            "7,8 (팔꿈치)": "낙상 동작의 주 신호는 상체 무게중심 이동 → 팔꿈치 정보 중복",
            "9,10 (손목)": "신뢰도 최하 (~0.33), 낙상 중 위치 불규칙해 오히려 노이즈",
            "13,14 (무릎)": "VHSSC 상관 낮음, 낙상 초기 단계에 무릎 변화 미미",
            "15,16 (발목)": "VHSSC 상관 낮음, 바닥 접촉 이후에야 변화 → 예측에 늦음",
        },
    },
    "recommended": {
        "kps": [0, 5, 6, 7, 8, 11, 12],
        "label": "권장 세트 (7개)",
        "desc": (
            "최소 세트에 팔꿈치 추가. 전면 낙상 시 팔을 앞으로 뻗는 패턴을 포착하여 "
            "전면/후면 방향 구별력 향상. 실험 성능과 모델 크기의 균형점."
        ),
        "remove": [1, 2, 3, 4, 9, 10, 13, 14, 15, 16],
        "remove_reason": {
            "1,2 (눈)":    "코와 위치 거의 동일, HSSC 기여분 미미",
            "3,4 (귀)":    "신뢰도·분별력 모두 낮음",
            "9,10 (손목)": "신뢰도 최하, 팔꿈치가 이미 팔 궤적 대리",
            "13-16 (하체)": "VHSSC 상관 낮음 (< 0.05)",
        },
    },
    "current": {
        "kps": list(range(13)),   # kp0~kp12
        "label": "현재 세트 (13개, 1차 필터 후)",
        "desc": (
            "VHSSC 상관 분석으로 kp13~kp16(무릎·발목)만 제거한 세트. "
            "기준 모델 대비 비교용으로 사용."
        ),
        "remove": [13, 14, 15, 16],
        "remove_reason": {
            "13,14 (무릎)": "낙상 프레임 VHSSC 상관 < 0.05",
            "15,16 (발목)": "낙상 프레임 VHSSC 상관 < 0.05",
        },
    },
}


def get_direction(vid: str) -> str:
    parts = vid.split("_")
    return parts[3] if len(parts) >= 4 else "UNKNOWN"


# ════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════════════════════

def load_data(path: Path) -> pd.DataFrame:
    print(f"[1/5] 로딩: {path}")
    df = pd.read_csv(path, low_memory=False)
    df["direction"] = df["video_id"].apply(get_direction)
    print(f"      rows={len(df):,}  videos={df['video_id'].nunique():,}")
    vdc = df.drop_duplicates("video_id")["direction"].value_counts()
    for d in ["BY", "FY", "SY", "N"]:
        print(f"      {DIRECTION_LABELS[d]}: {vdc.get(d, 0)} 비디오")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. 키포인트 신뢰도 분석 + 키포인트 선정 3-티어 시각화
# ════════════════════════════════════════════════════════════════════════════

def analyze_and_select_keypoints(df: pd.DataFrame, out_dir: Path) -> dict:
    print("[2/5] 신뢰도 분석 + 키포인트 선정...")

    avail_s = [c for c in KP_S_COLS if c in df.columns]
    directions = ["BY", "FY", "SY", "N"]

    # 방향 × kp 평균 신뢰도 matrix
    conf_matrix = {}
    for d in directions:
        sub = df[df["direction"] == d][avail_s]
        conf_matrix[d] = sub.mean().values
    overall_conf = df[avail_s].mean()
    dir_mean = {d: conf_matrix[d].mean() for d in directions}

    # VHSSC 상관 (낙상 프레임)
    fall_df = df[df["label"] == 1]
    corr_vhssc = {}
    if "VHSSC" in fall_df.columns:
        for i in range(N_KP):
            col = f"kp{i}_y"
            if col in fall_df.columns:
                c = fall_df[[col, "VHSSC"]].dropna().corr().iloc[0, 1]
                corr_vhssc[i] = float(abs(c)) if not np.isnan(c) else 0.0

    # ── 1. 히트맵: 방향 × 키포인트 신뢰도 ────────────────────────────────
    conf_df = pd.DataFrame(
        conf_matrix,
        index=[f"kp{i} {KP_NAMES[i]}" for i in range(len(avail_s))],
    )
    fig, ax = plt.subplots(figsize=(9, 12))
    sns.heatmap(
        conf_df, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.0, vmax=1.0, linewidths=0.5, ax=ax,
    )
    ax.set_title("방향별 키포인트 평균 신뢰도", fontsize=14, pad=12)
    ax.set_xlabel("낙상 방향", fontsize=12)
    ax.set_ylabel("키포인트", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "conf_heatmap.png", dpi=150)
    plt.close(fig)

    # ── 2. 방향별 전체 평균 신뢰도 bar ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        [DIRECTION_LABELS[d] for d in directions],
        [dir_mean[d] for d in directions],
        color=[PALETTE[d] for d in directions],
        edgecolor="white", linewidth=1.2,
    )
    for bar, d in zip(bars, directions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{dir_mean[d]:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("평균 신뢰도", fontsize=12)
    ax.set_title("방향별 전체 평균 키포인트 신뢰도", fontsize=13)
    ax.axhline(0.30, color="red", linestyle="--", linewidth=1, label="임계값 0.30")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "conf_by_direction.png", dpi=150)
    plt.close(fig)

    # ── 3. 키포인트 선정 3-티어 비교 차트 ───────────────────────────────
    tier_keys = ["minimal", "recommended", "current"]
    tier_colors = {"minimal": "#e74c3c", "recommended": "#f39c12", "current": "#3498db"}

    fig, axes = plt.subplots(3, 1, figsize=(15, 13))
    conf_vals = [overall_conf.get(f"kp{i}_s", 0) for i in range(N_KP)]
    corr_vals = [corr_vhssc.get(i, 0) for i in range(N_KP)]

    for ax_idx, (ax, metric, vals, ylabel, thr) in enumerate(zip(
        axes,
        ["신뢰도", "VHSSC 상관", "신뢰도 (권장 세트 기준)"],
        [conf_vals, corr_vals, conf_vals],
        ["평균 신뢰도", "|VHSSC 상관계수|", "평균 신뢰도"],
        [0.30, 0.05, 0.30],
    )):
        if ax_idx == 2:
            # 3번째: 티어별 포함 여부 표시
            tier_key = "recommended"
            tier_kps = set(KP_TIERS[tier_key]["kps"])
            bar_colors = ["#2ecc71" if i in tier_kps else "#e74c3c" for i in range(N_KP)]
        else:
            tier_kps = set(KP_TIERS["minimal"]["kps"])
            bar_colors = ["#2ecc71" if i in tier_kps else
                          ("#f39c12" if i in set(KP_TIERS["recommended"]["kps"]) else "#e74c3c")
                          for i in range(N_KP)]

        bars = ax.bar(range(N_KP), vals, color=bar_colors, edgecolor="white", linewidth=0.8)
        ax.axhline(thr, color="red", linestyle="--", linewidth=1.2, label=f"임계값 {thr}")
        ax.set_xticks(range(N_KP))
        ax.set_xticklabels([f"kp{i}\n{KP_NAMES[i]}" for i in range(N_KP)],
                           rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6)

    axes[0].set_title(
        "키포인트 선정 분석\n초록=최소세트(5개), 주황=권장세트 추가(+2개), 빨강=제거", fontsize=12)
    axes[1].set_title("낙상 프레임 VHSSC 상관계수 (낮을수록 낙상 정보 희박)", fontsize=12)
    axes[2].set_title("권장 세트(7개) 선택 결과: 초록=포함, 빨강=제거", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "keypoint_selection.png", dpi=150)
    plt.close(fig)

    print(f"  방향별 평균 신뢰도: { {d: f'{v:.4f}' for d, v in dir_mean.items()} }")
    for i in range(N_KP):
        flag = "  ◀" if overall_conf.get(f"kp{i}_s", 1) < 0.35 else ""
        print(f"    kp{i:2d} {KP_NAMES[i]:8s}: conf={overall_conf.get(f'kp{i}_s',0):.4f}  "
              f"corr={corr_vhssc.get(i,0):.4f}{flag}")

    return {
        "conf_matrix": {d: v.tolist() for d, v in conf_matrix.items()},
        "overall_conf": {c: float(v) for c, v in overall_conf.items()},
        "dir_mean": dir_mean,
        "corr_vhssc": corr_vhssc,
        "tiers": {
            k: {"kps": v["kps"], "label": v["label"], "remove": v["remove"]}
            for k, v in KP_TIERS.items()
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. 비디오 단위 이상치 제거 (목표: ~85% 유지)
# ════════════════════════════════════════════════════════════════════════════

def remove_video_outliers(
    df: pd.DataFrame, out_dir: Path,
    z_thresh: float = 2.0,
    min_conf: float = 0.28,
    target_keep: float = 0.85,
) -> tuple[pd.DataFrame, dict]:
    """
    3단계 필터:
      A. 절대 신뢰도 필터  — 비디오 평균 신뢰도 < min_conf
      B. z-score 필터      — 방향 내 z > z_thresh (conf_mean, rwhc_std, vhssc_std)
      C. target_keep 초과시 z_thresh 자동 완화 (목표 유지율 보장)
    """
    print(f"[3/5] 이상치 제거 (목표 유지율 {target_keep*100:.0f}%, "
          f"z_thresh={z_thresh}, min_conf={min_conf})...")

    avail_s = [c for c in KP_S_COLS if c in df.columns]
    grp = df.groupby("video_id")

    video_stats = pd.DataFrame({
        "direction":   df.groupby("video_id")["direction"].first(),
        "label_mean":  grp["label"].mean(),
        "conf_mean":   grp[avail_s].mean().mean(axis=1),
        "rwhc_std":    grp["RWHC"].std() if "RWHC" in df.columns else pd.Series(0, index=grp.groups),
        "vhssc_std":   grp["VHSSC"].std() if "VHSSC" in df.columns else pd.Series(0, index=grp.groups),
        "frame_count": grp.size(),
    })

    outlier_flags = pd.Series(False, index=video_stats.index)
    outlier_reasons: dict[str, list[str]] = {}

    # A. 절대 신뢰도 필터
    low_conf_mask = video_stats["conf_mean"] < min_conf
    for vid in video_stats.index[low_conf_mask]:
        outlier_flags[vid] = True
        outlier_reasons.setdefault(vid, []).append(
            f"conf_mean={video_stats.loc[vid,'conf_mean']:.3f}<{min_conf}"
        )

    # B. z-score 필터 (방향별)
    for d in video_stats["direction"].unique():
        sub = video_stats[video_stats["direction"] == d]
        for col in ["conf_mean", "rwhc_std", "vhssc_std"]:
            vals = sub[col].fillna(sub[col].median())
            z = np.abs(stats.zscore(vals))
            bad = sub.index[z > z_thresh]
            for vid in bad:
                outlier_flags[vid] = True
                loc = sub.index.get_loc(vid)
                outlier_reasons.setdefault(vid, []).append(
                    f"{col}_z={z[loc]:.2f}"
                )

    removed_pct = outlier_flags.sum() / len(video_stats)
    keep_pct = 1 - removed_pct
    print(f"  필터 결과: 유지 {keep_pct*100:.1f}%  (목표 {target_keep*100:.0f}%)")

    # C. 목표 유지율보다 너무 많이 제거된 경우 → 완화 (안전장치)
    if keep_pct < target_keep - 0.03:
        print(f"  ⚠ 과제거 감지. z_thresh를 {z_thresh} → {z_thresh+0.5:.1f}로 완화.")
        z_thresh += 0.5
        outlier_flags[:] = False
        outlier_reasons = {}
        low_conf_mask = video_stats["conf_mean"] < min_conf
        for vid in video_stats.index[low_conf_mask]:
            outlier_flags[vid] = True
            outlier_reasons.setdefault(vid, []).append(
                f"conf_mean={video_stats.loc[vid,'conf_mean']:.3f}<{min_conf}"
            )
        for d in video_stats["direction"].unique():
            sub = video_stats[video_stats["direction"] == d]
            for col in ["conf_mean", "rwhc_std", "vhssc_std"]:
                vals = sub[col].fillna(sub[col].median())
                z = np.abs(stats.zscore(vals))
                bad = sub.index[z > z_thresh]
                for vid in bad:
                    outlier_flags[vid] = True
                    loc = sub.index.get_loc(vid)
                    outlier_reasons.setdefault(vid, []).append(f"{col}_z={z[loc]:.2f}")

    n_removed = outlier_flags.sum()
    bad_vids = set(outlier_flags[outlier_flags].index)
    df_clean = df[~df["video_id"].isin(bad_vids)].copy()

    print(f"  제거 비디오: {n_removed}/{len(video_stats)} ({n_removed/len(video_stats)*100:.1f}%)")
    print(f"  유지 rows:   {len(df_clean):,} / {len(df):,} ({len(df_clean)/len(df)*100:.1f}%)")

    removed_by_dir = video_stats[outlier_flags]["direction"].value_counts()
    for d, cnt in removed_by_dir.items():
        total = (video_stats["direction"] == d).sum()
        print(f"    {DIRECTION_LABELS.get(d, d)}: {cnt}/{total} ({cnt/total*100:.1f}%)")

    # ── 산점도: conf_mean vs (rwhc_std / vhssc_std) ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (xcol, ycol) in zip(axes, [("conf_mean", "rwhc_std"), ("conf_mean", "vhssc_std")]):
        for d in ["BY", "FY", "SY", "N"]:
            sub = video_stats[video_stats["direction"] == d]
            kept    = sub[~outlier_flags[sub.index]]
            removed = sub[outlier_flags[sub.index]]
            ax.scatter(kept[xcol], kept[ycol], c=PALETTE[d], s=4, alpha=0.4,
                       label=DIRECTION_LABELS[d])
            ax.scatter(removed[xcol], removed[ycol], c=PALETTE[d], s=30,
                       marker="x", alpha=0.9)
        ax.axvline(min_conf, color="purple", linestyle=":", linewidth=1,
                   label=f"min_conf={min_conf}")
        ax.set_xlabel(xcol, fontsize=11)
        ax.set_ylabel(ycol, fontsize=11)
        ax.set_title(f"{xcol} vs {ycol}  (× = 제거)", fontsize=11)
        ax.legend(fontsize=8, markerscale=2)
    plt.suptitle("비디오 단위 이상치 탐지", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "outlier_scatter.png", dpi=150)
    plt.close(fig)

    # ── 방향별 제거 비율 bar ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    dirs = ["BY", "FY", "SY", "N"]
    remove_rates = []
    for d in dirs:
        total = (video_stats["direction"] == d).sum()
        rm = removed_by_dir.get(d, 0)
        remove_rates.append(rm / total * 100 if total > 0 else 0)
    bars = ax.bar([DIRECTION_LABELS[d] for d in dirs], remove_rates,
                  color=[PALETTE[d] for d in dirs], edgecolor="white")
    for bar, rate in zip(bars, remove_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("이상치 비율 (%)", fontsize=12)
    ax.set_title("방향별 이상치 제거 비율", fontsize=13)
    ax.set_ylim(0, max(remove_rates) * 1.25 + 1)
    plt.tight_layout()
    fig.savefig(out_dir / "outlier_by_direction.png", dpi=150)
    plt.close(fig)

    return df_clean, {
        "total_videos": int(len(video_stats)),
        "removed_videos": int(n_removed),
        "removed_pct": float(n_removed / len(video_stats) * 100),
        "keep_pct": float(len(df_clean) / len(df) * 100),
        "removed_by_direction": {k: int(v) for k, v in removed_by_dir.items()},
        "z_thresh_used": z_thresh,
        "min_conf_used": min_conf,
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. Train / Val / Test 분할 (7:2:1, stratified by direction)
# ════════════════════════════════════════════════════════════════════════════

def split_dataset(df: pd.DataFrame, out_dir: Path,
                  ratios: tuple = (0.7, 0.2, 0.1), seed: int = 42) -> dict:
    """
    비디오 단위 stratified split. 컬럼은 원본 전체 유지.
    학습 시 필요한 키포인트만 추출하는 것은 학습 스크립트에서 처리.
    """
    print(f"[4/5] 데이터셋 분할 (train:val:test = {ratios})...")

    rng = np.random.default_rng(seed)
    directions = ["BY", "FY", "SY", "N"]
    vid_dir = df.drop_duplicates("video_id").set_index("video_id")["direction"]

    train_vids, val_vids, test_vids = [], [], []
    for d in directions:
        vids = sorted(vid_dir[vid_dir == d].index.tolist())
        rng.shuffle(vids)
        n = len(vids)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        train_vids.extend(vids[:n_tr])
        val_vids.extend(vids[n_tr:n_tr + n_va])
        test_vids.extend(vids[n_tr + n_va:])

    split_stats = {}
    for split_name, vid_list in [("train", train_vids), ("val", val_vids), ("test", test_vids)]:
        sub = df[df["video_id"].isin(vid_list)].copy()
        path = out_dir / f"{split_name}.csv"
        sub.to_csv(path, index=False)
        dd = sub.drop_duplicates("video_id")["direction"].value_counts().to_dict()
        ld = sub["label"].value_counts().to_dict() if "label" in sub.columns else {}
        split_stats[split_name] = {
            "videos": len(vid_list), "rows": len(sub),
            "dir_distribution": dd, "label_distribution": ld,
        }
        print(f"  {split_name:5s}: {len(vid_list):5d} videos  {len(sub):9,} rows  → {path.name}")

    # ── 파이 차트 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, sn in zip(axes, ["train", "val", "test"]):
        dist = split_stats[sn]["dir_distribution"]
        dirs = [d for d in directions if d in dist]
        ax.pie(
            [dist[d] for d in dirs],
            labels=[DIRECTION_LABELS[d] for d in dirs],
            colors=[PALETTE[d] for d in dirs],
            autopct="%1.1f%%", startangle=140,
            textprops={"fontsize": 9},
        )
        ax.set_title(
            f"{sn.upper()}\n{split_stats[sn]['videos']} videos / {split_stats[sn]['rows']:,} rows",
            fontsize=11)
    plt.suptitle("Train / Val / Test 방향별 분포", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "split_distribution.png", dpi=150)
    plt.close(fig)

    return split_stats


# ════════════════════════════════════════════════════════════════════════════
# 5. 리포트 생성
# ════════════════════════════════════════════════════════════════════════════

def write_report(
    out_dir: Path, conf_result: dict, outlier_result: dict,
    split_result: dict, input_rows: int, clean_rows: int,
) -> None:
    print("[5/5] 리포트 생성...")

    def _conf(i):
        return conf_result["overall_conf"].get(f"kp{i}_s", 0)

    def _corr(i):
        return conf_result["corr_vhssc"].get(i, 0)

    dir_conf_rows = "\n".join(
        f"| {DIRECTION_LABELS[d]:<14} | {conf_result['dir_mean'][d]:.4f} |"
        for d in ["BY", "FY", "SY", "N"]
    )

    kp_table_rows = "\n".join(
        f"| kp{i:2d} | {KP_NAMES[i]:<8} | {KP_EN[i]:<12} | {_conf(i):.4f} | {_corr(i):.4f} |"
        f" {'최소' if i in KP_TIERS['minimal']['kps'] else ('권장+' if i in KP_TIERS['recommended']['kps'] else '제거')} |"
        for i in range(N_KP)
    )

    tier_sections = ""
    for tk in ["minimal", "recommended", "current"]:
        t = KP_TIERS[tk]
        reasons = "\n".join(
            f"  - **{part}**: {reason}"
            for part, reason in t["remove_reason"].items()
        )
        tier_sections += f"""
#### {t['label']} — kp{{{', '.join(map(str, t['kps']))}}}

{t['desc']}

**제거 키포인트** ({len(t['remove'])}개):
{reasons}

"""

    split_rows = "\n".join(
        f"| {sn:5s} | {split_result[sn]['videos']:5d} |"
        f" {split_result[sn]['rows']:9,} |"
        f" BY:{split_result[sn]['dir_distribution'].get('BY',0)}"
        f" FY:{split_result[sn]['dir_distribution'].get('FY',0)}"
        f" SY:{split_result[sn]['dir_distribution'].get('SY',0)}"
        f" N:{split_result[sn]['dir_distribution'].get('N',0)} |"
        f" 낙상:{split_result[sn]['label_distribution'].get(1,0):,}"
        f" / 비낙상:{split_result[sn]['label_distribution'].get(0,0):,} |"
        for sn in ["train", "val", "test"]
    )

    removed_dir_rows = "\n".join(
        f"- {DIRECTION_LABELS.get(k, k)}: {v}개 제거"
        for k, v in outlier_result["removed_by_direction"].items()
    )

    report = f"""# 낙상 감지 데이터셋 분석 리포트

**생성일**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**원본 파일**: `dataset/final_dataset.csv`
**원본 rows**: {input_rows:,}
**정제 후 rows**: {clean_rows:,} (유지율 {clean_rows/input_rows*100:.1f}%)

---

## 1. 데이터셋 구조

### 1.1 video_id 명명 규칙

```
{{ID}}_{{피험자}}_{{동작}}_{{방향}}_{{카메라}}
예: 00050_H_A_BY_C2

방향 코드:
  BY = 후면(Back-facing) 낙상
  FY = 전면(Front-facing) 낙상
  SY = 측면(Side-facing) 낙상
  N  = 비낙상(Normal)
```

### 1.2 방향별 키포인트 평균 신뢰도

| 방향 | 전체 평균 신뢰도 |
|------|----------------|
{dir_conf_rows}

> **실험적 확인**: 후면(BY) 낙상의 신뢰도가 가장 낮음.
> 카메라에서 등을 보이고 쓰러지면 얼굴·어깨 전면 포인트가 가려지기 때문.
> 후면 데이터는 특히 데이터 증강(augmentation) 또는 가중 샘플링이 필요할 수 있음.

---

## 2. 키포인트별 상세 분석

| kp | 한국어 | 영어 | 평균신뢰도 | VHSSC상관 | 선정 |
|----|--------|------|-----------|----------|------|
{kp_table_rows}

> - **최소**: 최소 세트(5개)에 포함
> - **권장+**: 권장 세트에서 추가 포함(7개)
> - **제거**: 전체 3티어에서 제외 권장

---

## 3. 키포인트 선정 (3-Tier 제안)

MoveNet은 항상 17개 키포인트를 출력하므로, **학습 시 CSV에서 사용할 컬럼만 선택**하면 됨.
파생 피처(HSSC_y, HSSC_x, RWHC, VHSSC)는 별도 컬럼으로 이미 계산되어 있어
원시 kp 컬럼을 제거해도 독립적으로 사용 가능.

### 제거 기준

1. **신뢰도** — 전체 평균 < 0.30 (포즈 추정 자체가 불안정)
2. **낙상 식별력** — 낙상 프레임에서 VHSSC와 |상관계수| < 0.05 (낙상 동작 정보 희박)
3. **중복성** — 가까운 관절에 의해 이미 대리되는 키포인트
{tier_sections}

### 권장 실험 순서

```
1단계: 최소 세트(5개) → 기준 성능 측정
2단계: 권장 세트(7개) → 팔꿈치 추가 효과 확인
3단계: 현재 세트(13개) → 성능 비교
→ 성능 차이가 미미하면 최소 세트 채택 (모델 경량화 우선)
```

---

## 4. 이상치(Outlier) 제거

### 4.1 제거 기준

| 기준 | 설명 |
|------|------|
| 절대 신뢰도 필터 | 비디오 평균 신뢰도 < **{outlier_result['min_conf_used']}** |
| z-score 필터 | 방향 내 z-score > **{outlier_result['z_thresh_used']}** |
| 적용 지표 | `conf_mean`, `rwhc_std`, `vhssc_std` |
| 처리 단위 | 비디오 단위 (비디오 전체 프레임 일괄 제거) |

### 4.2 제거 결과

- 제거 비디오: **{outlier_result['removed_videos']}** / {outlier_result['total_videos']} ({outlier_result['removed_pct']:.1f}%)
- 원본 유지율: **{outlier_result['keep_pct']:.1f}%** (목표 ≤ 85%)
- 방향별 제거:
{removed_dir_rows}

> **의의**: 이상치는 낮은 신뢰도·비정상적 RWHC·VHSSC 분산을 가진 비디오로,
> 학습 데이터에 포함 시 그라디언트 오염 가능성이 있음.

---

## 5. Train / Val / Test 분할

### 5.1 분할 방법

- **단위**: 비디오 단위 stratified split (방향별 동일 비율 유지)
- **비율**: train **70%** / val **20%** / test **10%**
- **seed**: 42
- **컬럼**: 원본 전체 컬럼 유지 (학습 스크립트에서 티어별 컬럼 선택)

### 5.2 분할 결과

| split | 비디오 | rows | 방향 분포 | 라벨 분포 |
|-------|--------|------|-----------|-----------|
{split_rows}

---

## 6. 생성 파일

| 파일 | 설명 |
|------|------|
| `dataset/splits/train.csv` | 학습용 (70%) |
| `dataset/splits/val.csv`   | 검증용 (20%) |
| `dataset/splits/test.csv`  | 테스트용 (10%) |
| `docs/analysis/conf_heatmap.png` | 방향×키포인트 신뢰도 히트맵 |
| `docs/analysis/conf_by_direction.png` | 방향별 평균 신뢰도 |
| `docs/analysis/keypoint_selection.png` | 키포인트 3-티어 선정 차트 |
| `docs/analysis/outlier_scatter.png` | 이상치 산점도 |
| `docs/analysis/outlier_by_direction.png` | 방향별 이상치 비율 |
| `docs/analysis/split_distribution.png` | 분할 분포 파이차트 |

---

## 7. 다음 단계 권장사항

1. **기준 실험**: GRU/TCN 학습 시 `train.csv`에서 권장 세트(7개 kp + 파생 피처 4개) 사용
2. **키포인트 ablation**: 최소→권장→현재 순서로 val 성능 비교
3. **test.csv는 최종 1회만 사용** (하이퍼파라미터 탐색에 val 전용으로 유지)
4. **후면(BY) 증강 검토**: 신뢰도가 낮아 모델이 취약할 가능성
   - 수평 flip 후 방향 라벨은 유지 (카메라 좌우 반전이므로 BY는 BY)
5. **클래스 불균형 대응**: 낙상(label=1) 비율 ~8% → 가중 손실함수 또는 오버샘플링
"""

    report_path = out_dir / "dataset_analysis_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  → {report_path}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      "-i", default="dataset/final_dataset.csv")
    p.add_argument("--split-out",        default="dataset/splits")
    p.add_argument("--report-out",       default="docs/analysis")
    p.add_argument("--z-thresh",   type=float, default=2.0)
    p.add_argument("--min-conf",   type=float, default=0.28,
                   help="비디오 평균 신뢰도 절대 임계값 (default: 0.28)")
    p.add_argument("--target-keep", type=float, default=0.85,
                   help="목표 유지율 (default: 0.85)")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    split_dir  = Path(args.split_out)
    report_dir = Path(args.report_out)
    split_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.input))
    input_rows = len(df)

    conf_result = analyze_and_select_keypoints(df, report_dir)

    df_clean, outlier_result = remove_video_outliers(
        df, report_dir,
        z_thresh=args.z_thresh,
        min_conf=args.min_conf,
        target_keep=args.target_keep,
    )
    clean_rows = len(df_clean)

    split_result = split_dataset(df_clean, split_dir, seed=args.seed)

    write_report(report_dir, conf_result, outlier_result, split_result, input_rows, clean_rows)

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    summary = {
        "conf_analysis": conf_result,
        "outlier_removal": outlier_result,
        "split_stats": split_result,
        "keypoint_tiers": {
            k: {"kps": v["kps"], "remove": v["remove"], "label": v["label"]}
            for k, v in KP_TIERS.items()
        },
    }
    (report_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, cls=_NpEncoder), encoding="utf-8"
    )

    print(f"\n=== 완료 ===")
    print(f"  정제: {input_rows:,} → {clean_rows:,} rows ({clean_rows/input_rows*100:.1f}% 유지)")
    print(f"  분할 CSV: {split_dir}/")
    print(f"  리포트:   {report_dir}/")


if __name__ == "__main__":
    main()
