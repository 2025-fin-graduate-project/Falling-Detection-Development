"""
카메라 각도(C1~C8)별 키포인트 신뢰도 분석
- 카메라별 전체 평균 신뢰도
- 카메라 × 방향 교차 분석
- 카메라별 낙상/비낙상 신뢰도 비교
- 제거 후보 카메라 추천
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────
def _setup_korean_font():
    candidates = ["NanumGothic", "NanumBarunGothic", "Baekmuk Batang", "Baekmuk Gulim", "UnDotum"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    for path in [
        "/usr/share/fonts/truetype/baekmuk/batang.ttf",
        "/usr/share/fonts/truetype/baekmuk/gulim.ttf",
        "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
    ]:
        if os.path.exists(path):
            fe = fm.FontEntry(fname=path, name="KoreanFont")
            fm.fontManager.ttflist.insert(0, fe)
            plt.rcParams["font.family"] = "KoreanFont"
            plt.rcParams["axes.unicode_minus"] = False
            return path
    return None

font_path = _setup_korean_font()
if font_path:
    print(f"한글 폰트 설정 완료: {font_path}")
else:
    print("경고: 한글 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "dataset", "final_dataset.csv")
OUT_DIR = os.path.join(ROOT, "docs", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

KP_NAMES = [
    "nose", "L-eye", "R-eye", "L-ear", "R-ear",
    "L-sho", "R-sho", "L-elb", "R-elb",
    "L-wri", "R-wri", "L-hip", "R-hip",
    "L-kne", "R-kne", "L-ank", "R-ank",
]
KP_SCORE_COLS = [f"kp{i}_s" for i in range(17)]
DIR_KO = {"BY": "후면(BY)", "FY": "전면(FY)", "SY": "측면(SY)", "N": "비낙상(N)"}

print("데이터 로딩 중 ...")
df = pd.read_csv(CSV_PATH)
df["camera"] = df["video_id"].str.split("_").str[-1]
df["direction"] = df["video_id"].str.split("_").str[-2]
df["conf_mean"] = df[KP_SCORE_COLS].mean(axis=1)

cameras = sorted(df["camera"].unique())
directions = ["BY", "FY", "SY", "N"]
print(f"카메라: {cameras} | 행 수: {len(df):,}")

# ── 1. 카메라별 키포인트 신뢰도 행렬 ────────────────────────────────────────────
print("카메라별 키포인트 신뢰도 계산 중 ...")
cam_kp = (
    df.groupby("camera")[KP_SCORE_COLS].mean()
    .rename(columns=dict(zip(KP_SCORE_COLS, KP_NAMES)))
)
cam_kp_mean = df.groupby("camera")["conf_mean"].mean().rename("overall_mean")

# ── 2. 카메라 × 방향 교차 행렬 ──────────────────────────────────────────────────
cam_dir = df.groupby(["camera", "direction"])["conf_mean"].mean().unstack("direction")

# ── 3. 카메라 × 라벨(낙상/비낙상) ────────────────────────────────────────────────
cam_label = df.groupby(["camera", "label"])["conf_mean"].mean().unstack("label")
cam_label.columns = ["비낙상(0)", "낙상(1)"]

# ── 4. 카메라별 낙상 프레임에서 키포인트 신뢰도 ──────────────────────────────────
fall_df = df[df["label"] == 1]
cam_kp_fall = (
    fall_df.groupby("camera")[KP_SCORE_COLS].mean()
    .rename(columns=dict(zip(KP_SCORE_COLS, KP_NAMES)))
)

# ─────────────────────────────────────────────────────────────────────────────
#  시각화
# ─────────────────────────────────────────────────────────────────────────────

# ── Fig 1: 카메라별 전체 평균 신뢰도 바 차트 ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(cameras)))
bars = ax.bar(cameras, cam_kp_mean[cameras], color=colors, edgecolor="white", width=0.6)
ax.axhline(cam_kp_mean.mean(), color="gray", linestyle="--", linewidth=1, label=f"전체 평균 {cam_kp_mean.mean():.4f}")
for bar, val in zip(bars, cam_kp_mean[cameras]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.4f}",
            ha="center", va="bottom", fontsize=8)
ax.set_ylim(0.30, 0.50)
ax.set_title("카메라별 전체 평균 키포인트 신뢰도", fontsize=13, weight="bold")
ax.set_xlabel("카메라")
ax.set_ylabel("평균 신뢰도 (conf_mean)")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "camera_conf_overall.png"), dpi=150)
plt.close()
print("  → camera_conf_overall.png 저장")

# ── Fig 2: 카메라 × 키포인트 히트맵 ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(cam_kp.values, aspect="auto", cmap="RdYlGn", vmin=0.28, vmax=0.58)
ax.set_xticks(range(17))
ax.set_xticklabels(KP_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(cameras)))
ax.set_yticklabels(cameras)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="평균 신뢰도")
for i in range(len(cameras)):
    for j in range(17):
        ax.text(j, i, f"{cam_kp.values[i, j]:.2f}", ha="center", va="center",
                fontsize=6, color="black")
ax.set_title("카메라 × 키포인트 신뢰도 히트맵", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "camera_kp_heatmap.png"), dpi=150)
plt.close()
print("  → camera_kp_heatmap.png 저장")

# ── Fig 3: 카메라 × 방향 신뢰도 히트맵 ───────────────────────────────────────────
avail_dirs = [d for d in directions if d in cam_dir.columns]
plot_data = cam_dir[avail_dirs]
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(plot_data.values, aspect="auto", cmap="RdYlGn", vmin=0.30, vmax=0.50)
ax.set_xticks(range(len(avail_dirs)))
ax.set_xticklabels([DIR_KO.get(d, d) for d in avail_dirs], fontsize=10)
ax.set_yticks(range(len(cameras)))
ax.set_yticklabels(cameras)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="평균 신뢰도")
for i in range(len(cameras)):
    for j, d in enumerate(avail_dirs):
        val = plot_data.iloc[i, j]
        ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=8,
                color="black" if 0.35 < val < 0.48 else "white")
ax.set_title("카메라 × 방향별 평균 키포인트 신뢰도", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "camera_dir_heatmap.png"), dpi=150)
plt.close()
print("  → camera_dir_heatmap.png 저장")

# ── Fig 4: 낙상 프레임만 — 카메라 × 키포인트 히트맵 ─────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(cam_kp_fall.values, aspect="auto", cmap="RdYlGn", vmin=0.20, vmax=0.52)
ax.set_xticks(range(17))
ax.set_xticklabels(KP_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(cameras)))
ax.set_yticklabels(cameras)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="평균 신뢰도 (낙상 프레임)")
for i in range(len(cameras)):
    for j in range(17):
        ax.text(j, i, f"{cam_kp_fall.values[i, j]:.2f}", ha="center", va="center",
                fontsize=6, color="black")
ax.set_title("카메라 × 키포인트 신뢰도 히트맵 (낙상 프레임만)", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "camera_kp_heatmap_fall.png"), dpi=150)
plt.close()
print("  → camera_kp_heatmap_fall.png 저장")

# ── Fig 5: 카메라별 낙상 vs 비낙상 신뢰도 비교 ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(cameras))
w = 0.35
ax.bar(x - w / 2, cam_label.loc[cameras, "비낙상(0)"], w, label="비낙상", color="#5BA4CF")
ax.bar(x + w / 2, cam_label.loc[cameras, "낙상(1)"], w, label="낙상", color="#E8604C")
ax.set_xticks(x)
ax.set_xticklabels(cameras)
ax.set_ylim(0.25, 0.55)
ax.set_title("카메라별 낙상 / 비낙상 프레임 신뢰도 비교", fontsize=13, weight="bold")
ax.set_ylabel("평균 신뢰도")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "camera_fall_vs_normal.png"), dpi=150)
plt.close()
print("  → camera_fall_vs_normal.png 저장")

# ─────────────────────────────────────────────────────────────────────────────
#  제거 후보 판단
# ─────────────────────────────────────────────────────────────────────────────
OVERALL_MEAN = cam_kp_mean.mean()
FALL_CONF_MEAN = cam_kp_fall.mean(axis=1)  # 카메라별 낙상 프레임 평균

# 임계값 기준
THRESH_OVERALL = OVERALL_MEAN - 0.015   # 전체 평균보다 1.5pp 낮은 카메라
THRESH_FALL    = FALL_CONF_MEAN.mean() - 0.015  # 낙상 프레임 평균보다 1.5pp 낮은 카메라

summary = {}
for cam in cameras:
    ov = cam_kp_mean[cam]
    fa = FALL_CONF_MEAN[cam]
    low_kps = cam_kp.loc[cam][cam_kp.loc[cam] < 0.30].index.tolist()
    low_fall_kps = cam_kp_fall.loc[cam][cam_kp_fall.loc[cam] < 0.25].index.tolist()
    flag_overall = ov < THRESH_OVERALL
    flag_fall = fa < THRESH_FALL
    summary[cam] = {
        "overall_conf": round(ov, 4),
        "fall_conf": round(fa, 4),
        "low_kps_overall": low_kps,       # 전체 신뢰도 < 0.30
        "low_kps_fall": low_fall_kps,     # 낙상 신뢰도 < 0.25
        "flag_overall": flag_overall,
        "flag_fall": flag_fall,
        "remove_candidate": flag_overall or flag_fall,
    }

# ─────────────────────────────────────────────────────────────────────────────
#  결과 출력
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("카메라별 신뢰도 요약")
print("=" * 70)
print(f"{'카메라':^6} | {'전체 conf':^10} | {'낙상 conf':^10} | {'낮은 kp(전체)':^20} | {'제거 후보':^8}")
print("-" * 70)
for cam, s in summary.items():
    flag = "★ 제거 후보" if s["remove_candidate"] else ""
    low_str = ", ".join(s["low_kps_overall"]) if s["low_kps_overall"] else "-"
    print(f"  {cam}   | {s['overall_conf']:^10.4f} | {s['fall_conf']:^10.4f} | {low_str:^20} | {flag}")

print(f"\n[기준] 전체 평균 threshold: {THRESH_OVERALL:.4f}  |  낙상 프레임 threshold: {THRESH_FALL:.4f}")

candidates = [c for c, s in summary.items() if s["remove_candidate"]]
print(f"\n제거 후보 카메라: {candidates if candidates else '없음'}")

# ── JSON 저장 ────────────────────────────────────────────────────────────────
def _native(obj):
    """numpy 타입을 JSON 직렬화 가능한 Python 타입으로 변환."""
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_native(v) for v in obj]
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    return obj

result = _native({
    "camera_overall_conf": cam_kp_mean.to_dict(),
    "camera_fall_conf": FALL_CONF_MEAN.to_dict(),
    "camera_dir_conf": cam_dir.to_dict(),
    "camera_kp_conf": cam_kp.to_dict(),
    "remove_candidates": candidates,
    "threshold_overall": round(THRESH_OVERALL, 4),
    "threshold_fall": round(THRESH_FALL, 4),
    "per_camera_summary": summary,
})
json_path = os.path.join(OUT_DIR, "camera_analysis.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n결과 JSON → {json_path}")
print("완료.")
