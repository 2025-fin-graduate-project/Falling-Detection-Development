#!/usr/bin/env python3
"""
UR-Fall Dataset Keypoint Extraction Pipeline

Usage:
    uv run python scripts/extract_urfall_keypoints.py \
        --data-dir dataset/urfall/UR_fall_detection_dataset_cam0_rgb \
        --ann-dir  dataset/urfall \
        --model    model/st_movenet_lightning_a100_heatmaps_256_int8.tflite \
        --output   dataset/urfall_keypoints.csv

실제 디렉토리 구조 (Kaggle 다운로드 기준):
    dataset/urfall/
      UR_fall_detection_dataset_cam0_rgb/
        fall-01-cam0-rgb/
          fall-01-cam0-rgb-001.png
          fall-01-cam0-rgb-002.png
          ...
        adl-01-cam0-rgb/
          ...
      urfall-cam0-falls.csv   ← 원본 사이트에서 별도 다운로드

어노테이션 CSV 컬럼: seq_name, frame, label, ...
  label -1 = non_fall (낙상 전 정상 동작)
  label  0 = falling  (낙하 중)
  label  1 = fallen   (바닥에 쓰러진 후)

출력 label 매핑:
  0 = non_fall  (-1 또는 ADL)
  1 = falling   (0)
  2 = fallen    (1)
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

NUM_KP = 17
TARGET_DIM = 256


# ── TFLite 모델 ───────────────────────────────────────────────────────────────

def load_interpreter(model_path: str):
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
    except Exception:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


# ── 히트맵 → 키포인트 디코딩 ──────────────────────────────────────────────────

def decode_heatmaps(heatmap_int8: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """[64,64,17] int8 → (17,3) float32  [y, x, score] 정규화 [0,1]"""
    heatmap = (heatmap_int8.astype(np.float32) - zero_point) * scale
    H, W, K = heatmap.shape
    keypoints = np.zeros((K, 3), dtype=np.float32)
    for k in range(K):
        ch_sig = 1.0 / (1.0 + np.exp(-heatmap[:, :, k]))
        flat_idx = int(np.argmax(ch_sig))
        row, col = divmod(flat_idx, W)
        keypoints[k] = [row / H, col / W, float(ch_sig[row, col])]
    return keypoints


# ── 단일 프레임 추론 ──────────────────────────────────────────────────────────

def infer_frame(interp, bgr: np.ndarray, in_det: dict, out_det: dict) -> np.ndarray:
    h, w = bgr.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    rgb = cv2.cvtColor(cv2.resize(bgr[top:top+side, left:left+side], (TARGET_DIM, TARGET_DIM)),
                       cv2.COLOR_BGR2RGB)
    interp.set_tensor(in_det["index"], np.expand_dims(rgb, 0).astype(np.uint8))
    interp.invoke()
    raw = interp.get_tensor(out_det["index"])[0]          # [64,64,17]
    return decode_heatmaps(raw, out_det["quantization"][0], out_det["quantization"][1])


# ── 어노테이션 로드 ───────────────────────────────────────────────────────────

def load_annotations(ann_path: Path) -> dict[tuple[str, int], int]:
    """
    urfall-cam0-falls.csv 파싱.
    반환: {(seq_name, frame_no): raw_label(-1/0/1)}
    """
    ann: dict[tuple[str, int], int] = {}
    with open(ann_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            seq_name = row[0].strip()   # e.g. "fall-01"
            try:
                frame = int(row[1])
                label = int(row[2])
            except ValueError:
                continue
            ann[(seq_name, frame)] = label
    return ann


# ── CSV 헤더/행 ───────────────────────────────────────────────────────────────

def make_header() -> list[str]:
    cols = ["video_id", "frame", "time_sec"]
    for i in range(NUM_KP):
        cols += [f"kp{i}_y", f"kp{i}_x", f"kp{i}_s"]
    cols.append("label")
    return cols


def make_row(video_id: str, frame_idx: int, time_sec: float,
             kps: np.ndarray, label: int) -> list:
    row: list = [video_id, frame_idx, round(time_sec, 4)]
    for k in range(NUM_KP):
        row += [round(float(kps[k, 0]), 6),
                round(float(kps[k, 1]), 6),
                round(float(kps[k, 2]), 6)]
    row.append(label)
    return row


# ── 시퀀스 처리 ───────────────────────────────────────────────────────────────

RAW_TO_LABEL = {-1: 0, 0: 1, 1: 2}   # non_fall / falling / fallen


def process_sequence(interp, in_det, out_det,
                     seq_dir: Path, video_id: str,
                     ann: dict[tuple[str, int], int] | None,
                     seq_key: str, fps: float, writer) -> int:
    frames = sorted(seq_dir.glob("*.png"),
                    key=lambda p: int(re.sub(r"[^0-9]", "", p.stem) or 0))
    if not frames:
        print(f"  [SKIP] 프레임 없음: {seq_dir}")
        return 0

    count = 0
    for frame_idx, img_path in enumerate(frames, start=1):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        kps = infer_frame(interp, bgr, in_det, out_det)
        time_sec = (frame_idx - 1) / fps

        if ann is None:
            # ADL 시퀀스 → non_fall
            label = 0
        else:
            raw = ann.get((seq_key, frame_idx), -1)
            label = RAW_TO_LABEL.get(raw, 0)

        writer.writerow(make_row(video_id, frame_idx, time_sec, kps, label))
        count += 1

    return count


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UR-Fall 키포인트 추출")
    parser.add_argument("--data-dir",
                        default="dataset/urfall/UR_fall_detection_dataset_cam0_rgb",
                        help="fall-XX / adl-XX 폴더가 있는 디렉토리")
    parser.add_argument("--ann-dir", default="dataset/urfall",
                        help="urfall-camN-falls.csv 가 있는 디렉토리")
    parser.add_argument("--model",
                        default="model/st_movenet_lightning_a100_heatmaps_256_int8.tflite")
    parser.add_argument("--output", default="dataset/urfall_keypoints.csv")
    parser.add_argument("--cameras", nargs="+", type=int, default=[0],
                        help="카메라 번호 (기본: 0)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fall-seqs", type=int, default=30)
    parser.add_argument("--adl-seqs", type=int, default=40)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ann_dir = Path(args.ann_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"모델 로드: {args.model}")
    interp = load_interpreter(args.model)
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print(f"  입력: {in_det['shape']}  출력: {out_det['shape']}")

    total_rows = 0
    label_counts = {0: 0, 1: 0, 2: 0}

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(make_header())

        for cam in args.cameras:
            ann_path = ann_dir / f"urfall-cam{cam}-falls.csv"
            if ann_path.exists():
                ann = load_annotations(ann_path)
                print(f"어노테이션 로드: {ann_path}  ({len(ann)}행)")
            else:
                ann = None
                print(f"[WARNING] 어노테이션 없음: {ann_path}")

            # fall 시퀀스
            for seq in range(1, args.fall_seqs + 1):
                seq_name = f"fall-{seq:02d}"
                seq_dir = data_dir / f"{seq_name}-cam{cam}-rgb"
                if not seq_dir.exists():
                    continue
                video_id = f"{seq_name}-cam{cam}"
                print(f"  [{video_id}]", end=" ", flush=True)
                n = process_sequence(interp, in_det, out_det,
                                     seq_dir, video_id, ann, seq_name, args.fps, writer)
                print(f"{n} frames")
                total_rows += n

            # ADL 시퀀스 (전부 non_fall)
            for seq in range(1, args.adl_seqs + 1):
                seq_name = f"adl-{seq:02d}"
                seq_dir = data_dir / f"{seq_name}-cam{cam}-rgb"
                if not seq_dir.exists():
                    continue
                video_id = f"{seq_name}-cam{cam}"
                print(f"  [{video_id}]", end=" ", flush=True)
                n = process_sequence(interp, in_det, out_det,
                                     seq_dir, video_id, None, seq_name, args.fps, writer)
                print(f"{n} frames")
                total_rows += n

    # 라벨 분포 집계
    with open(out_path, newline="") as f:
        for row in csv.DictReader(f):
            label_counts[int(row["label"])] += 1

    print(f"\n완료: {out_path}  (총 {total_rows:,}행)")
    print(f"  non_fall(0): {label_counts[0]:,}")
    print(f"  falling(1):  {label_counts[1]:,}")
    print(f"  fallen(2):   {label_counts[2]:,}")


if __name__ == "__main__":
    main()
