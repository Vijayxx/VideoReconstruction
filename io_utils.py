from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def read_video_frames(
    video_path: str,
    *,
    downscale_width: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video '{video_path}'.")

    original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0

    if downscale_width <= 0:
        raise ValueError("Downscale width must be positive.")
    if original_width == 0 or original_height == 0:
        raise ValueError("Video reports zero dimensions; cannot proceed.")

    scale = downscale_width / float(original_width)
    downscale_height = max(1, int(round(original_height * scale)))

    full_frames: List[np.ndarray] = []
    small_frames: List[np.ndarray] = []

    success, frame = capture.read()
    while success:
        full_frames.append(frame.copy())
        small_frame = cv2.resize(frame, (downscale_width, downscale_height), interpolation=cv2.INTER_AREA)
        small_frames.append(small_frame)
        success, frame = capture.read()

    capture.release()

    if not full_frames:
        raise ValueError("No frames decoded from the video.")

    return full_frames, small_frames, fps


def write_video(
    frames: List[np.ndarray],
    path: str,
    *,
    fps: float,
) -> None:
    if not frames:
        raise ValueError("Attempting to write an empty frame sequence.")

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for '{path}'.")

    for frame in frames:
        writer.write(frame)
    writer.release()


def write_frame_order_csv(order: np.ndarray, path: str) -> None:
    import csv

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position", "frame_index"])
        for idx, frame_idx in enumerate(order):
            writer.writerow([idx, frame_idx])


def write_similarity_report(similarity: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, similarity, delimiter=",", fmt="%.6f")


def write_timings(timings: dict, path: str) -> None:
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(timings, handle, indent=2)
