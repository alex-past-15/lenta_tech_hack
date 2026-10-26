from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def iter_frames(
    path: str, sample_fps: float, min_sharpness: float, max_side: int
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Отдаёт (индекс кадра, время в секундах, кадр) с прореживанием и отсевом размытых."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / sample_fps))
    idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % step:
                continue
            if sharpness(frame) < min_sharpness:
                continue
            h, w = frame.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            yield idx, idx / src_fps, frame
    finally:
        cap.release()
