"""仅在 GUI 调试时使用，把识别结果叠加到原图。"""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from .lane_follow import LaneResult
from .landmark import LandmarkDetection, LandmarkType


_TYPE_COLOR = {
    LandmarkType.DUMP_ZONE: (0, 255, 255),
    LandmarkType.FORK: (0, 165, 255),
    LandmarkType.STAIR: (255, 0, 255),
    LandmarkType.DOCK_AREA: (255, 200, 0),
}


def overlay(
    bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    lane: Optional[LaneResult] = None,
    landmarks: Optional[Iterable[LandmarkDetection]] = None,
    state_text: str = "",
    error: float = 0.0,
    vyaw: float = 0.0,
    fps: float = 0.0,
) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]

    if mask is not None:
        m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        m3[:, :, 0] = 0
        m3[:, :, 2] = 0
        out = cv2.addWeighted(out, 0.7, m3, 0.3, 0.0)

    cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

    if lane and lane.found:
        if lane.debug_centroid is not None:
            cx, cy = lane.debug_centroid
            cv2.circle(out, (int(cx), int(cy)), 8, (0, 255, 0), -1)
            cv2.line(out, (w // 2, h - 30), (int(cx), h - 30), (0, 255, 0), 3)

    if landmarks:
        for det in landmarks:
            color = _TYPE_COLOR.get(det.type, (255, 255, 255))
            if det.bbox is not None:
                x, y, bw, bh = det.bbox
                cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            label = f"{det.type.value} {det.confidence:.2f}"
            tx = det.bbox[0] if det.bbox else 10
            ty = max(20, det.bbox[1] - 6) if det.bbox else 20
            cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    info = f"state={state_text} err={error:+.2f} vyaw={vyaw:+.2f} fps={fps:4.1f}"
    cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(out, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return out
