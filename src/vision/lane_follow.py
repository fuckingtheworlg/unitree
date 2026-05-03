"""巡线核心：从黄色掩膜里算出"中心线相对画面中心的横向偏差(归一化)"。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class LaneResult:
    found: bool
    error: float           # 横向误差, 归一化到 [-1, 1], 正=向右偏
    confidence: float      # [0, 1]
    centerline_x: List[int]
    debug_centroid: Optional[Tuple[int, int]] = None


def estimate_lane_error(
    mask: np.ndarray,
    n_strips: int = 8,
    min_pixels_per_strip: int = 60,
    weight_strategy: str = "near",
) -> LaneResult:
    """
    把 mask 按高度切成 n_strips 条，每条求黄色像素 x 重心；
    用近端权重对重心做加权平均，对画面中心算偏差。

    weight_strategy:
      - "near"  : 越靠近底部 (狗的脚下) 权重越高，适合稳态巡线
      - "uniform": 等权
    """
    if mask is None or mask.size == 0:
        return LaneResult(False, 0.0, 0.0, [])

    h, w = mask.shape[:2]
    strip_h = max(h // n_strips, 1)

    centers: List[int] = []
    weights: List[float] = []

    for i in range(n_strips):
        y0 = i * strip_h
        y1 = h if i == n_strips - 1 else (i + 1) * strip_h
        strip = mask[y0:y1]
        ys, xs = np.where(strip > 0)
        if xs.size < min_pixels_per_strip:
            centers.append(-1)
            weights.append(0.0)
            continue
        cx = int(xs.mean())
        centers.append(cx)
        if weight_strategy == "near":
            w_i = (i + 1) / n_strips
        else:
            w_i = 1.0
        weights.append(w_i * (xs.size / max(strip.size, 1)))

    valid = [(c, w_) for c, w_ in zip(centers, weights) if c >= 0]
    if not valid:
        return LaneResult(False, 0.0, 0.0, centers)

    cs = np.array([c for c, _ in valid], dtype=np.float32)
    ws = np.array([w_ for _, w_ in valid], dtype=np.float32)
    if ws.sum() <= 1e-6:
        return LaneResult(False, 0.0, 0.0, centers)

    weighted_cx = float((cs * ws).sum() / ws.sum())
    error = (weighted_cx - w / 2.0) / (w / 2.0)
    error = float(np.clip(error, -1.0, 1.0))

    confidence = min(1.0, len(valid) / float(n_strips))

    return LaneResult(
        found=True,
        error=error,
        confidence=confidence,
        centerline_x=centers,
        debug_centroid=(int(weighted_cx), h // 2),
    )


def find_largest_yellow_centroid(mask: np.ndarray, min_area: int) -> Optional[Tuple[int, int, int]]:
    """返回最大黄色连通块的 (cx, cy, area)。地标识别会复用。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(largest))
    if area < min_area:
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy, area
