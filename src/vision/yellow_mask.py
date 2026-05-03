"""黄色道路二值掩膜: HSV 阈值 (单组或多组并集) + 形态学.

支持:
  - 单组: yellow_mask(bgr, lower, upper)               (向后兼容)
  - 多组: yellow_mask(bgr, ranges=[(lo1,hi1), (lo2,hi2), ...])
          多组 inRange 取 OR, 用于覆盖不同光照下的黄
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def yellow_mask(
    bgr: np.ndarray,
    lower_hsv: Optional[Tuple[int, int, int]] = None,
    upper_hsv: Optional[Tuple[int, int, int]] = None,
    open_kernel: int = 3,
    close_kernel: int = 7,
    ranges: Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = None,
) -> np.ndarray:
    """返回 0/255 的二值掩膜.

    Args:
      bgr: 输入 BGR 图
      lower_hsv, upper_hsv: 单组 HSV 阈值 (向后兼容)
      ranges: 多组 [(lo1,hi1), ...], **若提供 ranges, 则忽略 lower/upper**
      open_kernel, close_kernel: 形态学核大小
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    if ranges:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = cv2.bitwise_or(mask, m)
    else:
        if lower_hsv is None or upper_hsv is None:
            raise ValueError("必须提供 ranges 或 (lower_hsv, upper_hsv)")
        mask = cv2.inRange(hsv,
                           np.array(lower_hsv, np.uint8),
                           np.array(upper_hsv, np.uint8))

    if open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def crop_roi(image: np.ndarray, top_ratio: float, bottom_ratio: float) -> Tuple[np.ndarray, int]:
    """裁掉天花板/天空, 只看前下方道路区域. 返回 (roi, y_offset)."""
    h = image.shape[0]
    y0 = int(h * top_ratio)
    y1 = int(h * bottom_ratio)
    return image[y0:y1].copy(), y0
