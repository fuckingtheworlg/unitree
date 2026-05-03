"""黄色道路二值掩膜：HSV 阈值 + 形态学。"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def yellow_mask(
    bgr: np.ndarray,
    lower_hsv: Tuple[int, int, int],
    upper_hsv: Tuple[int, int, int],
    open_kernel: int = 3,
    close_kernel: int = 7,
) -> np.ndarray:
    """返回 0/255 的二值掩膜，0=非黄 255=黄。

    实测过程中如果环境光偏暖（黄昏/暖色 LED），可以放宽 V/S 上下限，
    或在赛前用 scripts/tune_hsv.py 调一组值写回 params.yaml。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv, np.uint8), np.array(upper_hsv, np.uint8))

    if open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def crop_roi(image: np.ndarray, top_ratio: float, bottom_ratio: float) -> Tuple[np.ndarray, int]:
    """裁掉天花板/天空，只看前下方道路区域，返回 (roi, y_offset)。"""
    h = image.shape[0]
    y0 = int(h * top_ratio)
    y1 = int(h * bottom_ratio)
    return image[y0:y1].copy(), y0
