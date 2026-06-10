"""黄色道路二值掩膜: HSV 阈值 (单组或多组并集) + 自适应补偿 + 形态学.

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
    adaptive: bool = True,
    adaptive_h_range: Tuple[int, int] = (12, 48),
    adaptive_s_min: int = 12,
    adaptive_v_min: int = 45,
    adaptive_lab_b_min: int = 138,
    adaptive_lab_b_delta: int = 8,
    adaptive_rg_delta_min: int = 12,
) -> np.ndarray:
    """返回 0/255 的二值掩膜.

    Args:
      bgr: 输入 BGR 图
      lower_hsv, upper_hsv: 单组 HSV 阈值 (向后兼容)
      ranges: 多组 [(lo1,hi1), ...], **若提供 ranges, 则忽略 lower/upper**
      adaptive: 额外启用低饱和/曝光变化补偿. 适合白蓝地标导致自动曝光漂移时
        兜住偏白或偏暗的黄线, 同时用 LAB b 通道和 BGR 黄色综合特征抑制白/蓝误检.
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

    if adaptive:
        mask = cv2.bitwise_or(
            mask,
            _adaptive_yellow_mask(
                bgr,
                hsv,
                h_range=adaptive_h_range,
                s_min=int(adaptive_s_min),
                v_min=int(adaptive_v_min),
                lab_b_min=int(adaptive_lab_b_min),
                lab_b_delta=int(adaptive_lab_b_delta),
                rg_delta_min=int(adaptive_rg_delta_min),
            ),
        )

    if open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _adaptive_yellow_mask(
    bgr: np.ndarray,
    hsv: np.ndarray,
    *,
    h_range: Tuple[int, int],
    s_min: int,
    v_min: int,
    lab_b_min: int,
    lab_b_delta: int,
    rg_delta_min: int,
) -> np.ndarray:
    """光照补偿黄线检测.

    HSV 的 S/V 在白蓝区域进入画面后很容易被自动曝光、白平衡拉动；LAB 的 b
    通道对"黄 vs 蓝/白"更直接。这里用宽松 hue 锁定候选, 再用 LAB b 和
    min(R,G)-B 的黄色优势做双保险。
    """
    if bgr is None or bgr.size == 0:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    h_lo, h_hi = h_range
    hue_ok = (h >= h_lo) & (h <= h_hi)
    exposure_ok = v >= v_min

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_b = lab[:, :, 2].astype(np.int16)
    # ROI 里黄线通常占比不大, 用中位数做动态背景; 白/蓝背景 b 值会接近或低于中性.
    dyn_b_min = max(lab_b_min, int(np.median(lab_b)) + lab_b_delta)
    lab_yellow = lab_b >= dyn_b_min

    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    rg_floor = np.minimum(r, g)
    yellow_advantage = (rg_floor - b) >= rg_delta_min

    soft_yellow = hue_ok & exposure_ok & lab_yellow & (
        (s >= s_min) | yellow_advantage
    )
    return (soft_yellow.astype(np.uint8) * 255)


def crop_roi(image: np.ndarray, top_ratio: float, bottom_ratio: float) -> Tuple[np.ndarray, int]:
    """裁掉天花板/天空, 只看前下方道路区域. 返回 (roi, y_offset)."""
    h = image.shape[0]
    y0 = int(h * top_ratio)
    y1 = int(h * bottom_ratio)
    return image[y0:y1].copy(), y0
