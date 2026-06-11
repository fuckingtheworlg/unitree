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
    # ⚠ hue 下限 24: 木地板 H 16~23 (暖色木纹), 赛道黄 H≈28.
    # 旧值 (12,48) 会让 adaptive 分支把整片木地板吞为黄线 (国赛场地实测).
    adaptive_h_range: Tuple[int, int] = (24, 40),
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


def yellow_mask_lab(
    bgr: np.ndarray,
    open_kernel: int = 3,
    close_kernel: int = 7,
    *,
    b_delta_min: int = 22,
    hue_range: Tuple[int, int] = (20, 40),
    v_min: int = 60,
    rg_delta_min: int = 25,
) -> np.ndarray:
    """光照自适应黄线检测 (LAB 相对阈值版).

    原理: LAB 的 b 通道 = "黄-蓝"色度, 受亮度影响小. 用
        黄线得分 = b(像素) - median(b(画面))  >= b_delta_min
    做主判定 -- 灯光整体变亮变暗时中位数跟着变, 差值不变, 天然抗光照.

    再用两个轻量 gate 防误判:
      - hue gate: H 必须在黄色区 (排除红/绿对 b 的干扰)
      - rg gate: min(R,G)-B >= rg_delta_min (黄色的 RGB 本质特征)

    阈值依据 (3 场地实测):
      木地板:  b-median ≈ +3~+12,  rg_delta 19~39
      赛道黄:  b-median ≈ +40~+80, rg_delta 80+
    """
    if bgr is None or bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_b = lab[:, :, 2].astype(np.int16)
    b_med = int(np.median(lab_b))
    score_ok = (lab_b - b_med) >= b_delta_min

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_lo, h_hi = hue_range
    hue_ok = (hsv[:, :, 0] >= h_lo) & (hsv[:, :, 0] <= h_hi)
    v_ok = hsv[:, :, 2] >= v_min

    b_ch = bgr[:, :, 0].astype(np.int16)
    g_ch = bgr[:, :, 1].astype(np.int16)
    r_ch = bgr[:, :, 2].astype(np.int16)
    rg_ok = (np.minimum(r_ch, g_ch) - b_ch) >= rg_delta_min

    mask = (score_ok & hue_ok & v_ok & rg_ok).astype(np.uint8) * 255

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
