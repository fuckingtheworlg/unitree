"""地标识别 - 基于场地真实视觉特征.

省赛场地标志特征 (用户实测确认):
  - 倾倒区:  黄色圆环 + 内部白色 + 黑色字 "倾倒区"
  - 充电区:  蓝色矩形外框 + 内部白色 + 蓝色字 "充电区"
  - 台阶区:  蓝色矩形外框 + 内部白色 + 蓝色字 "台阶区"
  - 黄道:    黄色实心长条 (没字, 没白色填充)

充电区 vs 台阶区: 视觉特征完全相同 (都是蓝色矩形+白底+蓝字),
   只能靠**进入顺序**区分 -- 赛规要求先经过台阶, 再到充电区.
   FSM 用 stair_climbed 标志: 第一次见到"蓝矩形" -> 台阶, 第二次 -> 充电.

所有 landmark 检测都在**原图 ROI** 上跑 (不经 IPM):
   - 颜色饱和度保留 (IPM 双线性插值会冲淡颜色)
   - 远距离地标也能识别 (IPM 主要处理近端地面)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np


class LandmarkType(Enum):
    NONE = "none"
    DUMP_ZONE = "dump_zone"      # 黄色圆环
    FORK = "fork"                # Y 型分叉
    BLUE_RECT = "blue_rect"      # 蓝矩形 (台阶或充电, 由 FSM 顺序决定)
    BLUE_RING_OBSTACLE = "blue_ring_obstacle"
    STAIR = "stair"
    DOCK_AREA = "dock_area"


@dataclass
class LandmarkDetection:
    type: LandmarkType
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    extra: Optional[dict] = None


_BLUE_HSV_LOWER = np.array([95, 60, 50], np.uint8)
_BLUE_HSV_UPPER = np.array([130, 255, 255], np.uint8)

# 白色: 低饱和度 + 高亮度
_WHITE_HSV_LOWER = np.array([0, 0, 180], np.uint8)
_WHITE_HSV_UPPER = np.array([180, 60, 255], np.uint8)


def detect_dump_zone(
    yellow_mask: np.ndarray,
    bgr: np.ndarray,
    min_radius_ratio: float = 0.10,
    min_area_ratio: float = 0.02,
    circularity_min: float = 0.70,
    inner_white_ratio_min: float = 0.30,
    hough_param2: int = 25,
    ring_yellow_min: float = 0.35,
) -> Optional[LandmarkDetection]:
    """倾倒区: 黄色圆环 + 内部填充白色 + 黑色字.

    重要: 黄道连着圆环, findContours 把它们当一个不规则连通域 (圆度~0.26),
    所以不能用 contour 圆度找. 改用 HoughCircles 直接找圆.

    检测策略:
      1. HoughCircles 在黄色 mask 上找候选圆
      2. 验证圆心周围一片小区域是白色 (倾倒区中央白底)
      3. 验证圆环上有黄色像素 (在 r±5px 圆环上采样)
    """
    h, w = yellow_mask.shape[:2]
    if h < 10 or w < 10 or bgr is None:
        return None

    short_side = min(h, w)
    blurred = cv2.GaussianBlur(yellow_mask, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.5,
        minDist=int(short_side * 0.3),
        param1=100, param2=hough_param2,
        minRadius=max(int(short_side * min_radius_ratio), 20),
        maxRadius=int(short_side * 0.50),
    )
    if circles is None or circles.shape[1] == 0:
        return None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, _WHITE_HSV_LOWER, _WHITE_HSV_UPPER)

    best: Optional[LandmarkDetection] = None

    for cx, cy, r in circles[0]:
        cx, cy, r = int(cx), int(cy), int(r)
        if r <= 0:
            continue

        # 1. 圆心周围一小块的白色比例 (内部白底)
        inner_r = max(int(r * 0.5), 5)
        x0, x1 = max(0, cx - inner_r), min(w, cx + inner_r)
        y0, y1 = max(0, cy - inner_r), min(h, cy + inner_r)
        inner_patch = white_mask[y0:y1, x0:x1]
        if inner_patch.size == 0:
            continue
        white_ratio = float((inner_patch > 0).mean())
        if white_ratio < inner_white_ratio_min:
            continue

        # 2. 圆环上 (r±5px) 黄色比例 (圆环本身是黄色)
        ring_score = 0.0
        n_samples = 16
        ring_pixels = 0
        ring_yellow = 0
        for k in range(n_samples):
            theta = 2 * np.pi * k / n_samples
            for dr in (-5, 0, 5):
                px = int(cx + (r + dr) * np.cos(theta))
                py = int(cy + (r + dr) * np.sin(theta))
                if 0 <= px < w and 0 <= py < h:
                    ring_pixels += 1
                    if yellow_mask[py, px] > 0:
                        ring_yellow += 1
        if ring_pixels > 0:
            ring_score = ring_yellow / ring_pixels
        if ring_score < ring_yellow_min:
            continue

        # 3. 面积
        area = np.pi * r * r
        if area < min_area_ratio * h * w:
            continue

        confidence = float(min(1.0, white_ratio * 0.5 + ring_score * 0.5))
        det = LandmarkDetection(
            type=LandmarkType.DUMP_ZONE,
            confidence=confidence,
            bbox=(cx - r, cy - r, 2 * r, 2 * r),
            extra={
                "radius": float(r),
                "center": (float(cx), float(cy)),
                "inner_white": float(white_ratio),
                "ring_yellow": float(ring_score),
            },
        )
        if best is None or det.confidence > best.confidence:
            best = det
    return best


def detect_blue_rect(
    bgr: np.ndarray,
    min_area_ratio: float = 0.015,
    aspect_min: float = 0.4,
    aspect_max: float = 15.0,
    inner_white_ratio_min: float = 0.20,
    close_kernel: int = 11,
) -> Optional[LandmarkDetection]:
    """蓝边白底蓝字矩形 (省赛+国赛通用模式):
      - 省赛: 充电区/台阶区 (路径终点/楼梯)
      - 国赛: 充电区/台阶区/障碍区

    所有这些标志的共同视觉特征:
      - 蓝色矩形外框
      - 中间填充白色
      - 蓝色文字 ("充电区"/"台阶区"/"障碍区")

    **不识别"实心蓝矩形"** -- 省赛场地中实心蓝是干扰物 (不在赛规);
    国赛障碍区也是 蓝边白底 模式, 不会出现整片实心蓝.

    检测策略:
      1. 蓝色 mask + 闭运算 (把"边框+字"连成一片)
      2. 找面积/aspect 合理的最大连通域
      3. **必须满足: 内部白色 >= 20%** (这是"蓝边白底"的关键判据)
      4. 文字层级区分留给 FSM 顺序 (stair_climbed 标志)
    """
    h, w = bgr.shape[:2]
    if h < 10 or w < 10:
        return None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, _BLUE_HSV_LOWER, _BLUE_HSV_UPPER)
    white = cv2.inRange(hsv, _WHITE_HSV_LOWER, _WHITE_HSV_UPPER)

    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        blue_closed = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k)
    else:
        blue_closed = blue

    contours, _ = cv2.findContours(blue_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = float(h * w)
    best: Optional[LandmarkDetection] = None

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_ratio * img_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw <= 0 or bh <= 0:
            continue
        aspect = bw / float(bh)
        if aspect < aspect_min or aspect > aspect_max:
            continue

        # 内部白色比例 (区分实心蓝色 vs 蓝边白底; 实心蓝白色比例低)
        inner_white = white[y:y + bh, x:x + bw]
        inner_blue = blue[y:y + bh, x:x + bw]
        white_ratio = float((inner_white > 0).mean()) if inner_white.size > 0 else 0.0
        blue_ratio = float((inner_blue > 0).mean()) if inner_blue.size > 0 else 0.0
        rect_fill = area / float(bw * bh)

        if white_ratio < inner_white_ratio_min:
            continue

        confidence = float(min(1.0, white_ratio * 2.0))
        det = LandmarkDetection(
            type=LandmarkType.BLUE_RECT,
            confidence=confidence,
            bbox=(x, y, bw, bh),
            extra={
                "aspect": float(aspect),
                "white_ratio": float(white_ratio),
                "blue_ratio": float(blue_ratio),
                "rect_fill": float(rect_fill),
                "area_ratio": float(area / img_area),
                "cx": float(x + bw / 2.0),
                "image_w": int(w),
            },
        )
        if best is None or det.confidence > best.confidence:
            best = det

    return best


def detect_blue_obstacle_ring(
    bgr: np.ndarray,
    min_area_ratio: float = 0.015,
    min_circularity: float = 0.45,
    min_aspect: float = 0.65,
    max_aspect: float = 1.55,
    ring_blue_min: float = 0.16,
    inner_white_min: float = 0.25,
    inner_black_min: float = 0.01,
    close_kernel: int = 9,
) -> Optional[LandmarkDetection]:
    """障碍区: 蓝色圆环 + 白色内区 + 黑色"障碍区"字样.

    与蓝色矩形终点/台阶区不同, 这里要求外形接近圆/椭圆, 且内区有
    白底和少量黑字。返回的 right_band_x 可用于固定右绕时瞄准右侧蓝环。
    """
    h, w = bgr.shape[:2]
    if h < 20 or w < 20:
        return None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, _BLUE_HSV_LOWER, _BLUE_HSV_UPPER)
    white = cv2.inRange(hsv, _WHITE_HSV_LOWER, _WHITE_HSV_UPPER)
    black = cv2.inRange(hsv, np.array([0, 0, 0], np.uint8), np.array([180, 80, 90], np.uint8))

    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        blue_work = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k)
    else:
        blue_work = blue

    contours, _ = cv2.findContours(blue_work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = float(h * w)
    best: Optional[LandmarkDetection] = None

    yy, xx = np.indices((h, w))
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_ratio * img_area:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 1.0:
            continue
        circularity = float(4.0 * np.pi * area / (peri * peri))
        if circularity < min_circularity:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if bw <= 0 or bh <= 0:
            continue
        aspect = bw / float(bh)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius <= 5:
            continue
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        inner_mask = dist <= radius * 0.55
        ring_mask = (dist >= radius * 0.55) & (dist <= radius * 1.05)
        if not np.any(inner_mask) or not np.any(ring_mask):
            continue

        ring_blue = float(np.count_nonzero((blue > 0) & ring_mask)) / float(np.count_nonzero(ring_mask))
        inner_white = float(np.count_nonzero((white > 0) & inner_mask)) / float(np.count_nonzero(inner_mask))
        inner_black = float(np.count_nonzero((black > 0) & inner_mask)) / float(np.count_nonzero(inner_mask))
        if ring_blue < ring_blue_min:
            continue
        if inner_white < inner_white_min or inner_black < inner_black_min:
            continue

        confidence = float(min(
            1.0,
            0.35 * min(1.0, ring_blue / max(ring_blue_min, 1e-6))
            + 0.35 * min(1.0, inner_white / max(inner_white_min, 1e-6))
            + 0.20 * min(1.0, inner_black / max(inner_black_min * 4.0, 1e-6))
            + 0.10 * min(1.0, circularity / 0.80),
        ))
        right_band_x = min(float(w - 1), cx + radius * 0.65)
        left_band_x = max(0.0, cx - radius * 0.65)
        det = LandmarkDetection(
            type=LandmarkType.BLUE_RING_OBSTACLE,
            confidence=confidence,
            bbox=(x, y, bw, bh),
            extra={
                "center": (float(cx), float(cy)),
                "radius": float(radius),
                "circularity": float(circularity),
                "aspect": float(aspect),
                "ring_blue": float(ring_blue),
                "inner_white": float(inner_white),
                "inner_black": float(inner_black),
                "right_band_x": float(right_band_x),
                "left_band_x": float(left_band_x),
                "image_w": int(w),
            },
        )
        if best is None or det.confidence > best.confidence:
            best = det

    return best


def detect_dock_area(bgr: np.ndarray, **kwargs) -> Optional[LandmarkDetection]:
    """充电区: 跟台阶区视觉特征一样, 由 FSM 顺序决定语义.
    返回类型标记为 DOCK_AREA, 调用方根据 stair_climbed 标志重新解释."""
    det = detect_blue_rect(bgr, **kwargs)
    if det is not None:
        det.type = LandmarkType.DOCK_AREA
    return det


def detect_stair(bgr: np.ndarray, **kwargs) -> Optional[LandmarkDetection]:
    """台阶区: 跟充电区视觉特征一样, 由 FSM 顺序决定语义."""
    det = detect_blue_rect(bgr, **kwargs)
    if det is not None:
        det.type = LandmarkType.STAIR
    return det


def detect_fork(
    yellow_mask: np.ndarray,
    upper_band_ratio: float = 0.55,
    min_branch_area_ratio: float = 0.02,
    min_split_gap_ratio: float = 0.18,
) -> Optional[LandmarkDetection]:
    """看 ROI 上方一条带子里黄色像素的列直方图，是否能找到双峰."""
    h, w = yellow_mask.shape[:2]
    band = yellow_mask[: int(h * upper_band_ratio)]
    if band.size == 0:
        return None
    col_sum = band.sum(axis=0).astype(np.float32) / 255.0

    smooth = cv2.GaussianBlur(col_sum.reshape(1, -1), (1, 31), 0).ravel()

    threshold = max(smooth.max() * 0.4, min_branch_area_ratio * band.shape[0])
    above = smooth > threshold
    if not above.any():
        return None

    groups = []
    in_group = False
    g0 = 0
    for i, v in enumerate(above):
        if v and not in_group:
            in_group = True
            g0 = i
        elif not v and in_group:
            in_group = False
            groups.append((g0, i))
    if in_group:
        groups.append((g0, len(above)))

    if len(groups) < 2:
        return None

    groups.sort(key=lambda g: smooth[g[0]:g[1]].sum(), reverse=True)
    g_a, g_b = groups[0], groups[1]
    cx_a = (g_a[0] + g_a[1]) / 2.0
    cx_b = (g_b[0] + g_b[1]) / 2.0
    gap = abs(cx_a - cx_b)

    if gap < min_split_gap_ratio * w:
        return None

    left_x, right_x = sorted([cx_a, cx_b])
    return LandmarkDetection(
        type=LandmarkType.FORK,
        confidence=float(min(1.0, gap / w / 0.5)),
        bbox=(int(left_x), 0, int(right_x - left_x), int(h * upper_band_ratio)),
        extra={
            "left_x": float(left_x),
            "right_x": float(right_x),
            "gap_px": float(gap),
            "image_w": int(w),
        },
    )
