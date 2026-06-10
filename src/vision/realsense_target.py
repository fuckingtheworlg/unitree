"""D435i 颜色组合 detector (v4: day + night 双档 OR).

3 类地标:
  - 倾倒区:    白底 + 黑色大字       -> "BLACK_WHITE"
  - 台阶/终点: 白底 + 蓝色大字       -> "BLUE_WHITE"
  - 干扰物:                         -> "NONE"

stair vs dock 不在 detector 区分, FSM 顺序判:
  第 1 次 BLUE_WHITE = stair, 第 2 次 BLUE_WHITE = dock.

判定逻辑 (任一档过即命中):
  day_BW   = w≥0.50 且 k≥0.25 且 b≤0.20
  night_BW = w≥0.35 且 k≥0.30 且 b≤0.10
  day_BLW  = w≥0.50 且 b≥0.08 且 k≤0.20
  night_BLW= (w≥0.35 或 w+b≥0.85) 且 b≥0.06 且 k≤0.25
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np


class ColorLabel(Enum):
    NONE = "none"
    BLACK_WHITE = "black_white"
    BLUE_WHITE = "blue_white"


@dataclass
class ColorClassification:
    label: ColorLabel
    confidence: float
    white_ratio: float
    black_ratio: float
    blue_ratio: float
    yellow_ratio: float
    depth_median_m: float
    depth_valid_ratio: float
    n_valid_pixels: int
    roi_bbox: Tuple[int, int, int, int]
    red_ratio: float = 0.0
    rule_explain: str = ""


# HSV 阈值 (BGR -> HSV).
# 白: V 80~255 + S 0~80 (灯光下白底偏灰也兼容)
# 黑: V 0~80 (深色字)
# 蓝: H 90~135, S>=70, V>=40
# 黄: H 15~40, S>=70, V>=90
_HSV_WHITE = ((0, 0, 90), (180, 80, 255))
_HSV_BLACK = ((0, 0, 0), (180, 255, 80))
_HSV_BLUE = ((90, 70, 40), (135, 255, 255))
_HSV_YELLOW = ((15, 70, 90), (40, 255, 255))
# 红色在 HSV 里跨 0/180 两段, 取并集.
_HSV_RED1 = ((0, 90, 60), (10, 255, 255))
_HSV_RED2 = ((170, 90, 60), (180, 255, 255))


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    m1 = cv2.inRange(hsv, np.array(_HSV_RED1[0], np.uint8), np.array(_HSV_RED1[1], np.uint8))
    m2 = cv2.inRange(hsv, np.array(_HSV_RED2[0], np.uint8), np.array(_HSV_RED2[1], np.uint8))
    return cv2.bitwise_or(m1, m2)


def classify_color_combo(
    rgb: np.ndarray,
    depth_raw: Optional[np.ndarray] = None,
    depth_scale: float = 0.001,
    *,
    roi_w_ratio: float = 0.60,
    roi_h_ratio: float = 0.60,
    depth_min_m: float = 0.05,
    depth_max_m: float = 0.50,
    require_depth: bool = True,
    min_valid_depth_ratio: float = 0.30,
    # ---- day 档 (白天/明亮自然光) ----
    white_min_ratio: float = 0.50,
    black_min_ratio: float = 0.25,
    blue_min_ratio: float = 0.08,
    other_max_ratio: float = 0.20,
    # ---- night 档 (灯光下白底变灰) ----
    night_enabled: bool = True,
    night_white_min_ratio: float = 0.35,
    night_black_min_ratio: float = 0.30,
    night_blue_min_ratio: float = 0.06,
    night_other_max_ratio_blackwhite: float = 0.10,
    night_other_max_ratio_bluewhite: float = 0.25,
    night_blue_dominant_sum: float = 0.85,
    yellow_max_ratio: float = 0.30,
    # ---- 亮度自适应 (只在偏暗时放宽"白"的 V 下限, 亮场不变) ----
    adaptive_enabled: bool = True,
    adaptive_white_v_base: float = 90.0,
    adaptive_v_ref: float = 128.0,
) -> ColorClassification:
    """根据画面中央 ROI 像素颜色占比 + depth 过滤, 输出地标 label.

    判定: day 或 night 任一档过 -> 命中
    """
    if rgb is None or rgb.size == 0:
        return ColorClassification(
            label=ColorLabel.NONE, confidence=0.0,
            white_ratio=0, black_ratio=0, blue_ratio=0, yellow_ratio=0,
            depth_median_m=0, depth_valid_ratio=0,
            n_valid_pixels=0, roi_bbox=(0, 0, 0, 0),
            rule_explain="empty rgb",
        )

    h, w = rgb.shape[:2]
    rw = max(40, int(w * roi_w_ratio))
    rh = max(40, int(h * roi_h_ratio))
    x0 = (w - rw) // 2
    y0 = (h - rh) // 2
    roi_rgb = rgb[y0:y0 + rh, x0:x0 + rw]

    depth_median_m = 0.0
    depth_valid_ratio = 0.0
    color_pixel_mask: Optional[np.ndarray] = None
    n_valid = roi_rgb.shape[0] * roi_rgb.shape[1]
    if depth_raw is not None and depth_raw.shape[:2] == rgb.shape[:2]:
        roi_depth = depth_raw[y0:y0 + rh, x0:x0 + rw]
        z_lo_u = max(1, int(depth_min_m / depth_scale))
        z_hi_u = max(z_lo_u + 1, int(depth_max_m / depth_scale))
        valid = (roi_depth >= z_lo_u) & (roi_depth <= z_hi_u)
        n_valid = int(np.count_nonzero(valid))
        depth_valid_ratio = float(n_valid) / float(roi_depth.size)
        if n_valid > 0:
            depth_median_m = float(np.median(roi_depth[valid])) * depth_scale
            color_pixel_mask = valid.astype(np.uint8) * 255
    else:
        depth_valid_ratio = 1.0

    if require_depth and depth_valid_ratio < min_valid_depth_ratio:
        return ColorClassification(
            label=ColorLabel.NONE, confidence=0.0,
            white_ratio=0, black_ratio=0, blue_ratio=0, yellow_ratio=0,
            depth_median_m=depth_median_m,
            depth_valid_ratio=depth_valid_ratio,
            n_valid_pixels=n_valid,
            roi_bbox=(x0, y0, rw, rh),
            rule_explain=f"depth valid {depth_valid_ratio:.2f}<{min_valid_depth_ratio:.2f}",
        )

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)

    # 亮度自适应: 暗场下白色油漆 V 会掉到 ~120, 固定 V>=90 仍能过, 但灯光更暗时
    # (或反光不足) 会漏判. 这里按 ROI 中位亮度把"白"的 V 下限往下调, 亮场封顶=base
    # 保持原行为, 不会把灰当白. 黑/蓝/黄保持固定 (色相判定更稳).
    v_channel = hsv[:, :, 2]
    if color_pixel_mask is not None and int(np.count_nonzero(color_pixel_mask)) > 0:
        v_med = float(np.median(v_channel[color_pixel_mask > 0]))
    else:
        v_med = float(np.median(v_channel))
    if adaptive_enabled:
        white_v_min = int(np.clip(
            adaptive_white_v_base * (v_med / max(1.0, adaptive_v_ref)),
            55, adaptive_white_v_base,
        ))
    else:
        white_v_min = int(_HSV_WHITE[0][2])
    white_lo_arr = np.array((_HSV_WHITE[0][0], _HSV_WHITE[0][1], white_v_min), np.uint8)
    white_hi_arr = np.array(_HSV_WHITE[1], np.uint8)

    if color_pixel_mask is not None:
        hsv_for_stat = hsv.copy()
        hsv_for_stat[color_pixel_mask == 0] = 0
        white_mask = cv2.inRange(hsv_for_stat, white_lo_arr, white_hi_arr)
        black_mask = cv2.inRange(hsv_for_stat,
            np.array(_HSV_BLACK[0], np.uint8), np.array(_HSV_BLACK[1], np.uint8))
        blue_mask = cv2.inRange(hsv_for_stat,
            np.array(_HSV_BLUE[0], np.uint8), np.array(_HSV_BLUE[1], np.uint8))
        yellow_mask = cv2.inRange(hsv_for_stat,
            np.array(_HSV_YELLOW[0], np.uint8), np.array(_HSV_YELLOW[1], np.uint8))
        red_mask = _red_mask(hsv_for_stat)
        white_mask = cv2.bitwise_and(white_mask, color_pixel_mask)
        black_mask = cv2.bitwise_and(black_mask, color_pixel_mask)
        blue_mask = cv2.bitwise_and(blue_mask, color_pixel_mask)
        yellow_mask = cv2.bitwise_and(yellow_mask, color_pixel_mask)
        red_mask = cv2.bitwise_and(red_mask, color_pixel_mask)
        denom = max(1, int(np.count_nonzero(color_pixel_mask)))
    else:
        white_mask = cv2.inRange(hsv, white_lo_arr, white_hi_arr)
        black_mask = cv2.inRange(hsv,
            np.array(_HSV_BLACK[0], np.uint8), np.array(_HSV_BLACK[1], np.uint8))
        blue_mask = cv2.inRange(hsv,
            np.array(_HSV_BLUE[0], np.uint8), np.array(_HSV_BLUE[1], np.uint8))
        yellow_mask = cv2.inRange(hsv,
            np.array(_HSV_YELLOW[0], np.uint8), np.array(_HSV_YELLOW[1], np.uint8))
        red_mask = _red_mask(hsv)
        denom = max(1, hsv.shape[0] * hsv.shape[1])

    white_ratio = float(np.count_nonzero(white_mask)) / float(denom)
    black_ratio = float(np.count_nonzero(black_mask)) / float(denom)
    blue_ratio = float(np.count_nonzero(blue_mask)) / float(denom)
    yellow_ratio = float(np.count_nonzero(yellow_mask)) / float(denom)
    red_ratio = float(np.count_nonzero(red_mask)) / float(denom)

    label = ColorLabel.NONE
    confidence = 0.0
    explain = ""

    day_bw = (white_ratio >= white_min_ratio
              and black_ratio >= black_min_ratio
              and blue_ratio <= other_max_ratio)
    day_blw = (white_ratio >= white_min_ratio
               and blue_ratio >= blue_min_ratio
               and black_ratio <= other_max_ratio)
    night_bw = (
        night_enabled
        and white_ratio >= night_white_min_ratio
        and black_ratio >= night_black_min_ratio
        and blue_ratio <= night_other_max_ratio_blackwhite
    )
    night_blw = (
        night_enabled
        and (white_ratio >= night_white_min_ratio
             or (white_ratio + blue_ratio) >= night_blue_dominant_sum)
        and blue_ratio >= night_blue_min_ratio
        and black_ratio <= night_other_max_ratio_bluewhite
    )

    if yellow_ratio >= yellow_max_ratio:
        explain = f"yellow {yellow_ratio:.2f}>={yellow_max_ratio:.2f} (画面被黄道占满)"
    elif day_bw or night_bw:
        label = ColorLabel.BLACK_WHITE
        score_w = min(1.0, white_ratio / 0.40)
        score_b = min(1.0, black_ratio / 0.30)
        confidence = 0.5 * score_w + 0.5 * score_b
        which = "day" if day_bw else "night"
        explain = (f"BLACK_WHITE [{which}]: w={white_ratio:.2f} k={black_ratio:.2f} "
                   f"b={blue_ratio:.2f}")
    elif day_blw or night_blw:
        label = ColorLabel.BLUE_WHITE
        score_w = min(1.0, white_ratio / 0.40)
        score_bl = min(1.0, blue_ratio / 0.20)
        confidence = 0.5 * score_w + 0.5 * score_bl
        which = "day" if day_blw else "night"
        explain = (f"BLUE_WHITE [{which}]: w={white_ratio:.2f} b={blue_ratio:.2f} "
                   f"k={black_ratio:.2f}")
    else:
        explain = (f"NONE: w={white_ratio:.2f} k={black_ratio:.2f} "
                   f"b={blue_ratio:.2f} (day & night 都没过)")

    return ColorClassification(
        label=label,
        confidence=round(confidence, 3),
        white_ratio=round(white_ratio, 3),
        black_ratio=round(black_ratio, 3),
        blue_ratio=round(blue_ratio, 3),
        yellow_ratio=round(yellow_ratio, 3),
        depth_median_m=round(depth_median_m, 3),
        depth_valid_ratio=round(depth_valid_ratio, 3),
        n_valid_pixels=int(n_valid),
        roi_bbox=(int(x0), int(y0), int(rw), int(rh)),
        red_ratio=round(red_ratio, 3),
        rule_explain=explain,
    )


@dataclass
class ColorTriggerState:
    """N 帧稳定 + 冷却期管理 (按时间).

    设计:
      - hit_window: 最近 W 帧
      - 命中条件: hit_window 内 ≥ min_hits 帧命中 (容错)
      - cooldown: 触发后等 cooldown_after_time_s 才允许再次触发
      - 用时间不用距离: mission_dist 是命令积分跟实际位移可能差很多
    """
    name: str
    target_label: ColorLabel
    trigger_depth_m: float
    n_stable: int = 5
    min_hits: int = 4
    cooldown_after_time_s: float = 0.0
    armed: bool = True
    _recent_hits: List[bool] = field(default_factory=list)
    _last_trigger_ts: float = 0.0
    _has_triggered: bool = False

    def update(self, cls: ColorClassification, _unused: float = 0.0) -> bool:
        """每帧调一次. 返回 True = 触发."""
        import time as _time
        now = _time.time()
        if not self.armed:
            return False
        if (self._has_triggered and
                now - self._last_trigger_ts < self.cooldown_after_time_s):
            self._recent_hits.clear()
            return False
        is_hit = (
            cls.label == self.target_label
            and cls.depth_median_m > 0.0
            and cls.depth_median_m <= self.trigger_depth_m
        )
        self._recent_hits.append(is_hit)
        if len(self._recent_hits) > self.n_stable:
            self._recent_hits.pop(0)
        if (len(self._recent_hits) >= self.n_stable
                and sum(self._recent_hits) >= self.min_hits):
            self._last_trigger_ts = now
            self._has_triggered = True
            self._recent_hits.clear()
            return True
        return False

    def disarm(self) -> None:
        self.armed = False
        self._recent_hits.clear()


# ---- 旧接口空 stub 兼容老 import ----
class TargetKind(Enum):
    UNKNOWN = "unknown"


@dataclass
class WhiteTarget:
    kind: TargetKind = TargetKind.UNKNOWN


def detect_white_targets(*args, **kwargs) -> List[WhiteTarget]:
    return []


def select_target(targets, *args, **kwargs):
    return targets[0] if targets else None
