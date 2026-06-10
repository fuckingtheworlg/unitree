"""Bottom D435i lane assists.

The front fisheye remains the primary lane source.  This helper estimates a
coarse lateral error from the bottom camera so the robot can crawl through
brief front-camera dropouts around bright/blue-white landmark areas.  The same
shape is reused for blue-ring obstacle following so the controller can treat it
like a temporary lane.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .lane_follow import LaneResult, estimate_lane_error
from .yellow_mask import yellow_mask


_BLUE_HSV_LOWER = np.array([95, 60, 45], np.uint8)
_BLUE_HSV_UPPER = np.array([135, 255, 255], np.uint8)
_WHITE_HSV_LOWER = np.array([0, 0, 175], np.uint8)
_WHITE_HSV_UPPER = np.array([180, 70, 255], np.uint8)


def estimate_bottom_yellow_lane(
    bgr: np.ndarray,
    *,
    depth_raw: Optional[np.ndarray] = None,
    depth_scale: float = 0.001,
    depth_min_m: float = 0.05,
    depth_max_m: float = 0.50,
    roi_w_ratio: float = 0.80,
    roi_h_ratio: float = 0.70,
    min_depth_valid_ratio: float = 0.10,
    min_yellow_ratio: float = 0.015,
    min_pixels_per_strip: int = 30,
    n_strips: int = 6,
    error_sign: float = 1.0,
    lower_hsv: Optional[Tuple[int, int, int]] = None,
    upper_hsv: Optional[Tuple[int, int, int]] = None,
    ranges: Optional[Sequence[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = None,
    open_kernel: int = 3,
    close_kernel: int = 7,
    adaptive: bool = True,
    adaptive_h_range: Tuple[int, int] = (12, 48),
    adaptive_s_min: int = 12,
    adaptive_v_min: int = 45,
    adaptive_lab_b_min: int = 138,
    adaptive_lab_b_delta: int = 8,
    adaptive_rg_delta_min: int = 12,
) -> LaneResult:
    if bgr is None or bgr.size == 0:
        return LaneResult(False, 0.0, 0.0, [])

    h, w = bgr.shape[:2]
    rw = max(40, min(w, int(w * roi_w_ratio)))
    rh = max(40, min(h, int(h * roi_h_ratio)))
    x0 = (w - rw) // 2
    y0 = h - rh
    roi = bgr[y0:y0 + rh, x0:x0 + rw]

    mask = yellow_mask(
        roi,
        lower_hsv=lower_hsv,
        upper_hsv=upper_hsv,
        ranges=ranges,
        open_kernel=open_kernel,
        close_kernel=close_kernel,
        adaptive=adaptive,
        adaptive_h_range=adaptive_h_range,
        adaptive_s_min=adaptive_s_min,
        adaptive_v_min=adaptive_v_min,
        adaptive_lab_b_min=adaptive_lab_b_min,
        adaptive_lab_b_delta=adaptive_lab_b_delta,
        adaptive_rg_delta_min=adaptive_rg_delta_min,
    )

    if depth_raw is not None and depth_raw.shape[:2] == bgr.shape[:2]:
        depth_roi = depth_raw[y0:y0 + rh, x0:x0 + rw]
        z_lo = max(1, int(depth_min_m / depth_scale))
        z_hi = max(z_lo + 1, int(depth_max_m / depth_scale))
        valid = (depth_roi >= z_lo) & (depth_roi <= z_hi)
        if float(np.count_nonzero(valid)) / float(valid.size) < min_depth_valid_ratio:
            return LaneResult(False, 0.0, 0.0, [])
        mask = cv2.bitwise_and(mask, valid.astype(np.uint8) * 255)

    yellow_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if yellow_ratio < min_yellow_ratio:
        return LaneResult(False, 0.0, 0.0, [])

    lane = estimate_lane_error(
        mask,
        n_strips=n_strips,
        min_pixels_per_strip=min_pixels_per_strip,
        weight_strategy="near",
    )
    if not lane.found:
        return lane

    err = float(np.clip(lane.error * error_sign, -1.0, 1.0))
    return LaneResult(
        found=True,
        error=err,
        confidence=lane.confidence,
        centerline_x=lane.centerline_x,
        debug_centroid=lane.debug_centroid,
    )


def estimate_bottom_blue_ring_lane(
    bgr: np.ndarray,
    *,
    depth_raw: Optional[np.ndarray] = None,
    depth_scale: float = 0.001,
    depth_min_m: float = 0.05,
    depth_max_m: float = 0.55,
    roi_w_ratio: float = 0.90,
    roi_h_ratio: float = 0.80,
    min_depth_valid_ratio: float = 0.08,
    min_blue_ratio: float = 0.010,
    min_pixels_per_strip: int = 20,
    n_strips: int = 6,
    direction: str = "right",
) -> LaneResult:
    """Estimate lateral error for following the blue annulus as a temporary lane.

    For the required right-side route we deliberately ignore the left half when
    the right blue band is visible.  Positive error means the blue band is to the
    right of the robot center, which the existing PID convention turns into a
    right yaw.
    """
    if bgr is None or bgr.size == 0:
        return LaneResult(False, 0.0, 0.0, [])

    h, w = bgr.shape[:2]
    rw = max(40, min(w, int(w * roi_w_ratio)))
    rh = max(40, min(h, int(h * roi_h_ratio)))
    x0 = (w - rw) // 2
    y0 = h - rh
    roi = bgr[y0:y0 + rh, x0:x0 + rw]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, _BLUE_HSV_LOWER, _BLUE_HSV_UPPER)
    white = cv2.inRange(hsv, _WHITE_HSV_LOWER, _WHITE_HSV_UPPER)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, k)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k)

    if depth_raw is not None and depth_raw.shape[:2] == bgr.shape[:2]:
        depth_roi = depth_raw[y0:y0 + rh, x0:x0 + rw]
        z_lo = max(1, int(depth_min_m / depth_scale))
        z_hi = max(z_lo + 1, int(depth_max_m / depth_scale))
        valid = (depth_roi >= z_lo) & (depth_roi <= z_hi)
        if float(np.count_nonzero(valid)) / float(valid.size) < min_depth_valid_ratio:
            return LaneResult(False, 0.0, 0.0, [])
        blue = cv2.bitwise_and(blue, valid.astype(np.uint8) * 255)

    # If the white center dominates the near middle while no blue band is visible
    # nearby, avoid pretending the obstacle interior is a valid lane.
    near = slice(int(rh * 0.55), rh)
    center = slice(int(rw * 0.35), int(rw * 0.65))
    center_white = float(np.count_nonzero(white[near, center])) / float(max(1, white[near, center].size))
    center_blue = float(np.count_nonzero(blue[near, center])) / float(max(1, blue[near, center].size))
    if center_white > 0.45 and center_blue < 0.02:
        return LaneResult(False, 0.0, 0.0, [])

    side_mask = np.zeros_like(blue)
    if direction == "left":
        side_mask[:, : rw // 2] = blue[:, : rw // 2]
    else:
        side_mask[:, rw // 2:] = blue[:, rw // 2:]
    if float(np.count_nonzero(side_mask)) / float(side_mask.size) >= min_blue_ratio:
        mask = side_mask
    else:
        mask = blue

    blue_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if blue_ratio < min_blue_ratio:
        return LaneResult(False, 0.0, 0.0, [])

    lane = estimate_lane_error(
        mask,
        n_strips=n_strips,
        min_pixels_per_strip=min_pixels_per_strip,
        weight_strategy="near",
    )
    if not lane.found:
        return lane

    return LaneResult(
        found=True,
        error=float(np.clip(lane.error, -1.0, 1.0)),
        confidence=lane.confidence,
        centerline_x=lane.centerline_x,
        debug_centroid=lane.debug_centroid,
    )
