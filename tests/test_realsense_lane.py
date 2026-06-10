import numpy as np

from src.vision.realsense_lane import estimate_bottom_yellow_lane


def _bottom_yellow_frame(w=320, h=240, x_center=220, band_w=28):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (35, 35, 35)
    x0 = max(0, x_center - band_w // 2)
    x1 = min(w, x_center + band_w // 2)
    img[:, x0:x1] = (0, 220, 255)
    depth = np.full((h, w), 250, np.uint16)
    return img, depth


def test_bottom_yellow_lane_reports_right_error():
    img, depth = _bottom_yellow_frame(x_center=230)
    lane = estimate_bottom_yellow_lane(
        img,
        depth_raw=depth,
        depth_scale=0.001,
        lower_hsv=(18, 80, 120),
        upper_hsv=(38, 255, 255),
        adaptive=False,
        min_yellow_ratio=0.005,
    )
    assert lane.found
    assert lane.error > 0.25


def test_bottom_yellow_lane_respects_error_sign():
    img, depth = _bottom_yellow_frame(x_center=230)
    lane = estimate_bottom_yellow_lane(
        img,
        depth_raw=depth,
        depth_scale=0.001,
        lower_hsv=(18, 80, 120),
        upper_hsv=(38, 255, 255),
        adaptive=False,
        min_yellow_ratio=0.005,
        error_sign=-1.0,
    )
    assert lane.found
    assert lane.error < -0.25


def test_bottom_yellow_lane_rejects_empty_frame():
    img = np.zeros((240, 320, 3), np.uint8)
    lane = estimate_bottom_yellow_lane(
        img,
        lower_hsv=(18, 80, 120),
        upper_hsv=(38, 255, 255),
        adaptive=False,
    )
    assert not lane.found
