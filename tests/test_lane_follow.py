import numpy as np

from src.vision.lane_follow import estimate_lane_error, find_largest_yellow_centroid


def _vertical_band_mask(w=400, h=400, x_center=200, band_w=40) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    x0 = max(0, x_center - band_w // 2)
    x1 = min(w, x_center + band_w // 2)
    mask[:, x0:x1] = 255
    return mask


def test_lane_centered_zero_error():
    mask = _vertical_band_mask(x_center=200)
    res = estimate_lane_error(mask)
    assert res.found
    assert abs(res.error) < 0.05


def test_lane_right_positive_error():
    mask = _vertical_band_mask(x_center=320)
    res = estimate_lane_error(mask)
    assert res.found
    assert res.error > 0.3


def test_lane_left_negative_error():
    mask = _vertical_band_mask(x_center=80)
    res = estimate_lane_error(mask)
    assert res.found
    assert res.error < -0.3


def test_lane_empty_not_found():
    res = estimate_lane_error(np.zeros((400, 400), np.uint8))
    assert not res.found
    assert res.error == 0.0


def test_largest_centroid_finds_band():
    mask = _vertical_band_mask(x_center=150)
    out = find_largest_yellow_centroid(mask, min_area=500)
    assert out is not None
    cx, cy, area = out
    assert 130 <= cx <= 170
    assert area > 500
