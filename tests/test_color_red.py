import numpy as np

from src.vision.realsense_target import classify_color_combo


def _frame_with_red(w=200, h=200):
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (40, 40, 40)
    # BGR: 纯红 = (0, 0, 255)
    rgb[:, :] = (0, 0, 255)
    depth = np.full((h, w), 300, np.uint16)  # 0.3m
    return rgb, depth


def test_red_ratio_detected():
    rgb, depth = _frame_with_red()
    cls = classify_color_combo(rgb, depth_raw=depth, depth_scale=0.001)
    assert cls.red_ratio > 0.8


def test_red_ratio_absent_on_blue():
    rgb = np.zeros((200, 200, 3), np.uint8)
    rgb[:] = (255, 0, 0)  # BGR 蓝
    depth = np.full((200, 200), 300, np.uint16)
    cls = classify_color_combo(rgb, depth_raw=depth, depth_scale=0.001)
    assert cls.red_ratio < 0.05
