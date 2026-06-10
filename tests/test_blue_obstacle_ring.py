import cv2
import numpy as np

from src.vision.landmark import LandmarkType, detect_blue_obstacle_ring, detect_blue_rect
from src.vision.realsense_lane import estimate_bottom_blue_ring_lane


def _blue_ring_frame(w=360, h=260):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (40, 40, 40)
    center = (w // 2, h // 2)
    cv2.circle(img, center, 82, (255, 0, 0), 24)
    cv2.circle(img, center, 55, (245, 245, 245), -1)
    cv2.rectangle(img, (center[0] - 34, center[1] - 8), (center[0] + 34, center[1] + 8), (15, 15, 15), -1)
    cv2.rectangle(img, (center[0] - 22, center[1] + 16), (center[0] + 22, center[1] + 28), (15, 15, 15), -1)
    return img


def test_detect_blue_obstacle_ring():
    det = detect_blue_obstacle_ring(_blue_ring_frame())
    assert det is not None
    assert det.type == LandmarkType.BLUE_RING_OBSTACLE
    assert det.extra["ring_blue"] > 0.16
    assert det.extra["inner_white"] > 0.25
    assert det.extra["inner_black"] > 0.01


def test_blue_obstacle_ring_rejects_blue_rectangle():
    img = np.zeros((220, 360, 3), np.uint8)
    img[:] = (35, 35, 35)
    cv2.rectangle(img, (70, 55), (290, 165), (255, 0, 0), 18)
    cv2.rectangle(img, (95, 80), (265, 140), (245, 245, 245), -1)
    cv2.putText(img, "P", (160, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    assert detect_blue_rect(img) is not None
    assert detect_blue_obstacle_ring(img) is None


def test_blue_obstacle_ring_rejects_solid_blue():
    img = np.zeros((220, 320, 3), np.uint8)
    img[:] = (255, 0, 0)
    assert detect_blue_obstacle_ring(img) is None


def test_bottom_blue_ring_lane_reports_right_error():
    img = np.zeros((240, 320, 3), np.uint8)
    img[:] = (35, 35, 35)
    img[:, 220:250] = (255, 0, 0)
    img[:, 95:135] = (245, 245, 245)
    depth = np.full((240, 320), 260, np.uint16)

    lane = estimate_bottom_blue_ring_lane(
        img,
        depth_raw=depth,
        depth_scale=0.001,
        direction="right",
        min_blue_ratio=0.005,
    )

    assert lane.found
    assert lane.error > 0.25
