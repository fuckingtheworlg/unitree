from __future__ import annotations

import cv2
import numpy as np

from src.control.fsm import State
from src.main import MissionRunner, _DEFAULT_CFG
from src.utils.config import load_config
from src.vision.landmark import LandmarkDetection, LandmarkType
from src.vision.lane_follow import LaneResult
from src.vision.realsense_target import ColorClassification, ColorLabel


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _runner() -> MissionRunner:
    cfg = load_config(_DEFAULT_CFG)
    cfg["landmark"]["obstacle_enabled"] = False
    cfg["landmark"]["dump_consecutive_frames"] = 1
    cfg["landmark"]["dump_proximity_threshold"] = 0.10
    cfg["landmark"]["dump_ring_side"] = "lower"
    cfg["landmark"]["dump_ring_speed"] = 0.20
    cfg["landmark"]["dump_ring_min_follow_sec"] = 1.0
    cfg["landmark"]["dump_ring_min_follow_distance_m"] = 0.20
    cfg["landmark"]["dump_ring_max_sec"] = 8.0
    return MissionRunner(cfg, source=None, robot=None, logger=_Logger())


def _lane() -> LaneResult:
    return LaneResult(found=True, error=0.0, confidence=1.0, centerline_x=[])


def _dump_det(w: int = 320, h: int = 240) -> LandmarkDetection:
    cx, cy, r = w // 2, int(h * 0.62), 58
    return LandmarkDetection(
        type=LandmarkType.DUMP_ZONE,
        confidence=1.0,
        bbox=(cx - r, cy - r, 2 * r, 2 * r),
        extra={
            "center": (float(cx), float(cy)),
            "radius": float(r),
            "image_w": w,
        },
    )


def _ring_mask(w: int = 320, h: int = 240) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    det = _dump_det(w, h)
    cx, cy = det.extra["center"]
    r = int(det.extra["radius"])
    cv2.circle(mask, (int(cx), int(cy)), r, 255, 24)
    return mask


def _blackwhite(z: float = 0.23) -> ColorClassification:
    return ColorClassification(
        label=ColorLabel.BLACK_WHITE,
        confidence=1.0,
        white_ratio=0.7,
        black_ratio=0.35,
        blue_ratio=0.0,
        yellow_ratio=0.0,
        red_ratio=0.0,
        depth_median_m=z,
        depth_valid_ratio=1.0,
        n_valid_pixels=100,
        roi_bbox=(0, 0, 10, 10),
    )


def _runner_with_fast_realsense_dump() -> MissionRunner:
    cfg = load_config(_DEFAULT_CFG)
    cfg["landmark"]["obstacle_enabled"] = True
    cfg["landmark"]["dump_consecutive_frames"] = 1
    cfg["realsense"]["startup_protect_s"] = 3.0
    cfg["realsense"]["n_stable_frames"] = 1
    cfg["realsense"]["min_hits_in_window"] = 1
    runner = MissionRunner(cfg, source=None, robot=None, logger=_Logger())
    runner.rs_cam = object()
    runner._ensure_color_triggers()
    runner.start_time = 0.0
    runner._last_color_cls = _blackwhite()
    return runner


def test_blackwhite_does_not_trigger_dump_before_obstacle_done():
    runner = _runner_with_fast_realsense_dump()
    runner.fsm.state = State.FOLLOW_LANE

    runner._dispatch(_lane(), dump=None, fork=None, stair=None, dock=None)

    assert runner.fsm.state == State.FOLLOW_LANE
    assert not runner.fsm.flags.dumped


def test_blackwhite_after_obstacle_done_directly_triggers_dump_action():
    runner = _runner_with_fast_realsense_dump()
    runner.fsm.state = State.FOLLOW_LANE
    runner.fsm.flags.obstacle_avoided = True
    runner.fsm.flags.blue_ring_done = True

    runner._dispatch(_lane(), dump=None, fork=None, stair=None, dock=None)

    assert runner.fsm.state == State.DUMP_ACTION
    assert not runner.fsm.flags.dumped


def test_dump_detection_enters_follow_dump_ring():
    runner = _runner()
    runner.fsm.state = State.FOLLOW_LANE

    runner._dispatch(_lane(), _dump_det(), fork=None, stair=None, dock=None)

    assert runner.fsm.state == State.FOLLOW_DUMP_RING


def test_dump_ring_lane_uses_configured_lower_side():
    runner = _runner()
    runner._last_mask = _ring_mask()

    lane = runner._dump_ring_lane(_dump_det())

    assert lane.found
    assert lane.confidence > 0.2
    assert any(x >= 0 for x in lane.centerline_x)


def test_follow_dump_ring_requires_minimum_progress_not_blackwhite(monkeypatch):
    runner = _runner()
    runner.rs_cam = object()
    runner.fsm.state = State.FOLLOW_DUMP_RING
    runner.fsm.entered_at = 100.0
    runner._last_mask = _ring_mask()
    runner._last_color_cls = None

    now = [100.0]
    monkeypatch.setattr("src.main.time.time", lambda: now[0])

    vx, vyaw = runner._follow_dump_ring(_lane(), _dump_det())
    assert vx > 0.0
    assert runner.fsm.state == State.FOLLOW_DUMP_RING

    now[0] = 100.5
    runner._follow_dump_ring(_lane(), _dump_det())
    assert runner.fsm.state == State.FOLLOW_DUMP_RING

    now[0] = 101.2
    vx, vyaw = runner._follow_dump_ring(_lane(), _dump_det())
    assert (vx, vyaw) == (0.0, 0.0)
    assert runner.fsm.state == State.DUMP_ACTION
