from __future__ import annotations

from src.control.fsm import State
from src.main import MissionRunner, _DEFAULT_CFG
from src.utils.config import load_config
from src.vision.lane_follow import LaneResult
from src.vision.realsense_target import ColorClassification, ColorLabel


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _runner() -> MissionRunner:
    cfg = load_config(_DEFAULT_CFG)
    cfg["realsense"]["dock_blind_walk_speed"] = 0.25
    cfg["realsense"]["dock_post_stair_settle_sec"] = 0.8
    return MissionRunner(cfg, source=None, robot=None, logger=_Logger())


def _bluewhite(z: float = 0.23) -> ColorClassification:
    return ColorClassification(
        label=ColorLabel.BLUE_WHITE,
        confidence=1.0,
        white_ratio=0.7,
        black_ratio=0.05,
        blue_ratio=0.35,
        yellow_ratio=0.0,
        red_ratio=0.0,
        depth_median_m=z,
        depth_valid_ratio=1.0,
        n_valid_pixels=100,
        roi_bbox=(0, 0, 10, 10),
    )


def test_after_stair_blind_walk_ignores_right_yellow_when_realsense_inactive():
    runner = _runner()
    runner.fsm.state = State.FOLLOW_LANE
    runner.fsm.flags.dumped = True
    runner.fsm.flags.stair_climbed = True
    runner.rs_cam = None

    right_yellow_lane = LaneResult(
        found=True,
        error=0.85,
        confidence=1.0,
        centerline_x=[],
    )

    vx, vyaw = runner._dispatch(
        right_yellow_lane,
        dump=None,
        fork=None,
        stair=None,
        dock=None,
    )

    assert vx == 0.25
    assert vyaw == 0.0
    assert runner.fsm.state == State.FOLLOW_LANE


def test_stair_completion_starts_final_straight_settle_window(monkeypatch):
    runner = _runner()
    runner.fsm.state = State.APPROACH_STAIR
    runner.fsm.entered_at = 100.0
    runner._approach_track_state = State.APPROACH_STAIR
    runner._approach_last_tick = 100.0
    runner._stair_mode_entered = True
    runner.cfg["landmark"]["stair_through_distance_m"] = 0.0
    monkeypatch.setattr("src.main.time.time", lambda: 100.0)

    vx, vyaw = runner._dispatch(
        LaneResult(True, 0.0, 1.0, []),
        dump=None,
        fork=None,
        stair=None,
        dock=None,
    )

    assert (vx, vyaw) == (0.0, 0.0)
    assert runner.fsm.state == State.FOLLOW_LANE
    assert runner._post_stair_straight_ready_at == 100.8


def test_final_straight_waits_for_settle_before_moving(monkeypatch):
    runner = _runner()
    runner.fsm.state = State.FOLLOW_LANE
    runner.fsm.flags.dumped = True
    runner.fsm.flags.stair_climbed = True
    runner._post_stair_straight_ready_at = 100.8
    runner.rs_cam = None

    right_yellow_lane = LaneResult(True, 0.85, 1.0, [])
    now = [100.3]
    monkeypatch.setattr("src.main.time.time", lambda: now[0])

    vx, vyaw = runner._dispatch(
        right_yellow_lane,
        dump=None,
        fork=None,
        stair=None,
        dock=None,
    )
    assert (vx, vyaw) == (0.0, 0.0)

    now[0] = 100.9
    vx, vyaw = runner._dispatch(
        right_yellow_lane,
        dump=None,
        fork=None,
        stair=None,
        dock=None,
    )
    assert (vx, vyaw) == (0.25, 0.0)


def test_bluewhite_after_dump_directly_enters_stair_mode_without_delay(monkeypatch):
    runner = _runner()
    runner.rs_cam = object()
    runner._ensure_color_triggers()
    runner.fsm.state = State.FOLLOW_LANE
    runner.fsm.flags.dumped = True
    runner._dump_done_at = 100.0
    runner._last_color_cls = _bluewhite()
    runner.cfg["realsense"]["n_stable_frames"] = 1
    runner.cfg["realsense"]["min_hits_in_window"] = 1
    runner.cfg["realsense"]["dump_to_stair_min_s"] = 15.0
    runner._color_triggers = {}
    runner._ensure_color_triggers()
    runner.start_time = 0.0
    monkeypatch.setattr("src.main.time.time", lambda: 105.0)

    runner._dispatch(
        LaneResult(True, 0.0, 1.0, []),
        dump=None,
        fork=None,
        stair=None,
        dock=None,
    )

    assert runner.fsm.state == State.APPROACH_STAIR


def test_bluewhite_after_stair_enters_dock_post_trigger():
    runner = _runner()
    runner.rs_cam = object()
    runner.cfg["realsense"]["n_stable_frames"] = 1
    runner.cfg["realsense"]["min_hits_in_window"] = 1
    runner._ensure_color_triggers()
    runner.fsm.state = State.FOLLOW_LANE
    runner.fsm.flags.dumped = True
    runner.fsm.flags.stair_climbed = True
    runner._last_color_cls = _bluewhite()

    runner._dispatch(
        LaneResult(True, 0.0, 1.0, []),
        dump=None,
        fork=None,
        stair=None,
        dock=None,
    )

    assert runner.fsm.state == State.APPROACH_DOCK
    assert runner._dock_post_trigger_active
