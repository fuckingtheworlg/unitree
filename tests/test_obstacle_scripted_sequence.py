from __future__ import annotations

from src.control.fsm import State
from src.main import MissionRunner, _DEFAULT_CFG, _apply_start_stage
from src.utils.config import load_config
from src.vision.lane_follow import LaneResult
from src.vision.realsense_target import ColorClassification, ColorLabel


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class _OneFrameSource:
    def __init__(self):
        self._used = False

    def read(self):
        if self._used:
            return None
        self._used = True
        return object()


class _Robot:
    dry_run = False

    def __init__(self):
        self.moves = []

    def stand_up(self):
        pass

    def set_speed_level(self, _level):
        pass

    def set_velocity(self, vx, vy, vyaw):
        self.moves.append((vx, vy, vyaw))

    def shutdown(self):
        pass


def _runner() -> MissionRunner:
    cfg = load_config(_DEFAULT_CFG)
    cfg["landmark"]["obstacle_seq_turn_yaw"] = 0.6
    cfg["landmark"]["obstacle_seq_turn_vx"] = 0.0
    cfg["landmark"]["obstacle_seq_straight_vx"] = 0.22
    cfg["landmark"]["obstacle_seq_turn90_sec"] = 0.5
    cfg["landmark"]["obstacle_seq_right90_sec"] = 0.5
    cfg["landmark"]["obstacle_seq_left90a_sec"] = 0.5
    cfg["landmark"]["obstacle_seq_left90b_sec"] = 0.5
    cfg["landmark"]["obstacle_seq_right_final_sec"] = 0.5
    cfg["landmark"]["obstacle_seq_strafe_after_straight3_sec"] = 0.0
    cfg["landmark"]["obstacle_seq_final_forward_sec"] = 0.0
    cfg["landmark"]["obstacle_seq_straight3_sec"] = 1.0
    cfg["landmark"]["obstacle_seq_straight_grace_sec"] = 0.0
    cfg["landmark"]["obstacle_seq_red_consecutive"] = 2
    cfg["landmark"]["obstacle_seq_max_sec"] = 20.0
    return MissionRunner(cfg, source=None, robot=None, logger=_Logger())


def _lane() -> LaneResult:
    return LaneResult(found=True, error=0.0, confidence=1.0, centerline_x=[])


def _color(red_ratio: float = 0.0) -> ColorClassification:
    return ColorClassification(
        label=ColorLabel.NONE,
        confidence=0.0,
        white_ratio=0.0,
        black_ratio=0.0,
        blue_ratio=0.0,
        yellow_ratio=0.0,
        red_ratio=red_ratio,
        depth_median_m=0.23,
        depth_valid_ratio=1.0,
        n_valid_pixels=100,
        roi_bbox=(0, 0, 10, 10),
    )


def test_default_obstacle_profile_uses_color_recognition():
    cfg = load_config(_DEFAULT_CFG)

    assert cfg["landmark"]["obstacle_seq_use_red"] is True
    assert cfg["landmark"]["obstacle_seq_turn_yaw"] == 0.6
    assert cfg["landmark"]["obstacle_seq_straight_vx"] == 0.22
    assert cfg["landmark"]["obstacle_seq_turn90_sec"] == 2.6
    assert cfg["landmark"]["obstacle_seq_right90_sec"] == 2.6
    assert cfg["landmark"]["obstacle_seq_left90a_sec"] == 2.6
    assert cfg["landmark"]["obstacle_seq_left90b_sec"] == 2.6
    assert cfg["landmark"]["obstacle_seq_right_final_sec"] == 2.6
    assert cfg["landmark"]["obstacle_seq_straight3_sec"] == 3.0
    assert cfg["landmark"]["obstacle_seq_strafe_after_straight3_sec"] == 0.0
    assert cfg["landmark"]["obstacle_seq_final_forward_sec"] == 0.0


def test_default_obstacle_sequence_waits_for_red_in_internal_phases():
    cfg = load_config(_DEFAULT_CFG)

    assert cfg["landmark"]["obstacle_seq_red_consecutive"] == 3
    assert cfg["landmark"]["obstacle_seq_phase_max_sec"] == 8.0


def test_default_dump_proximity_threshold_is_more_permissive():
    cfg = load_config(_DEFAULT_CFG)

    assert cfg["landmark"]["dump_proximity_threshold"] == 0.58


def test_obstacle_entry_is_blocked_during_startup_window(monkeypatch):
    runner = _runner()
    runner.cfg["realsense"]["startup_protect_s"] = 3.0

    monkeypatch.setattr("src.main.time.time", lambda: 10.0)
    runner.start_time = 8.2
    assert not runner._obstacle_entry_allowed()

    runner.start_time = 6.9
    assert runner._obstacle_entry_allowed()


def test_start_after_obstacle_only_skips_obstacle_flags():
    runner = _runner()

    _apply_start_stage(runner, "after-obstacle", _Logger())

    assert runner.fsm.flags.obstacle_avoided
    assert runner.fsm.flags.blue_ring_done
    assert not runner.fsm.flags.dumped
    assert not runner.fsm.flags.stair_climbed


def test_run_forwards_obstacle_lateral_velocity_to_robot(monkeypatch):
    cfg = load_config(_DEFAULT_CFG)
    cfg["robot"]["startup_grace_sec"] = 0.0
    cfg["mission"]["max_duration_sec"] = 5.0
    robot = _Robot()
    runner = MissionRunner(cfg, source=_OneFrameSource(), robot=robot, logger=_Logger())

    def _fake_step(_frame):
        runner._cmd_vy = -0.2
        return 0.0, 0.0, object()

    runner._step = _fake_step
    runner.run(display=False)

    assert robot.moves == [(0.0, -0.2, 0.0)]


def test_scripted_obstacle_aborts_if_straight_waits_too_long_for_red(monkeypatch):
    runner = _runner()
    runner.cfg["landmark"]["obstacle_seq_use_red"] = True
    runner.cfg["landmark"]["obstacle_seq_phase_max_sec"] = 1.0
    runner.fsm.state = State.FOLLOW_OBSTACLE_RING
    runner.fsm.entered_at = 100.0
    runner._last_color_cls = _color(red_ratio=0.0)
    now = [100.0]
    monkeypatch.setattr("src.main.time.time", lambda: now[0])

    vx, vyaw = runner._obstacle_scripted_step(_lane())
    assert vx == 0.0
    assert vyaw < 0.0

    now[0] = 100.6
    vx, vyaw = runner._obstacle_scripted_step(_lane())
    assert vx > 0.0
    assert vyaw == 0.0
    assert runner._obstacle_phase == "straight1"

    now[0] = 101.7
    vx, vyaw = runner._obstacle_scripted_step(_lane())
    assert (vx, vyaw) == (0.0, 0.0)
    assert runner.fsm.state == State.FOLLOW_LANE
    assert runner.fsm.flags.obstacle_avoided
    assert runner.fsm.flags.blue_ring_done


def test_scripted_obstacle_straight2_timeout_continues_final_maneuver(monkeypatch):
    runner = _runner()
    runner.cfg["landmark"]["obstacle_seq_use_red"] = True
    runner.cfg["landmark"]["obstacle_seq_phase_max_sec"] = 1.0
    runner.fsm.state = State.FOLLOW_OBSTACLE_RING
    runner.fsm.entered_at = 100.0
    runner._obstacle_seq_active = True
    runner._obstacle_phase = "straight2"
    runner._obstacle_phase_t = 100.0
    runner._last_color_cls = _color(red_ratio=0.0)
    monkeypatch.setattr("src.main.time.time", lambda: 101.2)

    vx, vyaw = runner._obstacle_scripted_step(_lane())

    assert vx == 0.0
    assert vyaw > 0.0
    assert runner._obstacle_phase == "left90b"
    assert runner.fsm.state == State.FOLLOW_OBSTACLE_RING
    assert not runner.fsm.flags.obstacle_avoided


def test_scripted_obstacle_sequence_uses_red_red_then_final_right(monkeypatch):
    runner = _runner()
    runner.cfg["landmark"]["obstacle_seq_use_red"] = True
    runner.cfg["landmark"]["obstacle_seq_phase_max_sec"] = 5.0
    runner.fsm.state = State.FOLLOW_OBSTACLE_RING
    runner.fsm.entered_at = 200.0
    now = [200.0]
    monkeypatch.setattr("src.main.time.time", lambda: now[0])

    runner._last_color_cls = _color(0.0)
    assert runner._obstacle_scripted_step(_lane()) == (0.0, -0.6)

    now[0] = 200.6
    assert runner._obstacle_scripted_step(_lane()) == (0.22, 0.0)
    assert runner._obstacle_phase == "straight1"

    runner._last_color_cls = _color(0.7)
    now[0] = 200.7
    assert runner._obstacle_scripted_step(_lane()) == (0.22, 0.0)
    now[0] = 200.8
    assert runner._obstacle_scripted_step(_lane()) == (0.0, 0.6)
    assert runner._obstacle_phase == "left90a"

    runner._last_color_cls = _color(0.0)
    now[0] = 201.4
    assert runner._obstacle_scripted_step(_lane()) == (0.22, 0.0)
    assert runner._obstacle_phase == "straight2"

    runner._last_color_cls = _color(0.7)
    now[0] = 201.5
    assert runner._obstacle_scripted_step(_lane()) == (0.22, 0.0)
    now[0] = 201.6
    assert runner._obstacle_scripted_step(_lane()) == (0.0, 0.6)
    assert runner._obstacle_phase == "left90b"

    runner._last_color_cls = _color(0.0)
    now[0] = 202.2
    assert runner._obstacle_scripted_step(_lane()) == (0.22, 0.0)
    assert runner._obstacle_phase == "straight3"

    now[0] = 203.3
    assert runner._obstacle_scripted_step(_lane()) == (0.0, -0.6)
    assert runner._obstacle_phase == "right_final"

    now[0] = 203.9
    assert runner._obstacle_scripted_step(_lane()) == (0.0, 0.0)
    assert runner.fsm.state == State.FOLLOW_LANE


def test_scripted_obstacle_default_timed_sequence_ignores_red(monkeypatch):
    runner = _runner()
    runner.cfg["landmark"]["obstacle_seq_use_red"] = False
    runner.cfg["landmark"]["obstacle_seq_turn90_sec"] = 0.5
    runner.cfg["landmark"]["obstacle_seq_straight1_sec"] = 0.7
    runner.cfg["landmark"]["obstacle_seq_straight2_sec"] = 0.9
    runner.cfg["landmark"]["obstacle_seq_straight3_sec"] = 1.0
    runner.cfg["landmark"]["obstacle_seq_right_final_sec"] = 0.5
    runner.fsm.state = State.FOLLOW_OBSTACLE_RING
    runner.fsm.entered_at = 300.0
    runner._last_color_cls = _color(0.0)
    now = [300.0]
    monkeypatch.setattr("src.main.time.time", lambda: now[0])

    assert runner._obstacle_scripted_step(_lane()) == (0.0, -0.6)

    now[0] = 300.6
    assert runner._obstacle_scripted_step(_lane()) == (0.22, 0.0)
    assert runner._obstacle_phase == "straight1"

    runner._last_color_cls = _color(0.0)
    now[0] = 301.2
    assert runner._obstacle_scripted_step(_lane()) == (0.22, 0.0)
    assert runner._obstacle_phase == "straight1"

    now[0] = 301.31
    assert runner._obstacle_scripted_step(_lane()) == (0.0, 0.6)
    assert runner._obstacle_phase == "left90a"

    now[0] = 301.9
    assert runner._obstacle_scripted_step(_lane()) == (0.22, 0.0)
    assert runner._obstacle_phase == "straight2"

    now[0] = 302.8
    assert runner._obstacle_scripted_step(_lane()) == (0.0, 0.6)
    assert runner._obstacle_phase == "left90b"


def test_scripted_obstacle_can_issue_strafe_and_final_forward(monkeypatch):
    runner = _runner()
    runner.cfg["landmark"]["obstacle_seq_use_red"] = False
    runner.cfg["landmark"]["obstacle_seq_turn_yaw"] = 0.9
    runner.cfg["landmark"]["obstacle_seq_turn90_sec"] = 0.5
    runner.cfg["landmark"]["obstacle_seq_right90_sec"] = 0.5
    runner.cfg["landmark"]["obstacle_seq_left90a_sec"] = 0.5
    runner.cfg["landmark"]["obstacle_seq_left90b_sec"] = 0.5
    runner.cfg["landmark"]["obstacle_seq_right_final_sec"] = 0.5
    runner.cfg["landmark"]["obstacle_seq_straight1_sec"] = 0.5
    runner.cfg["landmark"]["obstacle_seq_straight2_sec"] = 0.5
    runner.cfg["landmark"]["obstacle_seq_straight3_sec"] = 0.5
    runner.cfg["landmark"]["obstacle_seq_strafe_after_straight3_sec"] = 0.5
    runner.cfg["landmark"]["obstacle_seq_strafe_vy"] = -0.2
    runner.cfg["landmark"]["obstacle_seq_final_forward_sec"] = 0.5
    runner.fsm.state = State.FOLLOW_OBSTACLE_RING
    runner.fsm.entered_at = 400.0
    runner._last_color_cls = _color(0.0)
    now = [400.0]
    monkeypatch.setattr("src.main.time.time", lambda: now[0])

    runner._obstacle_scripted_step(_lane())
    now[0] = 400.6
    runner._obstacle_scripted_step(_lane())
    now[0] = 401.2
    runner._obstacle_scripted_step(_lane())
    now[0] = 401.8
    runner._obstacle_scripted_step(_lane())
    now[0] = 402.4
    runner._obstacle_scripted_step(_lane())
    now[0] = 403.0
    runner._obstacle_scripted_step(_lane())
    now[0] = 403.6
    vx, vyaw = runner._obstacle_scripted_step(_lane())
    assert runner._obstacle_phase == "strafe_right"
    assert vx == 0.0
    assert vyaw == 0.0
    assert runner._cmd_vy == -0.2

    now[0] = 404.2
    vx, vyaw = runner._obstacle_scripted_step(_lane())
    assert runner._obstacle_phase == "right_final"
    assert runner._cmd_vy == 0.0
    assert vyaw < 0.0

    now[0] = 404.8
    vx, vyaw = runner._obstacle_scripted_step(_lane())
    assert runner._obstacle_phase == "final_forward"
    assert vx > 0.0
    assert vyaw == 0.0

    now[0] = 405.4
    vx, vyaw = runner._obstacle_scripted_step(_lane())
    assert (vx, vyaw) == (0.0, 0.0)
    assert runner.fsm.state == State.FOLLOW_LANE
