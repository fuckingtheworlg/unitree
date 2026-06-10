from scripts.test_lowlevel_lean_left import (
    LEG_ID,
    make_side_lean_target,
    side_lean_joint_indices,
    validate_delta_limits,
)


def test_right_side_lean_only_changes_right_thigh_and_calf_joints():
    base = [float(i) for i in range(12)]

    target = make_side_lean_target(base, "right", thigh_delta=0.18, calf_delta=-0.32)

    changed = {idx for idx, (old, new) in enumerate(zip(base, target)) if old != new}
    assert changed == {
        LEG_ID["FR_1"],
        LEG_ID["FR_2"],
        LEG_ID["RR_1"],
        LEG_ID["RR_2"],
    }
    assert target[LEG_ID["FR_1"]] == base[LEG_ID["FR_1"]] + 0.18
    assert target[LEG_ID["RR_1"]] == base[LEG_ID["RR_1"]] + 0.18
    assert target[LEG_ID["FR_2"]] == base[LEG_ID["FR_2"]] - 0.32
    assert target[LEG_ID["RR_2"]] == base[LEG_ID["RR_2"]] - 0.32


def test_left_side_lean_only_changes_left_thigh_and_calf_joints():
    base = [float(i) for i in range(12)]

    target = make_side_lean_target(base, "left", thigh_delta=0.1, calf_delta=-0.2)

    changed = {idx for idx, (old, new) in enumerate(zip(base, target)) if old != new}
    assert changed == {
        LEG_ID["FL_1"],
        LEG_ID["FL_2"],
        LEG_ID["RL_1"],
        LEG_ID["RL_2"],
    }


def test_side_lean_joint_indices_rejects_unknown_side():
    try:
        side_lean_joint_indices("front")
    except ValueError as e:
        assert "unknown lean side" in str(e)
    else:
        raise AssertionError("unknown side should be rejected")


def test_delta_limit_rejects_large_motion():
    validate_delta_limits(0.45, -0.45, 0.45)

    try:
        validate_delta_limits(0.46, 0.0, 0.45)
    except ValueError as e:
        assert "thigh_delta" in str(e)
    else:
        raise AssertionError("oversized thigh delta should be rejected")
