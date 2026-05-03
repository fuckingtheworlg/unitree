import time

from src.control.pid import PID, PIDParams


def test_pid_p_only():
    pid = PID(PIDParams(kp=1.0, ki=0.0, kd=0.0, out_min=-10, out_max=10))
    out = pid.step(2.0, now=1.0)
    assert abs(out - 2.0) < 1e-6


def test_pid_saturation():
    pid = PID(PIDParams(kp=100.0, ki=0.0, kd=0.0, out_min=-1, out_max=1))
    out = pid.step(2.0, now=1.0)
    assert out == 1.0
    out = pid.step(-2.0, now=2.0)
    assert out == -1.0


def test_pid_reset():
    pid = PID(PIDParams(kp=1.0, ki=1.0, kd=0.0, integral_limit=100))
    pid.step(1.0, now=1.0)
    pid.step(1.0, now=2.0)
    pid.reset()
    out = pid.step(0.0, now=3.0)
    assert abs(out) < 1e-6
