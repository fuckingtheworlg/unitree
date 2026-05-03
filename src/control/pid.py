from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PIDParams:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    out_min: float = -1.0
    out_max: float = 1.0
    integral_limit: float = 50.0


class PID:
    def __init__(self, params: PIDParams):
        self.p = params
        self._i = 0.0
        self._prev_err = 0.0
        self._prev_t: Optional[float] = None

    def reset(self) -> None:
        self._i = 0.0
        self._prev_err = 0.0
        self._prev_t = None

    def step(self, error: float, now: Optional[float] = None) -> float:
        if now is None:
            now = time.time()
        if self._prev_t is None:
            dt = 0.0
        else:
            dt = max(now - self._prev_t, 1e-3)
        self._prev_t = now

        self._i += error * dt
        self._i = max(-self.p.integral_limit, min(self.p.integral_limit, self._i))

        d = (error - self._prev_err) / dt if dt > 0 else 0.0
        self._prev_err = error

        out = self.p.kp * error + self.p.ki * self._i + self.p.kd * d
        return max(self.p.out_min, min(self.p.out_max, out))
