"""Passive recorder for manually driving the Go2 through the obstacle zone.

This script only subscribes to DDS state topics. It never initializes
SportClient, VideoClient, or Realsense, so it will not take over motion from
the wireless controller.

Run on the robot:
    python3 -m scripts.record_manual_obstacle --network eth0 --duration 90
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import threading
import time
from pathlib import Path
from typing import Any, Iterable


def _num(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _seq(value: Any, n: int, default: float = math.nan) -> list[float]:
    out: list[float] = []
    try:
        items: Iterable[Any] = list(value)
    except Exception:
        items = []
    for item in list(items)[:n]:
        out.append(_num(item, default))
    while len(out) < n:
        out.append(default)
    return out


def _age(now: float, stamp: float | None) -> float:
    if stamp is None:
        return math.nan
    return max(0.0, now - stamp)


class ManualObstacleRecorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sport_msg = None
        self.sport_t: float | None = None
        self.low_msg = None
        self.low_t: float | None = None
        self.wireless_msg = None
        self.wireless_t: float | None = None
        self.sport_count = 0
        self.low_count = 0
        self.wireless_count = 0

    def on_sport(self, msg) -> None:
        with self.lock:
            self.sport_msg = msg
            self.sport_t = time.time()
            self.sport_count += 1

    def on_low(self, msg) -> None:
        with self.lock:
            self.low_msg = msg
            self.low_t = time.time()
            self.low_count += 1

    def on_wireless(self, msg) -> None:
        with self.lock:
            self.wireless_msg = msg
            self.wireless_t = time.time()
            self.wireless_count += 1

    def snapshot(self) -> tuple[Any, float | None, Any, float | None, Any, float | None]:
        with self.lock:
            return (
                self.sport_msg,
                self.sport_t,
                self.low_msg,
                self.low_t,
                self.wireless_msg,
                self.wireless_t,
            )


def _columns() -> list[str]:
    cols = [
        "t_wall",
        "t_rel",
        "sport_age",
        "low_age",
        "wireless_age",
        "sport_mode",
        "sport_progress",
        "sport_gait_type",
        "sport_body_height",
        "sport_foot_raise_height",
        "sport_yaw_speed",
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "imu_roll",
        "imu_pitch",
        "imu_yaw",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "acc_x",
        "acc_y",
        "acc_z",
    ]
    cols.extend(f"range_obstacle_{i}" for i in range(4))
    cols.extend(f"sport_foot_force_{i}" for i in range(4))
    cols.extend(f"low_foot_force_{i}" for i in range(4))
    cols.extend(f"low_foot_force_est_{i}" for i in range(4))
    cols.extend(f"motor_q_{i}" for i in range(12))
    cols.extend(f"motor_dq_{i}" for i in range(12))
    cols.extend(f"motor_tau_{i}" for i in range(12))
    cols.extend([
        "wireless_lx",
        "wireless_ly",
        "wireless_rx",
        "wireless_ry",
        "wireless_keys",
        "low_tick",
        "power_v",
        "power_a",
    ])
    return cols


def _row(now: float, start: float, rec: ManualObstacleRecorder) -> dict[str, Any]:
    sport, sport_t, low, low_t, wireless, wireless_t = rec.snapshot()
    row: dict[str, Any] = {
        "t_wall": f"{now:.6f}",
        "t_rel": f"{now - start:.6f}",
        "sport_age": f"{_age(now, sport_t):.6f}" if sport_t is not None else "",
        "low_age": f"{_age(now, low_t):.6f}" if low_t is not None else "",
        "wireless_age": f"{_age(now, wireless_t):.6f}" if wireless_t is not None else "",
    }

    if sport is not None:
        imu = getattr(sport, "imu_state", None)
        pos = _seq(getattr(sport, "position", []), 3)
        vel = _seq(getattr(sport, "velocity", []), 3)
        rpy = _seq(getattr(imu, "rpy", []), 3)
        gyro = _seq(getattr(imu, "gyroscope", []), 3)
        acc = _seq(getattr(imu, "accelerometer", []), 3)
        row.update({
            "sport_mode": _int(getattr(sport, "mode", 0)),
            "sport_progress": _num(getattr(sport, "progress", math.nan)),
            "sport_gait_type": _int(getattr(sport, "gait_type", 0)),
            "sport_body_height": _num(getattr(sport, "body_height", math.nan)),
            "sport_foot_raise_height": _num(getattr(sport, "foot_raise_height", math.nan)),
            "sport_yaw_speed": _num(getattr(sport, "yaw_speed", math.nan)),
            "pos_x": pos[0],
            "pos_y": pos[1],
            "pos_z": pos[2],
            "vel_x": vel[0],
            "vel_y": vel[1],
            "vel_z": vel[2],
            "imu_roll": rpy[0],
            "imu_pitch": rpy[1],
            "imu_yaw": rpy[2],
            "gyro_x": gyro[0],
            "gyro_y": gyro[1],
            "gyro_z": gyro[2],
            "acc_x": acc[0],
            "acc_y": acc[1],
            "acc_z": acc[2],
        })
        for i, value in enumerate(_seq(getattr(sport, "range_obstacle", []), 4)):
            row[f"range_obstacle_{i}"] = value
        for i, value in enumerate(_seq(getattr(sport, "foot_force", []), 4, default=0.0)):
            row[f"sport_foot_force_{i}"] = int(value)

    if low is not None:
        for i, value in enumerate(_seq(getattr(low, "foot_force", []), 4, default=0.0)):
            row[f"low_foot_force_{i}"] = int(value)
        for i, value in enumerate(_seq(getattr(low, "foot_force_est", []), 4, default=0.0)):
            row[f"low_foot_force_est_{i}"] = int(value)
        motors = list(getattr(low, "motor_state", []) or [])
        for i in range(12):
            motor = motors[i] if i < len(motors) else None
            row[f"motor_q_{i}"] = _num(getattr(motor, "q", math.nan))
            row[f"motor_dq_{i}"] = _num(getattr(motor, "dq", math.nan))
            row[f"motor_tau_{i}"] = _num(getattr(motor, "tau_est", math.nan))
        row["low_tick"] = _int(getattr(low, "tick", 0))
        row["power_v"] = _num(getattr(low, "power_v", math.nan))
        row["power_a"] = _num(getattr(low, "power_a", math.nan))

    if wireless is not None:
        row.update({
            "wireless_lx": _num(getattr(wireless, "lx", math.nan)),
            "wireless_ly": _num(getattr(wireless, "ly", math.nan)),
            "wireless_rx": _num(getattr(wireless, "rx", math.nan)),
            "wireless_ry": _num(getattr(wireless, "ry", math.nan)),
            "wireless_keys": _int(getattr(wireless, "keys", 0)),
        })

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="eth0", help="DDS network interface")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--output", default="", help="CSV output path")
    args = parser.parse_args()

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
            LowState_,
            SportModeState_,
            WirelessController_,
        )
    except ImportError as e:
        raise SystemExit(f"unitree_sdk2py import failed: {e}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    output = Path(args.output or f"logs/manual_obstacle/manual_obstacle_{ts}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    rec = ManualObstacleRecorder()
    stop = threading.Event()

    def _stop(_sig, _frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"[manual_record] init DDS iface={args.network} domain={args.domain}")
    ChannelFactoryInitialize(args.domain, args.network)

    sport_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    low_sub = ChannelSubscriber("rt/lowstate", LowState_)
    wireless_sub = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
    sport_sub.Init(rec.on_sport, 10)
    low_sub.Init(rec.on_low, 10)
    wireless_sub.Init(rec.on_wireless, 10)

    columns = _columns()
    interval = 1.0 / max(1.0, float(args.rate_hz))
    start = time.time()
    next_t = start
    rows = 0

    print(f"[manual_record] output={output}")
    print(f"[manual_record] GO: use the wireless controller now ({args.duration:.1f}s max)")

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        while not stop.is_set():
            now = time.time()
            if now - start >= float(args.duration):
                break
            writer.writerow(_row(now, start, rec))
            rows += 1
            if rows % max(1, int(args.rate_hz)) == 0:
                f.flush()
                print(
                    "[manual_record] "
                    f"t={now - start:.1f}s rows={rows} "
                    f"sport={rec.sport_count} low={rec.low_count} wireless={rec.wireless_count}",
                    flush=True,
                )
            next_t += interval
            sleep_s = next_t - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.time()

    elapsed = time.time() - start
    summary = {
        "output": str(output),
        "elapsed_sec": elapsed,
        "rows": rows,
        "sport_count": rec.sport_count,
        "low_count": rec.low_count,
        "wireless_count": rec.wireless_count,
        "rate_hz": args.rate_hz,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manual_record] done rows={rows} elapsed={elapsed:.1f}s")
    print(f"[manual_record] csv={output}")
    print(f"[manual_record] summary={summary_path}")


if __name__ == "__main__":
    main()
