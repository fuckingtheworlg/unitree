"""Go2 低层接管/左右侧下沉倾倒实验脚本.

默认只做 HOLD: 读取当前 12 个关节角, 以 LowCmd 高频保持, 不释放 mcf, 不弯腿。

危险模式:
  --release-mcf 会释放当前高层模式, 由本脚本用 LowCmd 接管 12 个关节。
  --lean-side left/right 会在接管后让同侧前后腿大腿和小腿缓慢下沉,
  形成机身侧倾。

推荐测试顺序:
  1. 只看低层状态:
     .venv/bin/python -m scripts.test_lowlevel_lean_left --network en5
  2. 低层接管保持, 不倾斜:
     .venv/bin/python -m scripts.test_lowlevel_lean_left --network en5 \
       --release-mcf --confirm I_UNDERSTAND_LOWLEVEL_RISK
  3. 小幅右侧下沉:
     .venv/bin/python -m scripts.test_lowlevel_lean_left --network en5 \
       --release-mcf --lean-side right --thigh-delta 0.18 --calf-delta -0.32 \
       --confirm I_UNDERSTAND_LOWLEVEL_RISK

注意:
  低层控制必须同时控制 12 个关节。不要只给左腿发命令, 否则右侧也可能失去支撑。
  默认 delta 按官方 Go2 stand 示例中的下蹲方向设置: thigh 增大, calf 减小。
  实测低层 release + hold 已出现后腿下沉, 不建议比赛使用; 本脚本只供验证。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import List, Optional


CONFIRM_TEXT = "I_UNDERSTAND_LOWLEVEL_RISK"

LEG_ID = {
    "FR_0": 0, "FR_1": 1, "FR_2": 2,
    "FL_0": 3, "FL_1": 4, "FL_2": 5,
    "RR_0": 6, "RR_1": 7, "RR_2": 8,
    "RL_0": 9, "RL_1": 10, "RL_2": 11,
}

SIDE_LEAN_JOINTS = {
    "right": {
        "thigh": ("FR_1", "RR_1"),
        "calf": ("FR_2", "RR_2"),
    },
    "left": {
        "thigh": ("FL_1", "RL_1"),
        "calf": ("FL_2", "RL_2"),
    },
}

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0


class LowLevelHold:
    def __init__(self, kp: float, kd: float, hz: int):
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        self.kp = float(kp)
        self.kd = float(kd)
        self.interval = 1.0 / max(1, int(hz))
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()
        self.low_state = None
        self.target_q: Optional[List[float]] = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.publisher = None
        self.subscriber = None
        self.sent = 0

    def init_dds_io(self) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_

        self._init_low_cmd()
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._low_state_handler, 10)

    def wait_low_state(self, timeout_s: float = 3.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.low_state is not None:
                return True
            time.sleep(0.02)
        return False

    def capture_current_target(self) -> List[float]:
        if self.low_state is None:
            raise RuntimeError("low_state not ready")
        q = [float(self.low_state.motor_state[i].q) for i in range(12)]
        with self.lock:
            self.target_q = list(q)
        return q

    def start(self) -> None:
        self.stop.clear()
        self.thread = threading.Thread(target=self._loop, name="lowcmd-hold", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def set_target(self, q: List[float]) -> None:
        if len(q) != 12:
            raise ValueError("target q must have 12 joints")
        with self.lock:
            self.target_q = [float(v) for v in q]

    def interpolate_to(self, target: List[float], duration_s: float) -> None:
        with self.lock:
            start = list(self.target_q or target)
        steps = max(1, int(duration_s / self.interval))
        for n in range(steps):
            if self.stop.is_set():
                return
            a = (n + 1) / float(steps)
            q = [(1.0 - a) * s + a * t for s, t in zip(start, target)]
            self.set_target(q)
            time.sleep(self.interval)

    def _init_low_cmd(self) -> None:
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = POS_STOP_F
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].dq = VEL_STOP_F
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0

    def _low_state_handler(self, msg) -> None:
        self.low_state = msg

    def _loop(self) -> None:
        next_t = time.monotonic()
        while not self.stop.is_set():
            with self.lock:
                q = list(self.target_q) if self.target_q is not None else None
            if q is not None and self.publisher is not None:
                for i in range(12):
                    self.low_cmd.motor_cmd[i].mode = 0x01
                    self.low_cmd.motor_cmd[i].q = float(q[i])
                    self.low_cmd.motor_cmd[i].dq = 0.0
                    self.low_cmd.motor_cmd[i].kp = self.kp
                    self.low_cmd.motor_cmd[i].kd = self.kd
                    self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.crc = self.crc.Crc(self.low_cmd)
                self.publisher.Write(self.low_cmd)
                self.sent += 1
            next_t += self.interval
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()


def side_lean_joint_indices(side: str) -> dict[str, tuple[int, ...]]:
    side = side.lower()
    if side not in SIDE_LEAN_JOINTS:
        raise ValueError(f"unknown lean side: {side}")

    joints = SIDE_LEAN_JOINTS[side]
    return {
        "thigh": tuple(LEG_ID[name] for name in joints["thigh"]),
        "calf": tuple(LEG_ID[name] for name in joints["calf"]),
    }


def validate_delta_limits(
    thigh_delta: float,
    calf_delta: float,
    max_abs_delta: float,
) -> None:
    max_abs_delta = abs(float(max_abs_delta))
    if max_abs_delta <= 0:
        raise ValueError("--max-abs-delta must be positive")
    if abs(float(thigh_delta)) > max_abs_delta:
        raise ValueError(
            f"thigh_delta={thigh_delta:+.3f} exceeds max_abs_delta={max_abs_delta:.3f}"
        )
    if abs(float(calf_delta)) > max_abs_delta:
        raise ValueError(
            f"calf_delta={calf_delta:+.3f} exceeds max_abs_delta={max_abs_delta:.3f}"
        )


def make_side_lean_target(
    base: List[float],
    side: str,
    thigh_delta: float,
    calf_delta: float,
) -> List[float]:
    if len(base) != 12:
        raise ValueError("base q must have 12 joints")

    target = list(base)
    indices = side_lean_joint_indices(side)
    for idx in indices["thigh"]:
        target[idx] += float(thigh_delta)
    for idx in indices["calf"]:
        target[idx] += float(calf_delta)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="eth0")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--kp", type=float, default=40.0)
    parser.add_argument("--kd", type=float, default=4.0)
    parser.add_argument("--hz", type=int, default=500)
    parser.add_argument("--prehold-sec", type=float, default=1.0)
    parser.add_argument("--hold-sec", type=float, default=3.0)
    parser.add_argument("--lean-left", action="store_true",
                        help="兼容旧参数: 等同 --lean-side left")
    parser.add_argument("--lean-side", choices=["none", "left", "right"], default="none",
                        help="低层侧倾方向. right=右前/右后腿下沉; left=左前/左后腿下沉")
    parser.add_argument("--lean-duration-sec", type=float, default=2.0)
    parser.add_argument("--lean-hold-sec", type=float, default=2.0)
    parser.add_argument("--restore-sec", type=float, default=2.0)
    parser.add_argument("--thigh-delta", type=float, default=0.18,
                        help="同侧大腿关节相对当前姿态的增量; 默认按下蹲方向小幅增加")
    parser.add_argument("--calf-delta", type=float, default=-0.32,
                        help="同侧小腿关节相对当前姿态的增量; 默认按下蹲方向小幅减小")
    parser.add_argument("--max-abs-delta", type=float, default=0.45,
                        help="单个 thigh/calf delta 的安全上限, 默认 0.45rad")
    parser.add_argument("--release-mcf", action="store_true")
    parser.add_argument("--restore-mode", default="mcf",
                        help="释放高层后, 结束前尝试恢复的模式名; 默认 mcf")
    parser.add_argument("--confirm", default="",
                        help=f"执行 --release-mcf 时必须填 {CONFIRM_TEXT}")
    args = parser.parse_args()

    if args.release_mcf and args.confirm != CONFIRM_TEXT:
        print(f"拒绝执行 release: 请确认现场安全后加 --confirm {CONFIRM_TEXT}")
        return 2
    try:
        validate_delta_limits(args.thigh_delta, args.calf_delta, args.max_abs_delta)
    except ValueError as e:
        print(f"拒绝执行: {e}")
        return 2

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    except ImportError as e:
        print(f"unitree_sdk2py 未装: {e}", file=sys.stderr)
        return 2

    print(f"[lowlean] DDS init iface={args.network}")
    ChannelFactoryInitialize(args.domain, args.network)

    ctrl = LowLevelHold(kp=args.kp, kd=args.kd, hz=args.hz)
    ctrl.init_dds_io()
    print("[lowlean] waiting low_state...")
    if not ctrl.wait_low_state(timeout_s=3.0):
        print("[lowlean] low_state 超时, 不执行")
        return 1

    base = ctrl.capture_current_target()
    print("[lowlean] current q:")
    print("  " + " ".join(f"{v:+.3f}" for v in base))
    ctrl.start()
    print(f"[lowlean] LowCmd hold thread started @ {args.hz}Hz")
    time.sleep(float(args.prehold_sec))

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    code, data = msc.CheckMode()
    print(f"[lowlean] CheckMode: code={code}, data={data}")
    released = False
    restore_ok = not args.release_mcf

    try:
        if args.release_mcf:
            name = (data or {}).get("name", "")
            if name:
                print(f"[lowlean] ReleaseMode('{name}') while LowCmd is already holding...")
                code, raw = msc.ReleaseMode()
                print(f"[lowlean] ReleaseMode return: code={code}, data={raw}")
                if code != 0:
                    return 1
                released = True
                time.sleep(0.5)
            else:
                print("[lowlean] current mode is empty, skip ReleaseMode")
                released = True
        else:
            print("[lowlean] --release-mcf 未启用: 只发布 hold, 不释放高层模式")

        lean_side = "left" if args.lean_left else args.lean_side
        if lean_side != "none":
            if not args.release_mcf:
                print("[lowlean] --lean-side 需要同时加 --release-mcf, 本次不倾斜")
            else:
                touched = side_lean_joint_indices(lean_side)
                lean = make_side_lean_target(
                    base, lean_side, args.thigh_delta, args.calf_delta
                )
                print(
                    f"[lowlean] touched joints: thigh={touched['thigh']}, calf={touched['calf']}, "
                    f"delta=({args.thigh_delta:+.3f}, {args.calf_delta:+.3f})"
                )
                print(f"[lowlean] {lean_side} lean target q:")
                print("  " + " ".join(f"{v:+.3f}" for v in lean))
                print(f"[lowlean] interpolate to {lean_side} lean in {args.lean_duration_sec:.1f}s")
                ctrl.interpolate_to(lean, float(args.lean_duration_sec))
                print(f"[lowlean] hold lean {args.lean_hold_sec:.1f}s")
                time.sleep(float(args.lean_hold_sec))
                print(f"[lowlean] restore in {args.restore_sec:.1f}s")
                ctrl.interpolate_to(base, float(args.restore_sec))
        else:
            print(f"[lowlean] hold current posture {args.hold_sec:.1f}s")
            time.sleep(float(args.hold_sec))
        print(f"[lowlean] done, sent LowCmd frames={ctrl.sent}")
        return 0
    except KeyboardInterrupt:
        print("\n[lowlean] Ctrl+C, restore current hold target and stop")
        return 130
    finally:
        ctrl.set_target(base)
        time.sleep(0.3)
        if released:
            print(f"[lowlean] restoring high-level mode '{args.restore_mode}' before stopping LowCmd...")
            try:
                code, raw = msc.SelectMode(args.restore_mode)
                print(f"[lowlean] SelectMode('{args.restore_mode}') return: code={code}, data={raw}")
                time.sleep(2.0)
                code2, data2 = msc.CheckMode()
                print(f"[lowlean] CheckMode after restore: code={code2}, data={data2}")
                restore_ok = (code == 0)
            except Exception as e:
                print(f"[lowlean] restore mode failed: {e}")

        if released and not restore_ok:
            print()
            print("[lowlean] WARNING: high-level restore failed. Keeping LowCmd hold alive.")
            print("[lowlean] Use Unitree App/reboot/support stand to recover before stopping this process.")
            print("[lowlean] Press Ctrl+C only when the robot is physically supported.")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass

        ctrl.close()
        print("[lowlean] LowCmd thread stopped")


if __name__ == "__main__":
    sys.exit(main())
