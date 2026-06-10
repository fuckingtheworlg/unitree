"""实验性: 不释放 mcf, 尝试 SetSilent(True) 后执行 Euler 倾倒.

背景:
  实测 ReleaseMode(mcf) 会让 Go2 直接失去高层支撑而瘫倒, 所以不要用 release
  来换取 Euler 控制权。本脚本只调用 MotionSwitcher.SetSilent(True), 再做小幅
  Euler roll 测试, 最后 SetSilent(False) 还原。

用法:
  .venv/bin/python -m scripts.test_euler_silent_dump --network en5 --confirm I_UNDERSTAND_RISK

如果这个脚本能让狗侧倾, 主程序的倾倒动作应该改成:
  SetSilent(True) -> Euler spam -> Euler reset -> SetSilent(False)
"""

from __future__ import annotations

import argparse
import json
import sys
import time


CONFIRM_TEXT = "I_UNDERSTAND_RISK"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="eth0")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--roll", type=float, default=-0.30,
                        help="roll rad, 负数通常向右倾; 先用 -0.30 保守测试")
    parser.add_argument("--hold-sec", type=float, default=3.0)
    parser.add_argument("--spam-hz", type=int, default=30)
    parser.add_argument("--confirm", default="",
                        help=f"必须填 {CONFIRM_TEXT} 才会发运动指令")
    args = parser.parse_args()

    if args.confirm != CONFIRM_TEXT:
        print(f"拒绝执行: 请确认现场安全后加 --confirm {CONFIRM_TEXT}")
        return 2

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_api import (
            MOTION_SWITCHER_API_ID_GET_SILENT,
            MOTION_SWITCHER_API_ID_SET_SILENT,
        )
        from unitree_sdk2py.go2.sport.sport_client import SportClient
    except ImportError as e:
        print(f"unitree_sdk2py 未装: {e}", file=sys.stderr)
        return 2

    print(f"[euler_silent] DDS init iface={args.network}")
    ChannelFactoryInitialize(args.domain, args.network)

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    sport = SportClient()
    sport.SetTimeout(10.0)
    sport.Init()

    code, data = msc.CheckMode()
    print(f"[euler_silent] CheckMode: code={code}, data={data}")
    if code != 0:
        return 1

    def set_silent(value: bool) -> int:
        code_, raw_ = msc._Call(
            MOTION_SWITCHER_API_ID_SET_SILENT,
            json.dumps({"data": bool(value)}),
        )
        print(f"[euler_silent] SetSilent({value}) -> code={code_}, data={raw_}")
        return int(code_)

    def get_silent() -> None:
        code_, raw_ = msc._Call(MOTION_SWITCHER_API_ID_GET_SILENT, json.dumps({}))
        print(f"[euler_silent] GetSilent -> code={code_}, data={raw_}")

    try:
        get_silent()
        if set_silent(True) != 0:
            print("[euler_silent] SetSilent 失败, 不执行 Euler")
            return 1
        time.sleep(0.5)
        get_silent()

        print("[euler_silent] StopMove")
        print("  ret=", sport.StopMove())
        time.sleep(0.3)
        print("[euler_silent] BalanceStand")
        print("  ret=", sport.BalanceStand())
        time.sleep(0.8)

        interval = 1.0 / max(1, int(args.spam_hz))
        end_at = time.time() + float(args.hold_sec)
        bad = []
        count = 0
        print(
            f"[euler_silent] Euler roll={args.roll:.3f} for "
            f"{args.hold_sec:.1f}s @ {args.spam_hz}Hz"
        )
        while time.time() < end_at:
            ret = sport.Euler(float(args.roll), 0.0, 0.0)
            if ret != 0 and len(bad) < 10:
                bad.append(ret)
            count += 1
            time.sleep(interval)
        print(f"[euler_silent] sent={count}, nonzero_sample={bad}")

        print("[euler_silent] reset Euler(0,0,0) for 1s")
        end_at = time.time() + 1.0
        while time.time() < end_at:
            sport.Euler(0.0, 0.0, 0.0)
            time.sleep(interval)
        print("[euler_silent] BalanceStand")
        print("  ret=", sport.BalanceStand())
        time.sleep(0.3)
        print("[euler_silent] StopMove")
        print("  ret=", sport.StopMove())
        return 0
    finally:
        try:
            set_silent(False)
        except Exception as e:
            print(f"[euler_silent] SetSilent(False) 失败: {e}")


if __name__ == "__main__":
    sys.exit(main())
