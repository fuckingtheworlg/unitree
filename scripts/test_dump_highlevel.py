"""Go2 高层卸料动作测试.

不释放 mcf, 不进入低层控制。只测试 SportClient 高层动作组合:
  - stretch: 官方 Stretch, 前低后高
  - sit_rise: Sit 后 RiseSit
  - scrape: 官方 Scrape 预置动作, 可观察是否有侧向倒料效果
  - cross_step: 官方 CrossStep 预置动作, 可观察是否有侧向倒料效果
  - pose_stretch: Pose(True) 后 Stretch, 再 Pose(False)

用法:
  .venv/bin/python -m scripts.test_dump_highlevel --network en5 \
    --method stretch --confirm I_UNDERSTAND_RISK
"""

from __future__ import annotations

import argparse
import sys
import time

from src.robot.go2_client import Go2Client
from src.robot.motions import execute_dump_action


CONFIRM_TEXT = "I_UNDERSTAND_RISK"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="eth0")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--method", default="stretch",
                        choices=[
                            "stretch", "sit_rise", "scrape",
                            "cross_step", "pose_stretch", "euler",
                        ])
    parser.add_argument("--hold-sec", type=float, default=2.5)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--roll", type=float, default=-0.30,
                        help="仅 method=euler 使用")
    parser.add_argument("--confirm", default="",
                        help=f"必须填 {CONFIRM_TEXT} 才会发运动指令")
    args = parser.parse_args()

    if args.confirm != CONFIRM_TEXT:
        print(f"拒绝执行: 请确认现场安全后加 --confirm {CONFIRM_TEXT}")
        return 2

    robot = Go2Client(
        network_iface=args.network,
        domain_id=int(args.domain),
        dry_run=False,
        logger=None,
    )
    try:
        robot.init()
        print("[dump_high] StopMove + BalanceStand")
        robot.stop_move()
        robot.balance_stand()
        time.sleep(1.0)
        print(
            f"[dump_high] method={args.method} hold={args.hold_sec}s "
            f"repeat={args.repeat}"
        )
        execute_dump_action(
            robot,
            method=args.method,
            roll_rad=float(args.roll),
            hold_sec=float(args.hold_sec),
            repeat=int(args.repeat),
        )
        print("[dump_high] done")
        return 0
    except KeyboardInterrupt:
        print("\n[dump_high] Ctrl+C")
        return 130
    finally:
        try:
            robot.stop_move()
            robot.balance_stand()
        except Exception:
            pass
        robot.shutdown()


if __name__ == "__main__":
    sys.exit(main())
