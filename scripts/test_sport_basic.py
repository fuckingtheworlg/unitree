"""SportClient 最小通信测试 - 让狗站起来 (不移动).

使用前提:
   - 狗放在地面上, 周围空旷
   - 你站在狗旁边, 准备好随时按 Ctrl+C

行为:
   1. ChannelFactoryInitialize
   2. SportClient.StandUp + BalanceStand
   3. 保持 3 秒让你观察狗是否站稳
   4. StopMove (但不 Damp / 不 StandDown, 让狗保持站立结束)

如果这一步成功且狗站得稳, 才能进入下一步 test_sport_walk.py
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="eth0")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--hold-sec", type=float, default=3.0,
                        help="站起来后保持的秒数")
    args = parser.parse_args()

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient
    except ImportError as e:
        print(f"❌ unitree_sdk2py 未装: {e}", file=sys.stderr)
        return 2

    print(f"[test_basic] DDS init iface={args.network}")
    ChannelFactoryInitialize(args.domain, args.network)

    sport = SportClient()
    sport.SetTimeout(10.0)
    sport.Init()
    print("[test_basic] SportClient 就绪")

    print("[test_basic] 倒计时 3 秒, 准备让狗 StandUp + BalanceStand...")
    for i in (3, 2, 1):
        print(f"  {i}...")
        time.sleep(1)

    print("[test_basic] 发 StandUp")
    code = sport.StandUp()
    print(f"  return code={code} (0=成功)")
    time.sleep(1.5)

    print("[test_basic] 发 BalanceStand")
    code = sport.BalanceStand()
    print(f"  return code={code} (0=成功)")

    print(f"[test_basic] 保持站立 {args.hold_sec:.1f}s, 观察狗是否站稳...")
    time.sleep(args.hold_sec)

    print("[test_basic] 发 StopMove (狗保持站立, 不会 Damp/趴下)")
    sport.StopMove()

    print("[test_basic] 完成 ✓ 如果狗站起来且稳定, 可以进行下一步 test_sport_walk.py")
    print("[test_basic] 如果想让狗趴下, 在新终端跑: ")
    print("            python3 -c 'from unitree_sdk2py.core.channel import ChannelFactoryInitialize; ChannelFactoryInitialize(0, \"eth0\"); from unitree_sdk2py.go2.sport.sport_client import SportClient; s=SportClient(); s.Init(); s.StandDown()'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
