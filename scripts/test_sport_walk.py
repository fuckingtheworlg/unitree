"""SportClient 慢速直行测试 - 让狗以 0.1 m/s 走 2 秒 (~20cm).

使用前提:
   - 已经成功跑过 test_sport_basic.py, 狗能站起来
   - 狗站在赛道起点, **正前方至少 1 米空旷**
   - 你准备好随时按 Ctrl+C 或断网

行为:
   1. BalanceStand
   2. Move(vx=0.1, vy=0, vyaw=0) 重发 2 秒 (期间狗低速直行)
   3. StopMove
   4. 打印理论位移, 让你拿尺量验证

成功标准:
   狗朝**机身正前方**走出大约 20cm, 没有往侧面歪.
   实际距离会因步态略小于理论 (~15-25cm 都正常).
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="eth0")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--vx", type=float, default=0.1, help="前进速度 m/s")
    parser.add_argument("--duration", type=float, default=2.0, help="移动时长 s")
    parser.add_argument("--hz", type=int, default=30, help="Move 命令重发频率")
    args = parser.parse_args()

    if args.vx > 0.3:
        print(f"❌ vx={args.vx} 超过 0.3 m/s, 拒绝执行 (首次测试要慢)", file=sys.stderr)
        return 2
    if args.duration > 5.0:
        print(f"❌ duration={args.duration} 超过 5s, 拒绝执行", file=sys.stderr)
        return 2

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient
    except ImportError as e:
        print(f"❌ unitree_sdk2py 未装: {e}", file=sys.stderr)
        return 2

    print(f"[test_walk] DDS init iface={args.network}")
    ChannelFactoryInitialize(args.domain, args.network)

    sport = SportClient()
    sport.SetTimeout(10.0)
    sport.Init()
    print("[test_walk] SportClient 就绪")

    print(f"[test_walk] 计划: 以 vx={args.vx} m/s 直行 {args.duration:.1f} 秒")
    print(f"           理论位移 = {args.vx * args.duration * 100:.1f} cm")
    print(f"[test_walk] 倒计时 5 秒, 准备好按 Ctrl+C...")
    for i in (5, 4, 3, 2, 1):
        print(f"  {i}...")
        time.sleep(1)

    print("[test_walk] BalanceStand")
    sport.BalanceStand()
    time.sleep(1.0)

    print(f"[test_walk] 开始直行...")
    interval = 1.0 / args.hz
    end_at = time.time() + args.duration
    cmd_count = 0
    try:
        while time.time() < end_at:
            sport.Move(float(args.vx), 0.0, 0.0)
            cmd_count += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[test_walk] Ctrl+C, 立刻停下")
    finally:
        sport.StopMove()
        time.sleep(0.3)
        sport.BalanceStand()

    print(f"[test_walk] 已停下, 共发了 {cmd_count} 条 Move 命令")
    print("[test_walk] 完成 ✓ 现在拿尺量狗实际走了多远:")
    print(f"           - 距离 ≈ 15~25cm: 正常 (实际略小于理论 {args.vx * args.duration * 100:.0f}cm)")
    print(f"           - 完全没动: SportClient 通信异常, 检查日志")
    print(f"           - 走得偏左/右 > 5cm: 视觉巡线时 PID 需要补偿")
    print(f"           - 走得过远 (> 30cm): vx 实际更大, 后面要降 forward_speed.follow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
