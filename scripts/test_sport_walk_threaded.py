"""模仿用户给的参考程序: 用独立线程 20Hz 持续 spam Move, 验证 mcf 模式下能否驱动狗.

跟 test_sport_walk.py 的关键区别:
   - **不切 MotionSwitcher 模式** (保持 mcf)
   - **不调 BalanceStand** (参考程序里也没调, 直接 SportClient.Init 就 Move)
   - 用独立 daemon 线程 20Hz 重发 Move (跟参考程序一致)

用法:
   python3 -m scripts.test_sport_walk_threaded --network eth0

成功 = 狗朝正前方走 ~20cm 后停下
失败 = 狗 60 帧 Move 后还是不动 -> 那 mcf 真的拦截 Move
"""

from __future__ import annotations

import argparse
import sys
import threading
import time


_TARGET = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}
_LOCK = threading.Lock()
_KEEP_RUNNING = True


def _move_loop(sport_client, hz: int) -> None:
    interval = 1.0 / max(hz, 1)
    sent = 0
    while _KEEP_RUNNING:
        with _LOCK:
            vx = _TARGET["vx"]
            vy = _TARGET["vy"]
            vyaw = _TARGET["vyaw"]
        try:
            sport_client.Move(float(vx), float(vy), float(vyaw))
            sent += 1
        except Exception as e:
            print(f"[move-thread] Move 异常: {e}")
            break
        time.sleep(interval)
    print(f"[move-thread] 线程退出, 共发 {sent} 条 Move")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="eth0")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--vx", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--hz", type=int, default=20)
    parser.add_argument("--with-balance", action="store_true",
                        help="测试时先发 BalanceStand (参考程序不调, 默认也不调)")
    args = parser.parse_args()

    if args.vx > 0.3:
        print(f"❌ vx={args.vx} 超过 0.3 m/s, 拒绝执行", file=sys.stderr)
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

    print(f"[walk_t] DDS init iface={args.network}")
    ChannelFactoryInitialize(args.domain, args.network)

    sport = SportClient()
    sport.SetTimeout(5.0)
    sport.Init()
    print("[walk_t] SportClient 就绪 (未 SelectMode, 未 BalanceStand)")

    if args.with_balance:
        print("[walk_t] 先发 BalanceStand")
        sport.BalanceStand()
        time.sleep(1.0)

    print(f"[walk_t] 启动 Move 线程 ({args.hz}Hz spam)")
    move_thread = threading.Thread(
        target=_move_loop, args=(sport, args.hz), daemon=True
    )
    move_thread.start()

    print(f"[walk_t] 计划: 以 vx={args.vx} m/s 直行 {args.duration:.1f} 秒")
    print(f"          理论位移 = {args.vx * args.duration * 100:.1f} cm")
    print(f"[walk_t] 倒计时 5 秒, 准备好按 Ctrl+C...")
    for i in (5, 4, 3, 2, 1):
        print(f"  {i}...")
        time.sleep(1)

    print(f"[walk_t] 设目标速度 vx={args.vx}")
    with _LOCK:
        _TARGET["vx"] = float(args.vx)

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\n[walk_t] Ctrl+C, 立刻设速度 0")

    print("[walk_t] 设目标速度 0, 0, 0 (停下)")
    with _LOCK:
        _TARGET["vx"] = 0.0
        _TARGET["vy"] = 0.0
        _TARGET["vyaw"] = 0.0

    time.sleep(0.5)

    global _KEEP_RUNNING
    _KEEP_RUNNING = False
    move_thread.join(timeout=2.0)

    print()
    print("[walk_t] 完成 ✓ 现在拿尺量狗实际走了多远:")
    print(f"          - 距离 ≈ 15~25cm: 正常, **mcf 模式下用线程 spam 是 work 的**")
    print(f"          - 完全没动: mcf 真的拦截 Move, 必须先 ReleaseMode + SelectMode")
    print(f"          - 走了 但又自动停下: SDK 内部超时, 检查 spam 频率")
    return 0


if __name__ == "__main__":
    sys.exit(main())
