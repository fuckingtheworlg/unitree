"""实验性: 用 MotionSwitcher.SetSilent(True) 让 mcf 让出 Move 控制权.

背景:
   MotionSwitcher 服务有 5 个 API, SDK Python 只暴露了 3 个:
     - CheckMode (1001)   ✓ 已暴露
     - SelectMode (1002)  ✓ 已暴露
     - ReleaseMode (1003) ✓ 已暴露
     - SetSilent (1004)   ✗ 未暴露 → 本脚本直接走底层 _Call
     - GetSilent (1005)   ✗ 未暴露 → 同上

假设 (实验性, 不保证):
   SetSilent(True) = mcf 进入"静默": 保持电机使能, 但不抢 Move 控制权.

行为:
   1. 查当前模式 (应为 mcf)
   2. SetSilent(True)
   3. (可选) 立刻跑一次 SportClient.Move 看狗动不动

如果狗在静默模式下能 Move:
   赛前每次启动加一行 SetSilent(True), 巡线就能正常跑.
如果还是不动:
   只能走"用官方 App 切模式"路径.

赛后还原:
   python3 -m scripts.test_motion_silent --network eth0 --silent off
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time


_TARGET = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}
_LOCK = threading.Lock()
_KEEP_RUNNING = True


def _move_loop(sport_client, hz: int) -> None:
    interval = 1.0 / max(hz, 1)
    while _KEEP_RUNNING:
        with _LOCK:
            vx = _TARGET["vx"]; vy = _TARGET["vy"]; vyaw = _TARGET["vyaw"]
        try:
            sport_client.Move(float(vx), float(vy), float(vyaw))
        except Exception:
            break
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="eth0")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--silent", choices=["on", "off"], default="on",
                        help="on=mcf 进入静默 (让出 Move); off=退出静默 (恢复)")
    parser.add_argument("--with-walk-test", action="store_true",
                        help="设置 silent 后, 立刻跑 0.1m/s × 2s 直行测试")
    args = parser.parse_args()

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_api import (
            MOTION_SWITCHER_API_ID_SET_SILENT,
            MOTION_SWITCHER_API_ID_GET_SILENT,
        )
    except ImportError as e:
        print(f"❌ unitree_sdk2py 未装: {e}", file=sys.stderr)
        return 2

    print(f"[silent] DDS init iface={args.network}")
    ChannelFactoryInitialize(args.domain, args.network)

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    print("[silent] 查当前模式...")
    code, data = msc.CheckMode()
    print(f"  CheckMode return: code={code}, data={data}")

    print(f"[silent] 查当前 silent 状态...")
    code, raw = msc._Call(MOTION_SWITCHER_API_ID_GET_SILENT, json.dumps({}))
    if code == 0:
        try:
            d = json.loads(raw or "{}")
            print(f"  GetSilent: {d}")
        except Exception as e:
            print(f"  GetSilent raw={raw}, parse fail: {e}")
    else:
        print(f"  GetSilent failed code={code}")

    silent_value = (args.silent == "on")
    print(f"[silent] SetSilent({silent_value}) ...")
    code, raw = msc._Call(MOTION_SWITCHER_API_ID_SET_SILENT,
                          json.dumps({"data": silent_value}))
    print(f"  SetSilent return: code={code}, data={raw}")
    if code != 0:
        print(f"❌ SetSilent 失败 (code={code}). 这条路走不通, 试官方 App.")
        return 1

    time.sleep(1.0)
    print("[silent] 验证 silent 状态...")
    code, raw = msc._Call(MOTION_SWITCHER_API_ID_GET_SILENT, json.dumps({}))
    print(f"  GetSilent after: code={code}, data={raw}")

    if not args.with_walk_test:
        print()
        print(f"[silent] 设置完成 ✓ (silent={silent_value})")
        print("[silent] 想立刻测试狗能不能走, 重跑加 --with-walk-test")
        return 0

    if args.silent == "off":
        print("[silent] 已关 silent, 不做 walk 测试")
        return 0

    print()
    print("[silent] 现在测试 SportClient.Move 是否生效 (vx=0.1 m/s × 2s)")
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    sport = SportClient()
    sport.SetTimeout(5.0)
    sport.Init()
    print("[silent] BalanceStand")
    sport.BalanceStand()
    time.sleep(1.5)

    print("[silent] 启动 20Hz Move 线程")
    move_thread = threading.Thread(
        target=_move_loop, args=(sport, 20), daemon=True
    )
    move_thread.start()

    print("[silent] 倒计时 5 秒, 准备好按 Ctrl+C...")
    for i in (5, 4, 3, 2, 1):
        print(f"  {i}...")
        time.sleep(1)

    print("[silent] 设 vx=0.1, 持续 2 秒")
    with _LOCK:
        _TARGET["vx"] = 0.1
    try:
        time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n[silent] Ctrl+C")

    print("[silent] 设 vx=0 停下")
    with _LOCK:
        _TARGET["vx"] = 0.0; _TARGET["vy"] = 0.0; _TARGET["vyaw"] = 0.0
    time.sleep(0.5)

    global _KEEP_RUNNING
    _KEEP_RUNNING = False
    move_thread.join(timeout=2.0)

    print()
    print("[silent] 测试结束. 拿尺量狗实际走了多远:")
    print("          - 走了 ~20cm: 🎉 silent 模式有效, 这是巡线主程序前的标准启动步骤")
    print("          - 还是不动: silent 也不解决, 必须走官方 App 切模式")
    return 0


if __name__ == "__main__":
    sys.exit(main())
