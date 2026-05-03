"""检测 + 切换 Go2 的 MotionSwitcher 模式.

背景:
   Go2 EDU 默认运行 MotionSwitcher 服务, 它管理"运动栈"模式:
     - "normal":   普通模式, SportClient.Move(vx,vy,vyaw) 在此生效
     - "ai":       AI 模式 (避障/跟随等), 屏蔽 Move
     - "advanced": 高级模式
   StandUp / BalanceStand 在所有模式都接受, 但 Move 只在 normal 接受.

用法:
   # 只看不切
   python3 -m scripts.test_motion_mode --network eth0

   # 切到 normal
   python3 -m scripts.test_motion_mode --network eth0 --select normal

   # 切到 ai (赛后还原)
   python3 -m scripts.test_motion_mode --network eth0 --select ai
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="eth0")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--select", default=None,
                        choices=[None, "normal", "ai", "advanced", "mcf", "ai-w"],
                        help="不指定 = 只读不切; 指定 = 切到该模式")
    parser.add_argument("--release", action="store_true",
                        help="先 ReleaseMode 把当前模式释放掉再切换 (从 mcf 切到 normal 必备)")
    parser.add_argument("--release-only", action="store_true",
                        help="只 ReleaseMode 后退出")
    args = parser.parse_args()

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    except ImportError as e:
        print(f"❌ unitree_sdk2py 未装: {e}", file=sys.stderr)
        return 2

    print(f"[mode] DDS init iface={args.network}")
    ChannelFactoryInitialize(args.domain, args.network)

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    print("[mode] MotionSwitcherClient 就绪")

    print("[mode] 查询当前模式...")
    code, data = msc.CheckMode()
    print(f"  CheckMode return: code={code}, data={data}")

    if code != 0:
        print(f"❌ CheckMode 失败 (code={code}). MotionSwitcher 服务可能没跑.")
        return 1

    current = (data or {}).get("name", "(unknown)")
    print(f"[mode] 当前模式 = '{current}'")

    if args.release or args.release_only:
        print(f"[mode] 先 ReleaseMode 释放 '{current}' ...")
        code, _ = msc.ReleaseMode()
        print(f"  ReleaseMode return: code={code} (0=成功)")
        time.sleep(1.5)
        code, data = msc.CheckMode()
        after_release = (data or {}).get("name", "(unknown)")
        print(f"[mode] 释放后模式 = '{after_release}'")
        if args.release_only:
            return 0
        current = after_release

    if args.select is None:
        print()
        print("[mode] 仅查询. 如果当前不是 'normal', 切换建议:")
        if current == "mcf":
            print("       (mcf 是 EDU 特有的融合模式, 必须先 release 再切, 加 --release):")
            print("       python3 -m scripts.test_motion_mode --network eth0 --release --select normal")
        else:
            print("       python3 -m scripts.test_motion_mode --network eth0 --select normal")
        return 0

    if current == args.select:
        print(f"[mode] 已经是 '{args.select}', 无需切换")
        return 0

    print(f"[mode] 切换 '{current}' -> '{args.select}'...")
    code, _ = msc.SelectMode(args.select)
    print(f"  SelectMode return: code={code} (0=成功)")
    if code != 0:
        print(f"❌ SelectMode 失败 (code={code})")
        if code == 7004:
            print("   提示: 7004 通常是 '当前已被锁定/必须先释放'.")
            print("   尝试: python3 -m scripts.test_motion_mode --network eth0 --release --select normal")
        return 1

    time.sleep(2.0)
    print("[mode] 验证切换结果...")
    code, data = msc.CheckMode()
    new_mode = (data or {}).get("name", "(unknown)")
    print(f"[mode] 现在是 '{new_mode}'")
    if new_mode != args.select:
        print(f"❌ 切换后实际是 '{new_mode}', 不是 '{args.select}'")
        return 1
    print(f"[mode] 切换成功 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
