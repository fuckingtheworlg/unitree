"""验证 Mac 端 OpenCV 弹窗 + 摄像头实时画面。

用法:
    python scripts/check_cv_window.py                 # 合成动画 (无相机)
    python scripts/check_cv_window.py --camera        # 默认设备 0, AVFoundation
    python scripts/check_cv_window.py --camera --device 1   # 试设备 1 (有时是 iPhone 连续互通)
    python scripts/check_cv_window.py --probe         # 只列出可用摄像头, 不开窗口
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np


def _nothing(_x):
    return


def probe_cameras(max_idx: int = 5) -> list[tuple[int, int, int]]:
    """枚举设备 0~max_idx, 用 AVFoundation 后端探测能否拿到画面。"""
    found: list[tuple[int, int, int]] = []
    for idx in range(max_idx + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
        for _ in range(8):
            cap.read()
            time.sleep(0.05)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None and frame.size > 0:
            found.append((idx, frame.shape[1], frame.shape[0]))
            print(f"  [device {idx}] OK, 分辨率 {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"  [device {idx}] 打开成功但拿不到画面 (黑帧)")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    print(f"OpenCV 版本: {cv2.__version__}")
    info = cv2.getBuildInformation()
    print(f"Cocoa 后端: {'YES' if 'Cocoa:                       YES' in info else '?'}")
    print(f"AVFoundation 后端: {'YES' if 'AVFOUNDATION:                YES' in info else '?'}")

    if args.probe:
        print("\n>>> 枚举可用摄像头 (AVFoundation):")
        found = probe_cameras(5)
        if not found:
            print("\n❌ 没找到任何能出画面的摄像头.")
            print("   - 系统设置 → 隐私与安全性 → 摄像头, 把当前运行 python 的 App 勾上")
            print("   - 然后**重启那个 App** (终端/Cursor) 再试")
        else:
            print(f"\n✓ 共 {len(found)} 个设备可用. 用 --camera --device <n> 指定其中一个")
        return

    cv2.namedWindow("Mac OpenCV 实时窗口测试", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mac OpenCV 实时窗口测试", 800, 600)
    cv2.createTrackbar("Hue", "Mac OpenCV 实时窗口测试", 60, 179, _nothing)

    cap = None
    if args.camera:
        print(f"\n>>> 打开设备 {args.device}, 强制 AVFoundation 后端...")
        cap = cv2.VideoCapture(args.device, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print(f"❌ 设备 {args.device} 打不开. 试 --probe 看看哪些索引可用.")
            cap = None
        else:
            print("    isOpened=True, 预热 10 帧...")
            for i in range(10):
                ok, _ = cap.read()
                time.sleep(0.05)

    t0 = time.time()
    black_frame_count = 0

    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                black_frame_count += 1
                if black_frame_count == 1:
                    print(f"⚠ read() 拿不到画面 (ok={ok}, frame={'None' if frame is None else frame.shape})")
                if black_frame_count >= 30:
                    print("\n❌ 连续 30 帧拿不到画面, 退到合成画面.")
                    print("可能原因 + 解决:")
                    print("  1) 跑 python 的 App 没有摄像头权限")
                    print("     → 系统设置 → 隐私与安全性 → 摄像头, 找 Terminal/Cursor 勾上")
                    print("     → 然后**完全退出并重开 App**, 重跑")
                    print("  2) 摄像头被别的 App 占用 (FaceTime/Zoom/Photo Booth)")
                    print("     → 关掉那些 App")
                    print("  3) 设备索引不对")
                    print("     → 跑 python scripts/check_cv_window.py --probe")
                    cap.release()
                    cap = None
                    continue
        if cap is None:
            t = time.time() - t0
            frame = np.zeros((600, 800, 3), np.uint8)
            for i in range(0, 800, 40):
                cv2.line(frame, (i, 0), (i + int(60 * np.sin(t)), 600), (40, 40, 40), 1)

        h = cv2.getTrackbarPos("Hue", "Mac OpenCV 实时窗口测试")
        hsv_strip = np.full((40, frame.shape[1], 3), [h, 220, 220], np.uint8)
        bgr_strip = cv2.cvtColor(hsv_strip, cv2.COLOR_HSV2BGR)
        frame[:40] = bgr_strip

        cv2.putText(
            frame,
            f"OpenCV {cv2.__version__} | Hue={h} | press q to quit",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
        )

        cv2.imshow("Mac OpenCV 实时窗口测试", frame)
        key = cv2.waitKey(15) & 0xFF
        if key in (ord("q"), 27):
            print("用户退出")
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("窗口测试结束")


if __name__ == "__main__":
    main()
