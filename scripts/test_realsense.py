"""D435i 驱动 + 取流诊断脚本（macOS / Linux 通用）

逐级降级测试：
  1) 枚举设备：能否找到 D435i
  2) 仅彩色流 (640x480 @ 30fps)：USB 链路是否正常
  3) 彩色 + 深度流：完整功能
按 q 关闭窗口；脚本会一直显示视频流直到关闭。

用法：
    python scripts/test_realsense.py              # 自动跑完三级测试
    python scripts/test_realsense.py --enum-only  # 只枚举不开流
    python scripts/test_realsense.py --color-only # 只开彩色，跳过 depth
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as e:
    print("[ERR] 无法 import pyrealsense2：", e)
    print("Mac venv 安装：pip install pyrealsense2-macosx")
    print("Linux venv 安装：pip install pyrealsense2")
    sys.exit(1)


def step1_enumerate() -> list:
    print("\n=== [1/3] 枚举 RealSense 设备 ===")
    ctx = rs.context()
    devices = list(ctx.query_devices())
    print(f"找到 {len(devices)} 台设备")
    info_list = []
    for i, d in enumerate(devices):
        try:
            name = d.get_info(rs.camera_info.name)
            serial = d.get_info(rs.camera_info.serial_number)
            fw = d.get_info(rs.camera_info.firmware_version)
            usb = d.get_info(rs.camera_info.usb_type_descriptor)
            print(f"  [{i}] {name}  SN={serial}  FW={fw}  USB={usb}")
            info_list.append({"name": name, "serial": serial, "usb": usb})
        except Exception as e:
            print(f"  [{i}] 取信息失败: {e}")
            print("        => 通常是 USB 链路问题：换 USB 口直连，不要 hub/转接头")
            print("        => D435i 必须 USB 3.0 直连；MBA M4 的 USB-C 口都支持")
            info_list.append(None)
    return info_list


def step2_color_only(width: int = 640, height: int = 480, fps: int = 30) -> bool:
    print(f"\n=== [2/3] 仅彩色流 {width}x{height} @ {fps}fps ===")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"[FAIL] 启动彩色流失败: {e}")
        return False
    print("启动成功，按 q 关闭窗口")
    try:
        t0 = time.time()
        n = 0
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            n += 1
            fps_now = n / max(time.time() - t0, 1e-6)
            cv2.putText(img, f"COLOR {width}x{height} {fps_now:.1f}fps  q=quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("D435i COLOR", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return True


def step3_color_depth(width: int = 640, height: int = 480, fps: int = 30) -> bool:
    print(f"\n=== [3/3] 彩色 + 深度 {width}x{height} @ {fps}fps ===")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    align = rs.align(rs.stream.color)
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"[FAIL] 启动彩色+深度失败: {e}")
        print("提示：D435i 同时开两路流需要 USB 3.0；如失败请检查 USB 链路")
        return False

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"depth_scale = {depth_scale}  (depth_unit_m = depth_pixel * {depth_scale})")
    print("启动成功，按 q 关闭窗口；窗口左上角显示画面中心点的真实距离 (m)")

    try:
        t0 = time.time()
        n = 0
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue
            color_img = np.asanyarray(color.get_data())
            depth_raw = np.asanyarray(depth.get_data())
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET
            )
            cy, cx = depth_raw.shape[0] // 2, depth_raw.shape[1] // 2
            d_m = float(depth_raw[cy, cx]) * depth_scale
            n += 1
            fps_now = n / max(time.time() - t0, 1e-6)

            cv2.drawMarker(color_img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.drawMarker(depth_vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(color_img, f"center={d_m:.3f}m  {fps_now:.1f}fps",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(depth_vis, "DEPTH",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            stacked = np.hstack((color_img, depth_vis))
            cv2.imshow("D435i COLOR + DEPTH", stacked)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enum-only", action="store_true")
    parser.add_argument("--color-only", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    info = step1_enumerate()
    if not info:
        print("\n[STOP] 没找到设备：检查 USB 是否插好（D435i 需 USB 3.0 直连）")
        return 1

    if args.enum_only:
        return 0

    ok = step2_color_only(args.width, args.height, args.fps)
    if not ok or args.color_only:
        return 0 if ok else 2

    step3_color_depth(args.width, args.height, args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
