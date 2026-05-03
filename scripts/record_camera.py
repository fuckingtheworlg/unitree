"""录制 Go2 前置相机视频，供赛前在 Mac 上离线调参用。

⚠ 安全说明:
   本脚本**只**初始化 VideoClient, 完全不碰 SportClient,
   所以录像期间狗不会有任何动作 (即使你在狗趴着 / 被抱着 / 在桌面上时跑).

⚠ 容器格式说明:
   默认输出 .avi (MJPG 编码), 因为 .mp4 中途 Ctrl+C 会因为 moov atom
   没写而损坏成 "moov atom not found" 不可读. .avi 任何时刻中断都能播放.

用法 (在狗上 ssh 跑):
    cd ~/go2-patrol
    python3 -m scripts.record_camera --network eth0 --duration 60 \\
        --output recordings/lap01.avi

录完拷回 Mac:
    scp unitree@192.168.123.18:~/go2-patrol/recordings/lap01.avi ./tests/data/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="eth0", help="DDS 网卡名")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output", default="recordings/clip.avi",
                        help="输出文件; 推荐 .avi (中断也能播); .mp4 必须录完不能 Ctrl+C")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--display", action="store_true", help="弹 cv2 窗口 (默认关; 狗端没 GUI)")
    args = parser.parse_args()

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.video.video_client import VideoClient
    except ImportError as e:
        raise SystemExit(f"未找到 unitree_sdk2py: {e}")

    print(f"[record] init DDS on iface={args.network} domain={args.domain}")
    ChannelFactoryInitialize(args.domain, args.network)

    video = VideoClient()
    video.SetTimeout(3.0)
    video.Init()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frame_count = 0
    fail_count = 0
    start = time.time()

    print(f"[record] recording -> {args.output} (max {args.duration:.0f}s)")
    print(f"[record] press Ctrl+C to stop early")

    try:
        while time.time() - start < args.duration:
            code, data = video.GetImageSample()
            if code != 0 or not data:
                fail_count += 1
                if fail_count % 30 == 0:
                    print(f"[record] {fail_count} 帧失败 (code={code}), 继续...")
                continue
            buf = np.frombuffer(bytes(data), dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                fail_count += 1
                continue

            if writer is None:
                h, w = frame.shape[:2]
                ext = Path(args.output).suffix.lower()
                if ext == ".avi":
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                elif ext == ".mp4":
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    print("[record] ⚠ 用 .mp4 容器, Ctrl+C 中断会让文件损坏不可播; 强烈建议改成 .avi")
                else:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
                print(f"[record] 第一帧到达, 分辨率 {w}x{h}, 容器=.{ext.lstrip('.')}, 开始写文件")
            writer.write(frame)
            frame_count += 1

            if args.display:
                cv2.imshow("recording", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if frame_count % 30 == 0:
                elapsed = time.time() - start
                print(f"[record] {frame_count} 帧 / {elapsed:.1f}s "
                      f"(fps={frame_count/max(elapsed,0.1):.1f})")
    except KeyboardInterrupt:
        print("\n[record] Ctrl+C, 收尾保存...")
    finally:
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
        elapsed = time.time() - start
        print(f"[record] 完成: {frame_count} 帧 / {elapsed:.1f}s "
              f"(fps={frame_count/max(elapsed,0.1):.1f}), {fail_count} 帧失败")
        print(f"[record] 输出: {args.output}")


if __name__ == "__main__":
    main()
