"""狗端: 静态拍 D435i RGB+Depth snap 用于校准.

用法:
  python3 scripts/snap_realsense_pose.py [--mjpeg-port 8089]

流程:
  1. 启动 D435i (后台读流)
  2. 启动 MJPEG 流 (默认 :8089) 让 Mac 浏览器看实时画面
  3. 终端提示输 label, 输 'quit' 退出
  4. 输 label 后程序连续拍 30 帧, 平均存为 .npz + 标注图 .jpg

合法 label (脚本写死, 想加新种类改这里):
  dump_approach, dump_center,
  stair_approach, stair_center,
  dock_approach, dock_center
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger  # noqa: E402
from src.utils.mjpeg_server import MjpegServer  # noqa: E402
from src.vision.realsense_camera import RealsenseCamera  # noqa: E402


_VALID_LABELS = {
    "dump_approach", "dump_center",
    "stair_approach", "stair_center",
    "dock_approach", "dock_center",
}


def _avg_frames(cam: RealsenseCamera, n: int = 30, log=None) -> dict:
    rgbs, depths = [], []
    fx = fy = ppx = ppy = depth_scale = 0.0
    while len(rgbs) < n:
        f = cam.read_latest(max_age_s=1.0)
        if f is None:
            time.sleep(0.05)
            continue
        rgbs.append(f.rgb)
        depths.append(f.depth_raw)
        fx, fy, ppx, ppy = f.fx, f.fy, f.ppx, f.ppy
        depth_scale = f.depth_scale
        time.sleep(0.03)
        if log and len(rgbs) % 10 == 0:
            log.info("  采样 %d/%d", len(rgbs), n)
    rgb_avg = np.mean(np.stack(rgbs).astype(np.float32), axis=0).astype(np.uint8)
    depth_med = np.median(np.stack(depths).astype(np.float32), axis=0).astype(np.uint16)
    return {
        "rgb": rgb_avg,
        "depth_raw": depth_med,
        "depth_scale": depth_scale,
        "fx": fx, "fy": fy, "ppx": ppx, "ppy": ppy,
        "avg_frames": n,
        "timestamp": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjpeg-port", type=int, default=8089)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out-root", default="logs/realsense/snaps")
    parser.add_argument("--avg-frames", type=int, default=30)
    args = parser.parse_args()

    log = get_logger("snap", level="INFO", save_dir="logs")

    log.info("启动 D435i ...")
    cam = RealsenseCamera(args.width, args.height, args.fps,
                          enable_depth=True, align_to_color=True, logger=log)
    if not cam.open():
        log.error("D435i 启动失败")
        return 2

    mjpeg = None
    if args.mjpeg_port > 0:
        mjpeg = MjpegServer(port=args.mjpeg_port, jpeg_quality=70)
        try:
            mjpeg.start()
            log.info("MJPEG live: http://<robot-ip>:%d/", args.mjpeg_port)
        except Exception as e:
            log.error("MJPEG start failed: %s", e)
            mjpeg = None

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("输出目录: %s", run_dir)

    import threading
    stop_evt = threading.Event()

    def _push_loop():
        while not stop_evt.is_set():
            f = cam.read_latest(max_age_s=0.5)
            if f is not None and mjpeg is not None:
                mjpeg.push_frame(f.rgb)
            time.sleep(0.05)
    pusher = threading.Thread(target=_push_loop, daemon=True)
    pusher.start()

    try:
        while True:
            print()
            print(f"输入 label (合法: {sorted(_VALID_LABELS)}) 或 'quit' 退出:")
            try:
                label = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not label:
                continue
            if label.lower() in ("q", "quit", "exit"):
                break
            if label not in _VALID_LABELS:
                print(f"[ignore] 不认识 '{label}', 合法 label: {', '.join(sorted(_VALID_LABELS))}")
                continue

            log.info("拍 %s (平均 %d 帧)...", label, args.avg_frames)
            data = _avg_frames(cam, n=args.avg_frames, log=log)

            npz_path = run_dir / f"{label}.npz"
            np.savez_compressed(str(npz_path), label=label, **data)
            log.info("保存 %s", npz_path)

            depth_m = data["depth_raw"].astype(np.float32) * data["depth_scale"]
            valid = depth_m > 0
            med = float(np.median(depth_m[valid])) if valid.any() else 0.0
            mn = float(depth_m[valid].min()) if valid.any() else 0.0
            mx = float(depth_m[valid].max()) if valid.any() else 0.0
            valid_ratio = float(np.count_nonzero(valid)) / valid.size

            dvis = depth_m.copy()
            lo, hi = (max(0.1, mn - 0.05), min(2.0, mx + 0.05))
            if hi - lo < 0.05:
                hi = lo + 0.05
            d_norm = np.clip((dvis - lo) / (hi - lo), 0, 1)
            d_u8 = (d_norm * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
            depth_vis[~valid] = 0

            cat = np.hstack([data["rgb"], depth_vis])
            cv2.putText(cat, f"depth all valid={valid_ratio:.2f} "
                              f"min={mn:.2f}m med={med:.2f}m max={mx:.2f}m",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(cat, f"label={label}", (10, cat.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            jpg_path = run_dir / f"{label}.jpg"
            cv2.imwrite(str(jpg_path), cat)
            log.info("保存 %s (depth med=%.2fm)", jpg_path, med)
    finally:
        stop_evt.set()
        cam.close()
        if mjpeg is not None:
            mjpeg.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
