"""D435i RealSense 后台线程版取流."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class RealsenseFrame:
    rgb: np.ndarray
    depth_raw: np.ndarray
    depth_scale: float
    fx: float
    fy: float
    ppx: float
    ppy: float
    timestamp: float


class RealsenseCamera:
    """后台线程持续读 (rgb, depth) 帧, 主线程随时 read_latest()."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        align_to_color: bool = True,
        logger=None,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.align_to_color = align_to_color
        self.log = logger

        self._pipeline = None
        self._align = None
        self._depth_scale: Optional[float] = None
        self._intrinsics: Optional[Tuple[float, float, float, float]] = None
        self._latest: Optional[RealsenseFrame] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._failures = 0

    def open(self) -> bool:
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            if self.log:
                self.log.error("无法 import pyrealsense2: %s", e)
            return False

        try:
            self._pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, self.width, self.height,
                              rs.format.bgr8, self.fps)
            if self.enable_depth:
                cfg.enable_stream(rs.stream.depth, self.width, self.height,
                                  rs.format.z16, self.fps)
            profile = self._pipeline.start(cfg)
            if self.enable_depth and self.align_to_color:
                self._align = rs.align(rs.stream.color)
            if self.enable_depth:
                self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self._intrinsics = (color_intr.fx, color_intr.fy,
                                color_intr.ppx, color_intr.ppy)
            if self.log:
                self.log.info(
                    "D435i opened: %dx%d@%dHz fx=%.1f ppx=%.1f depth_scale=%s",
                    self.width, self.height, self.fps,
                    color_intr.fx, color_intr.ppx, self._depth_scale,
                )
        except Exception as e:
            if self.log:
                self.log.error("D435i pipeline.start 失败: %s", e)
            try:
                if self._pipeline is not None:
                    self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            return False

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker, name="d435i-reader", daemon=True
        )
        self._thread.start()
        return True

    def _worker(self) -> None:
        if self._pipeline is None or self._intrinsics is None:
            return
        fx, fy, ppx, ppy = self._intrinsics
        depth_scale = self._depth_scale or 0.001

        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=2000)
            except Exception as e:
                self._failures += 1
                if self.log and self._failures % 10 == 1:
                    self.log.warning("D435i wait_for_frames 失败 #%d: %s",
                                     self._failures, e)
                time.sleep(0.05)
                continue
            if self._align is not None:
                frames = self._align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame() if self.enable_depth else None
            if not color:
                continue
            rgb = np.asanyarray(color.get_data())
            depth_raw = (
                np.asanyarray(depth.get_data())
                if (depth is not None and self.enable_depth)
                else np.zeros(rgb.shape[:2], dtype=np.uint16)
            )
            frame = RealsenseFrame(
                rgb=rgb, depth_raw=depth_raw,
                depth_scale=depth_scale,
                fx=fx, fy=fy, ppx=ppx, ppy=ppy,
                timestamp=time.monotonic(),
            )
            with self._lock:
                self._latest = frame
            self._failures = 0

    def read_latest(self, max_age_s: float = 0.5) -> Optional[RealsenseFrame]:
        with self._lock:
            f = self._latest
        if f is None:
            return None
        if time.monotonic() - f.timestamp > max_age_s:
            return None
        return f

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            if self._pipeline is not None:
                self._pipeline.stop()
        except Exception:
            pass
        self._pipeline = None
        self._align = None
        if self.log:
            self.log.info("D435i closed")
