"""相机数据源：实机 (Go2 VideoClient) + 离线 (本地视频文件)。

官方 VideoClient.GetImageSample() 返回 JPEG 字节流 (单帧, 非视频流),
参考 unitree_sdk2/example/go2/go2_video_client.cpp:
    std::vector<uint8_t> image_sample;
    video_client.GetImageSample(image_sample);
    // image_sample 直接写成 .jpg 文件即可

Python 解码:
    cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

已知限制 (issue #116): 在高频循环里 GetImageSample 会偶发返回错误码 3104,
    这里做了静默重试 + 跳帧容忍。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

import cv2
import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        ...

    def release(self) -> None:
        return None

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame


class VideoFileSource(FrameSource):
    def __init__(self, path: str, loop: bool = False):
        self.path = path
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {path}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok:
                    return None
            else:
                return None
        return frame

    def release(self) -> None:
        self.cap.release()


class Go2CameraSource(FrameSource):
    """从 Go2 VideoClient 拉一张 JPEG 帧, 解码成 BGR ndarray。

    单次 read() 内会重试若干次, 一直拿不到才返回 None。
    """

    def __init__(self, client, max_retry_per_read: int = 3, logger=None):
        self.client = client
        self.max_retry = max(1, max_retry_per_read)
        self.log = logger
        self._consecutive_failures = 0

    def read(self) -> Optional[np.ndarray]:
        for attempt in range(self.max_retry):
            code, data = self.client.get_image_sample()
            if code == 0 and data:
                self._consecutive_failures = 0
                buf = np.frombuffer(bytes(data), dtype=np.uint8)
                if buf.size == 0:
                    continue
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
                if self.log:
                    self.log.debug("imdecode 失败 (attempt=%d, bytes=%d)", attempt, buf.size)
            else:
                if self.log:
                    self.log.debug("GetImageSample code=%s (attempt=%d)", code, attempt)
        self._consecutive_failures += 1
        if self._consecutive_failures >= 30 and self.log:
            self.log.warning("连续 %d 次取帧失败, 检查相机服务", self._consecutive_failures)
        return None
