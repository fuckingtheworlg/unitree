"""鱼眼去畸变 + 透视(俯视)变换 IPM。

Go2 前置鱼眼有视场畸变。如果你已标定相机，把 K/D 填到 calibrate() 里；
没标定也能用 — IPM 只用图像四点变到俯视图，足以做巡线。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class IPMConfig:
    src_tl: Tuple[float, float]
    src_tr: Tuple[float, float]
    src_br: Tuple[float, float]
    src_bl: Tuple[float, float]
    dst_size: Tuple[int, int]


class IPMTransformer:
    """把图像底部一段梯形区域 warp 成 dst_size 的俯视图。"""

    def __init__(self, ipm_cfg: IPMConfig, image_size: Optional[Tuple[int, int]] = None):
        self.cfg = ipm_cfg
        self._M: Optional[np.ndarray] = None
        self._image_size = image_size
        if image_size is not None:
            self._build(image_size)

    def _build(self, image_size: Tuple[int, int]) -> None:
        w, h = image_size
        cfg = self.cfg
        src = np.float32([
            [cfg.src_tl[0] * w, cfg.src_tl[1] * h],
            [cfg.src_tr[0] * w, cfg.src_tr[1] * h],
            [cfg.src_br[0] * w, cfg.src_br[1] * h],
            [cfg.src_bl[0] * w, cfg.src_bl[1] * h],
        ])
        dw, dh = cfg.dst_size
        dst = np.float32([[0, 0], [dw, 0], [dw, dh], [0, dh]])
        self._M = cv2.getPerspectiveTransform(src, dst)
        self._image_size = image_size

    def warp(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if self._M is None or self._image_size != (w, h):
            self._build((w, h))
        return cv2.warpPerspective(
            image, self._M, self.cfg.dst_size, flags=cv2.INTER_LINEAR
        )


class FisheyeUndistorter:
    """可选项：如有相机标定结果，把鱼眼校正回 pinhole 图像。

    没有标定时直接 passthrough。
    """

    def __init__(
        self,
        K: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        balance: float = 0.0,
    ):
        self.K = K
        self.D = D
        self.balance = balance
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None
        self._size: Optional[Tuple[int, int]] = None

    @property
    def enabled(self) -> bool:
        return self.K is not None and self.D is not None

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image
        h, w = image.shape[:2]
        if self._map1 is None or self._size != (w, h):
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D, (w, h), np.eye(3), balance=self.balance
            )
            self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
            self._size = (w, h)
        return cv2.remap(image, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)
