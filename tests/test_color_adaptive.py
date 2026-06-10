import numpy as np

from src.vision.realsense_target import ColorLabel, classify_color_combo


def _dim_black_white(w=200, h=200):
    """偏暗的"白底+黑字"倾倒区: 白只到 V~120 (灯光不足)."""
    rgb = np.full((h, w, 3), 120, np.uint8)  # 偏暗的白 (灰白)
    rgb[:, : w // 3] = 25                     # 黑字区
    depth = np.full((h, w), 230, np.uint16)   # 0.23m
    return rgb, depth


def test_adaptive_recovers_dim_white():
    rgb, depth = _dim_black_white()
    strict = classify_color_combo(
        rgb, depth_raw=depth, depth_scale=0.001, adaptive_enabled=False,
    )
    adaptive = classify_color_combo(
        rgb, depth_raw=depth, depth_scale=0.001, adaptive_enabled=True,
    )
    # 暗白 (V~120) 在固定档可能偏低, 自适应后白占比应更高
    assert adaptive.white_ratio >= strict.white_ratio


def test_bright_white_unchanged():
    rgb = np.full((200, 200, 3), 230, np.uint8)
    rgb[:, :60] = 20
    depth = np.full((200, 200), 230, np.uint16)
    a = classify_color_combo(rgb, depth_raw=depth, depth_scale=0.001, adaptive_enabled=True)
    b = classify_color_combo(rgb, depth_raw=depth, depth_scale=0.001, adaptive_enabled=False)
    # 亮场封顶=base, 两者白占比应一致
    assert abs(a.white_ratio - b.white_ratio) < 1e-6
