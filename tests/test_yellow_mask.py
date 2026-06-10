import numpy as np

from src.vision.yellow_mask import crop_roi, yellow_mask


def _make_yellow_image(h=200, w=200) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = 0
    img[:, :, 1] = 215
    img[:, :, 2] = 255
    return img


def test_yellow_mask_full_yellow():
    img = _make_yellow_image()
    mask = yellow_mask(img, (18, 90, 90), (38, 255, 255))
    assert mask.shape == img.shape[:2]
    assert (mask == 255).mean() > 0.9


def test_yellow_mask_no_yellow():
    img = np.full((100, 100, 3), 50, np.uint8)
    mask = yellow_mask(img, (18, 90, 90), (38, 255, 255))
    assert (mask == 0).all()


def test_yellow_mask_adaptive_recovers_low_saturation_yellow():
    img = np.full((160, 220, 3), 230, np.uint8)
    # 模拟白蓝地标拉曝光后偏白、低饱和的黄线: 固定 S>=80 HSV 很容易漏掉.
    img[:, :90] = (255, 120, 40)      # 蓝白混合区里的蓝色干扰
    img[60:105, 110:170] = (170, 215, 225)

    strict_only = yellow_mask(
        img, (18, 80, 120), (38, 255, 255),
        adaptive=False,
        open_kernel=1,
        close_kernel=1,
    )
    adaptive = yellow_mask(
        img, (18, 80, 120), (38, 255, 255),
        adaptive=True,
        open_kernel=1,
        close_kernel=1,
    )

    yellow_roi = np.s_[60:105, 110:170]
    blue_roi = np.s_[:, :90]
    assert (strict_only[yellow_roi] == 255).mean() < 0.2
    assert (adaptive[yellow_roi] == 255).mean() > 0.8
    assert (adaptive[blue_roi] == 255).mean() < 0.05


def test_crop_roi_shape():
    img = np.zeros((100, 200, 3), np.uint8)
    roi, y0 = crop_roi(img, 0.5, 1.0)
    assert roi.shape == (50, 200, 3)
    assert y0 == 50
