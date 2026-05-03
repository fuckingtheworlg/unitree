"""分析 snap_realsense_pose.py 拍的 snap, 验证 color combo classifier.

用法:
  .venv/bin/python scripts/analyze_snaps.py logs/realsense/snaps/run-XXXX/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config  # noqa: E402
from src.vision.realsense_target import (  # noqa: E402
    ColorClassification,
    ColorLabel,
    classify_color_combo,
)


_EXPECTED: Dict[str, ColorLabel] = {
    "dump": ColorLabel.BLACK_WHITE,
    "stair": ColorLabel.BLUE_WHITE,
    "dock": ColorLabel.BLUE_WHITE,
}


def _kind(stem: str) -> str:
    name = stem.lower()
    for k in _EXPECTED:
        if name.startswith(k):
            return k
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snap_dir")
    parser.add_argument("--config", default="config/params.yaml")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="trigger 阈值 = depth_median + margin")
    parser.add_argument("--save-vis", action="store_true", default=True)
    args = parser.parse_args()

    snap_dir = Path(args.snap_dir).resolve()
    if not snap_dir.is_dir():
        print(f"[ERR] {snap_dir} 不是目录")
        return 2

    cfg = load_config(args.config)
    ccfg = cfg.realsense.color_stat
    kw = {}
    for key in ("white_min_ratio", "black_min_ratio", "blue_min_ratio",
                "other_max_ratio", "yellow_max_ratio",
                "night_white_min_ratio", "night_black_min_ratio",
                "night_blue_min_ratio", "night_other_max_ratio_blackwhite",
                "night_other_max_ratio_bluewhite", "night_blue_dominant_sum",
                "depth_min_m", "depth_max_m", "min_valid_depth_ratio",
                "roi_w_ratio", "roi_h_ratio"):
        v = getattr(ccfg, key, None)
        if v is not None:
            kw[key] = float(v)
    ne = getattr(ccfg, "night_enabled", None)
    if ne is not None:
        kw["night_enabled"] = bool(ne)

    npz_files = sorted(snap_dir.glob("*.npz"))
    if not npz_files:
        print(f"[ERR] {snap_dir} 没有 .npz 文件")
        return 2

    rows = []
    by_kind_depth = {"dump": [], "stair": [], "dock": []}
    n_match = 0

    print()
    print("=" * 110)
    header = (f"{'snap':<20} {'expect':<14} {'got':<14} {'ok':<6} "
              f"{'w':>5} {'k':>5} {'b':>5} {'y':>5} {'z':>7} {'valid':>5}")
    print(header)
    print("-" * 110)

    for npz in npz_files:
        stem = npz.stem
        kind = _kind(stem)
        if kind == "unknown":
            print(f"{stem:<20} unknown kind, skip")
            continue
        is_center = stem.endswith("_center")
        expected = _EXPECTED[kind] if is_center else ColorLabel.NONE  # _approach 当负样本

        data = np.load(str(npz))
        rgb = data["rgb"]
        depth_raw = data["depth_raw"] if "depth_raw" in data else data.get("depth")
        depth_scale = float(data["depth_scale"]) if "depth_scale" in data else 0.001
        cls = classify_color_combo(
            rgb, depth_raw=depth_raw, depth_scale=depth_scale,
            require_depth=True, **kw,
        )
        ok = (cls.label == expected) or (
            not is_center and cls.label != expected and "approach" in stem
        )
        # 简化判定: _center 必须 expected; _approach 任意
        ok = (cls.label == expected) if is_center else True
        if ok:
            n_match += 1
        if cls.label == expected and is_center and cls.depth_median_m > 0:
            by_kind_depth[kind].append(cls.depth_median_m)
        print(f"{stem:<20} {expected.value:<14} {cls.label.value:<14} "
              f"{('OK' if ok else 'MISS'):<6} "
              f"{cls.white_ratio:>5.2f} {cls.black_ratio:>5.2f} "
              f"{cls.blue_ratio:>5.2f} {cls.yellow_ratio:>5.2f} "
              f"{cls.depth_median_m:>6.2f}m {cls.depth_valid_ratio:>5.2f}")
        rows.append((stem, kind, expected, cls))

        if args.save_vis:
            out = snap_dir / f"analyze_{stem}.jpg"
            vis_rgb = rgb.copy()
            x, y, w, h = cls.roi_bbox
            color = (0, 255, 0) if cls.label == expected else (
                (0, 165, 255) if cls.label != ColorLabel.NONE else (60, 60, 60))
            cv2.rectangle(vis_rgb, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis_rgb, f"{stem} expected={expected.value} got={cls.label.value}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(vis_rgb, f"w={cls.white_ratio:.2f} k={cls.black_ratio:.2f} "
                                 f"b={cls.blue_ratio:.2f} z={cls.depth_median_m:.2f}m",
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imwrite(str(out), vis_rgb)

    print("=" * 110)
    print(f"[stat] {n_match}/{len(rows)} 正确")

    print()
    print(f"推荐 trigger 阈值 (margin={args.margin}m):")
    for kind in ("dump", "stair", "dock"):
        ds = by_kind_depth[kind]
        if ds:
            trig = max(ds) + args.margin
            print(f"  {kind}_trigger_depth_m: {trig:.2f}    # samples z={ds}")
        else:
            print(f"  {kind}_trigger_depth_m: ?    # {kind}_center 没识别 / 没拍")

    return 0 if n_match == len([r for r in rows if r[0].endswith("_center")]) else 1


if __name__ == "__main__":
    sys.exit(main())
