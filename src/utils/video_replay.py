"""离线视频回灌：在 Mac 上用一段录屏 / 拍的视频，跑整个视觉管线 + FSM。

例:
    python -m src.utils.video_replay tests/data/sample.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.main import MissionRunner, _DEFAULT_CFG
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.vision.camera import VideoFileSource


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="本地视频文件 (mp4/avi)")
    parser.add_argument("--config", default=str(_DEFAULT_CFG))
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    if not Path(args.video).exists():
        raise FileNotFoundError(args.video)

    cfg = load_config(args.config)
    log = get_logger("replay", level=cfg.logging.level)

    source = VideoFileSource(args.video, loop=False)
    runner = MissionRunner(cfg, source, robot=None, logger=log)
    try:
        runner.run(display=not args.no_display)
    finally:
        source.release()


if __name__ == "__main__":
    main()
