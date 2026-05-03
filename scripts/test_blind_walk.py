"""狗盲走测试 - 完全不依赖视觉/FSM, 只测狗能不能 set_velocity 走起来.

用法 (狗端):
  python3 scripts/test_blind_walk.py --network eth0 --duration 5 --vx 0.25

注意: vx 必须 ≥ 0.22 (Go2 sport_mode 起步门槛), 否则只原地踏步.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.robot.go2_client import Go2Client, Go2ClientError  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/params.yaml")
    parser.add_argument("--network", default=None)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--vx", type=float, default=0.25)
    parser.add_argument("--vyaw", type=float, default=0.0)
    parser.add_argument("--no-stand", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = get_logger("blindwalk", level="INFO", save_dir="logs")
    iface = args.network or cfg.network.interface

    log.info("===== 盲走测试 (vx=%.2f duration=%.1fs) =====",
             args.vx, args.duration)
    if args.vx < 0.22:
        log.warning("⚠ vx=%.2f < 0.22, Go2 sport_mode 可能只原地踏步!", args.vx)

    try:
        robot = Go2Client(
            network_iface=iface,
            domain_id=int(cfg.network.domain_id),
            max_vx=float(getattr(cfg.robot, "max_vx", 0.5)),
            max_vy=float(getattr(cfg.robot, "max_vy", 0.3)),
            max_vyaw=float(getattr(cfg.robot, "max_vyaw", 1.0)),
            velocity_hz=int(getattr(cfg.robot, "velocity_hz", 20)),
            dry_run=False, logger=log,
        )
        robot.init()
    except Go2ClientError as e:
        log.error("连狗失败: %s", e)
        return 2

    stopping = {"flag": False}
    def _on_sigint(_s, _f):
        log.warning("Ctrl+C, 停下狗")
        stopping["flag"] = True
    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    try:
        if not args.no_stand:
            robot.stand_up()
            time.sleep(1.5)
            robot.balance_stand()
            time.sleep(0.5)

        log.info("开始盲走...")
        t0 = time.time()
        last_log = t0
        while not stopping["flag"]:
            elapsed = time.time() - t0
            if elapsed >= args.duration:
                break
            robot.set_velocity(args.vx, 0.0, args.vyaw)
            now = time.time()
            if now - last_log >= 1.0:
                log.info("  [%.1fs / %.1fs]", elapsed, args.duration)
                last_log = now
            time.sleep(0.05)

        log.info("停下")
        robot.set_velocity(0.0, 0.0, 0.0)
        time.sleep(0.3)
        robot.stop_move()
        time.sleep(0.5)
    finally:
        robot.shutdown()
    log.info("===== 结束 =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
