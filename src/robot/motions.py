"""组合动作: 卸料姿态 / 爬楼步态切换 / 停车收尾.

关键约束 (因为新 Go2Client 是线程化 Move spam):
  - Go2 EDU 实测 mcf 模式下 Euler/低层 release 都不可靠, 默认卸料改用高层
    Stretch/Sit/RiseSit 组合, 不释放 mcf.
  - 若显式选择 euler, 仍必须先 pause_velocity()
    (否则 velocity 线程的 Move(0,0,0) 会跟 Euler 抢)
  - 爬楼切 ClassicWalk -> 也最好 pause_velocity 防止切换瞬间冲突
  - 退出后 resume_velocity() 让主巡线 set_velocity 继续生效

只用 SportClient 真实存在的 API (已对照 .deps/ 本地源码核实):
  Sit / RiseSit / Stretch / Scrape / Euler / Pose(bool) / ClassicWalk(bool) /
  CrossStep(bool) / TrotRun / StaticWalk
"""

from __future__ import annotations

import time

from .go2_client import Go2Client


def execute_dump_action(
    client: Go2Client,
    method: str = "stretch",
    roll_rad: float = -0.45,
    hold_sec: float = 2.5,
    spam_hz: int = 30,
    repeat: int = 2,
) -> None:
    """执行卸料动作.

    method="sit_rise" 可实现实测成功的动作: "前腿伸直/基本不变, 后腿跪下"。
    默认 method 保持由配置决定；本函数只提供可选动作实现。

    可选 method:
      - sit_rise: 后腿跪下保持后再起立
      - stretch: 高层伸展动作, 会前后交替下沉
      - scrape: 官方 Scrape 预置动作, 可能包含侧向刮地/晃动, 需实测
      - cross_step: 官方 CrossStep 预置动作, 可能包含侧向交叉步, 需实测
      - pose_stretch: Pose(True) 后 Stretch, 部分固件可能不接受
      - euler: 旧方案, mcf 下实测可能无效, 仅保留作回退
    """
    method = (method or "stretch").strip().lower()
    client.pause_velocity()
    time.sleep(0.3)
    try:
        client.balance_stand()
        time.sleep(0.5)

        if method == "stretch":
            _dump_by_stretch(client, hold_sec=hold_sec, repeat=repeat)
        elif method == "sit_rise":
            _dump_by_sit_rise(client, hold_sec=hold_sec, repeat=repeat)
        elif method == "scrape":
            _dump_by_scrape(client, hold_sec=hold_sec, repeat=repeat)
        elif method == "cross_step":
            _dump_by_cross_step(client, hold_sec=hold_sec)
        elif method == "pose_stretch":
            client.pose(True)
            time.sleep(0.2)
            _dump_by_stretch(client, hold_sec=hold_sec, repeat=repeat)
            client.pose(False)
        elif method == "euler":
            _dump_by_euler(client, roll_rad=roll_rad, hold_sec=hold_sec, spam_hz=spam_hz)
        else:
            raise ValueError(f"unknown dump_action method: {method}")

        client.balance_stand()
        time.sleep(0.5)
    finally:
        client.resume_velocity()


def _dump_by_stretch(client: Go2Client, hold_sec: float, repeat: int) -> None:
    for _ in range(max(1, int(repeat))):
        client.stretch()
        time.sleep(max(0.8, float(hold_sec)))
        client.balance_stand()
        time.sleep(0.4)


def _dump_by_sit_rise(client: Go2Client, hold_sec: float, repeat: int) -> None:
    for _ in range(max(1, int(repeat))):
        client.sit()
        time.sleep(max(1.0, float(hold_sec)))
        client.rise_sit()
        time.sleep(1.0)
        client.balance_stand()
        time.sleep(0.4)


def _dump_by_scrape(client: Go2Client, hold_sec: float, repeat: int) -> None:
    for _ in range(max(1, int(repeat))):
        client.scrape()
        time.sleep(max(1.0, float(hold_sec)))
        client.balance_stand()
        time.sleep(0.5)


def _dump_by_cross_step(client: Go2Client, hold_sec: float) -> None:
    client.cross_step(True)
    time.sleep(max(1.0, float(hold_sec)))
    client.cross_step(False)
    time.sleep(0.8)


def _dump_by_euler(
    client: Go2Client,
    roll_rad: float,
    hold_sec: float,
    spam_hz: int,
) -> None:
    interval = 1.0 / max(1, spam_hz)
    end_at = time.time() + hold_sec
    while time.time() < end_at:
        client.euler(float(roll_rad), 0.0, 0.0)
        time.sleep(interval)

    end_at = time.time() + 0.5
    while time.time() < end_at:
        client.euler(0.0, 0.0, 0.0)
        time.sleep(interval)


def enter_stair_mode(client: Go2Client) -> None:
    """切到经典步态 (抬腿幅度更大、落足更稳, 适合上下台阶).
    切换瞬间会暂停 velocity 线程, 切完恢复."""
    client.pause_velocity()
    time.sleep(0.2)
    client.classic_walk(True)
    time.sleep(0.5)
    client.resume_velocity()


def exit_stair_mode(client: Go2Client) -> None:
    """退出经典步态, 回到默认 trot 步态."""
    client.pause_velocity()
    time.sleep(0.2)
    client.classic_walk(False)
    time.sleep(0.5)
    client.balance_stand()
    time.sleep(0.3)
    client.resume_velocity()


def final_dock(client: Go2Client) -> None:
    """到达充电区后的收尾: 停下 + 站立保持."""
    client.pause_velocity()
    time.sleep(0.3)
    client.balance_stand()
    time.sleep(0.5)
    client.resume_velocity()
    client.stop_move()
