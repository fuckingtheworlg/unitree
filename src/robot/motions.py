"""组合动作: 卸料姿态 / 爬楼步态切换 / 停车收尾.

关键约束 (因为新 Go2Client 是线程化 Move spam):
  - 卸料用 Euler 持续保持身体倾角 -> 必须先 pause_velocity()
    (否则 velocity 线程的 Move(0,0,0) 会跟 Euler 抢)
  - 爬楼切 ClassicWalk -> 也最好 pause_velocity 防止切换瞬间冲突
  - 退出后 resume_velocity() 让主巡线 set_velocity 继续生效

只用 SportClient 真实存在的 API (已对照 .deps/ 本地源码核实):
  Sit / RiseSit / Stretch / Euler / ClassicWalk(bool) / TrotRun / StaticWalk
"""

from __future__ import annotations

import time

from .go2_client import Go2Client


def execute_dump_action(
    client: Go2Client,
    roll_rad: float = -0.45,
    hold_sec: float = 2.5,
    spam_hz: int = 30,
) -> None:
    """向右倾斜身体, 让物料从右侧滑入倾倒区.

    Go2 Euler(roll, pitch, yaw):
      负 roll = 身体向**右**倾斜 → 物料从右侧滑出
      正 roll = 身体向**左**倾斜

    赛前对着筐子和水瓶调 roll_rad 的大小 (0.3~0.6 rad 范围内).

    流程 (velocity 线程被暂停, 狗静止不动):
      1. pause_velocity + StopMove
      2. BalanceStand 稳住
      3. hold_sec 内持续 spam Euler(roll_rad, 0, 0)
      4. 复位 Euler(0,0,0) + BalanceStand
      5. resume_velocity
    """
    client.pause_velocity()
    time.sleep(0.3)
    client.balance_stand()
    time.sleep(0.5)

    interval = 1.0 / max(1, spam_hz)
    end_at = time.time() + hold_sec
    while time.time() < end_at:
        client.euler(float(roll_rad), 0.0, 0.0)
        time.sleep(interval)

    end_at = time.time() + 0.5
    while time.time() < end_at:
        client.euler(0.0, 0.0, 0.0)
        time.sleep(interval)
    client.balance_stand()
    time.sleep(0.3)

    client.resume_velocity()


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
    client.resume_velocity()


def final_dock(client: Go2Client) -> None:
    """到达充电区后的收尾: 停下 + 站立保持."""
    client.pause_velocity()
    time.sleep(0.3)
    client.balance_stand()
    time.sleep(0.5)
    client.resume_velocity()
    client.stop_move()
