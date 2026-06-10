"""任务状态机。

状态流：
INIT
  → STAND_UP
  → FOLLOW_LANE     (默认巡线)
  → FOLLOW_OBSTACLE_RING (起点后障碍区蓝色圆环: 固定右侧沿环带绕行)
  → FOLLOW_LANE
  → FOLLOW_DUMP_RING (检测到倾倒区: 沿黄色圆环一侧慢速到动作点)
  → DUMP_ACTION     (执行卸料姿态)
  → FOLLOW_LANE
  → CHOOSE_FORK     (检测到岔路: 选最短/速度最快路径 = 选当前画面里更"直"或者更近的那支)
  → FOLLOW_LANE
  → APPROACH_STAIR  (检测到台阶接近: 切爬楼步态)
  → CLIMB_STAIR     (定时上楼)
  → APPROACH_DOCK   (检测充电区蓝色矩形后继续进区)
  → DOCK            (停车收尾)
  → DONE

异常分支：
  - 视野丢失黄色超过 lost_lane_timeout_sec → 进入 SEARCH_LANE (原地小角度摆头找线)
  - 整体任务超时 max_duration_sec → 直接进入 EMERGENCY_STOP
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class State(str, Enum):
    INIT = "INIT"
    STAND_UP = "STAND_UP"
    FOLLOW_LANE = "FOLLOW_LANE"
    FOLLOW_OBSTACLE_RING = "FOLLOW_OBSTACLE_RING"
    AVOID_OBSTACLE = "AVOID_OBSTACLE"
    FOLLOW_DUMP_RING = "FOLLOW_DUMP_RING"
    APPROACH_DUMP = "APPROACH_DUMP"
    DUMP_ACTION = "DUMP_ACTION"
    CHOOSE_FORK = "CHOOSE_FORK"
    APPROACH_STAIR = "APPROACH_STAIR"
    CLIMB_STAIR = "CLIMB_STAIR"
    APPROACH_DOCK = "APPROACH_DOCK"
    DOCK = "DOCK"
    SEARCH_LANE = "SEARCH_LANE"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    DONE = "DONE"


@dataclass
class FlowFlags:
    """已经做过哪些不可逆事件，避免反复触发。"""
    dumped: bool = False
    obstacle_avoided: bool = False
    blue_ring_done: bool = False
    fork_chosen: bool = False
    stair_climbed: bool = False


class MissionFSM:
    def __init__(self, logger=None):
        self.state: State = State.INIT
        self.flags = FlowFlags()
        self.entered_at: float = time.time()
        self.log = logger
        self._consecutive_dump = 0
        self._consecutive_obstacle = 0
        self._consecutive_fork = 0
        self._consecutive_stair = 0
        self._consecutive_dock = 0
        self._consecutive_lane = 0
        self._consecutive_lost = 0

    def transit(self, new_state: State) -> None:
        if new_state == self.state:
            return
        if self.log:
            self.log.info("[FSM] %s -> %s", self.state.value, new_state.value)
        self.state = new_state
        self.entered_at = time.time()
        self._consecutive_dump = 0
        self._consecutive_obstacle = 0
        self._consecutive_fork = 0
        self._consecutive_stair = 0
        self._consecutive_dock = 0
        self._consecutive_lane = 0
        self._consecutive_lost = 0

    def time_in_state(self) -> float:
        return time.time() - self.entered_at

    def vote_dump(self, hit: bool, n_required: int) -> bool:
        self._consecutive_dump = self._consecutive_dump + 1 if hit else 0
        return self._consecutive_dump >= n_required

    def vote_obstacle(self, hit: bool, n_required: int) -> bool:
        self._consecutive_obstacle = self._consecutive_obstacle + 1 if hit else 0
        return self._consecutive_obstacle >= n_required

    def vote_fork(self, hit: bool, n_required: int) -> bool:
        self._consecutive_fork = self._consecutive_fork + 1 if hit else 0
        return self._consecutive_fork >= n_required

    def vote_stair(self, hit: bool, n_required: int) -> bool:
        self._consecutive_stair = self._consecutive_stair + 1 if hit else 0
        return self._consecutive_stair >= n_required

    def vote_dock(self, hit: bool, n_required: int) -> bool:
        self._consecutive_dock = self._consecutive_dock + 1 if hit else 0
        return self._consecutive_dock >= n_required

    def vote_lane_recovered(self, hit: bool, n_required: int) -> bool:
        """连续 N 帧黄道高置信度识别 = 视为狗已穿过当前区域, 黄道重现."""
        self._consecutive_lane = self._consecutive_lane + 1 if hit else 0
        return self._consecutive_lane >= n_required

    def vote_landmark_lost(self, lost: bool, n_required: int) -> bool:
        """连续 N 帧地标消失 = 狗可能已经走过去了 (用作退出 APPROACH_X 的辅助条件)."""
        self._consecutive_lost = self._consecutive_lost + 1 if lost else 0
        return self._consecutive_lost >= n_required


def choose_shortest_fork_branch(left_x: float, right_x: float, image_w: int) -> str:
    """
    岔路选径策略 = 速度最快/时间最少：
      因省赛地图右侧岔路是顺时针的"O 型"绕路、左侧是直道，
      选**离当前航向中心更近**的那支即可：偏差小 = 拐弯少 = 用时短。
    """
    cx = image_w / 2.0
    if abs(left_x - cx) <= abs(right_x - cx):
        return "left"
    return "right"
