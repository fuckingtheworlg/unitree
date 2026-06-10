"""
Go2 EDU 高层运动 + 视频客户端的薄封装 (线程化 Move 版本).

为什么要线程化:
  Go2 EDU 上 sport_mode 服务要求 Move(vx,vy,vyaw) **持续以 ~20Hz 重发**才会真正
  保持行走 (单次发送 1 秒后就停下). 主控制循环 30Hz 跑视觉/PID/FSM, 中间 imshow
  和 PID 计算可能造成抖动间隔. 把 Move spam 放到独立 daemon 线程, 主线程只设
  目标速度 (`set_velocity`), 让 spam 线程稳定 20Hz 下发, 是最稳的写法.

仅使用 unitree_sdk2_python 官方真实存在的 API (已对照 .deps/ 本地源码核实):
  - SportClient: Init, SetTimeout, BalanceStand, StopMove, StandUp, StandDown,
                 RecoveryStand, Damp, Move, Euler, SpeedLevel, Sit, RiseSit,
                 Hello, Stretch, Scrape, Pose(bool), ClassicWalk(bool),
                 CrossStep(bool), StaticWalk, TrotRun
  - VideoClient: Init, SetTimeout, GetImageSample
  - ChannelFactoryInitialize(id, networkInterface)

赛前必须做的 "开机仪式" (一次性):
  1. 用 Unitree Go2 App 进入 "服务状态" 页, 关闭 mcf, 打开 sport (互斥)
  2. 重启狗后此设置丢失, 比赛前再做一次

使用模式:
  client = Go2Client(...)
  client.init()                          # 启动 SDK + 启动 velocity 线程
  client.stand_up()                      # 让狗站起来
  client.set_velocity(0.3, 0, 0)         # 后台线程开始 20Hz spam Move(0.3, 0, 0)
  ...                                    # 主线程做视觉/PID, 任意时刻调 set_velocity 改速度
  client.set_velocity(0, 0, 0)           # 改 0 = 停下 (但线程还在跑)
  client.pause_velocity()                # 完全暂停后台 Move (用于 Euler 卸料动作)
  client.euler(0, -0.45, 0)              # 主线程直接发 Euler (不会被 Move 覆盖)
  client.resume_velocity()               # 恢复 Move spam (从 0 开始)
  client.shutdown()                      # 停线程 + StopMove + BalanceStand
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple


class Go2ClientError(RuntimeError):
    pass


class Go2Client:
    def __init__(
        self,
        network_iface: str = "eth0",
        domain_id: int = 0,
        max_vx: float = 0.5,
        max_vy: float = 0.3,
        max_vyaw: float = 1.0,
        velocity_hz: int = 20,
        dry_run: bool = False,
        logger=None,
    ):
        self.iface = network_iface
        self.domain_id = domain_id
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_vyaw = max_vyaw
        self.velocity_hz = max(1, int(velocity_hz))
        self.dry_run = bool(dry_run)
        self.log = logger

        self._sport = None
        self._video = None
        self._initialized = False

        self._velocity_lock = threading.Lock()
        self._sport_lock = threading.Lock()
        self._target_vx = 0.0
        self._target_vy = 0.0
        self._target_vyaw = 0.0

        self._velocity_thread: Optional[threading.Thread] = None
        self._velocity_stop = threading.Event()
        self._velocity_paused = threading.Event()

    # -------- 生命周期 --------

    def init(self) -> None:
        if self._initialized:
            return
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            from unitree_sdk2py.go2.video.video_client import VideoClient
        except ImportError as e:
            raise Go2ClientError(
                "未找到 unitree_sdk2_python，先在本机或狗端跑 setup_*.sh"
            ) from e

        ChannelFactoryInitialize(self.domain_id, self.iface)

        self._sport = SportClient()
        self._sport.SetTimeout(5.0)
        self._sport.Init()

        self._video = VideoClient()
        self._video.SetTimeout(3.0)
        self._video.Init()

        self._initialized = True
        self._log("Go2 SDK initialized on iface=%s domain=%d (dry_run=%s)",
                  self.iface, self.domain_id, self.dry_run)

        if not self.dry_run:
            self._start_velocity_thread()
        else:
            self._log("dry_run 模式: velocity 线程不启动, 所有运动指令变 noop")

    def shutdown(self, damp: bool = False) -> None:
        if not self._initialized or self._sport is None:
            return
        self._stop_velocity_thread()
        if self.dry_run:
            return
        try:
            with self._sport_lock:
                self._sport.StopMove()
                time.sleep(0.1)
                if damp:
                    self._sport.Damp()
                else:
                    self._sport.BalanceStand()
        except Exception as e:
            self._log("shutdown 异常 (忽略): %s", e)

    # -------- velocity 线程 --------

    def _start_velocity_thread(self) -> None:
        if self._velocity_thread is not None and self._velocity_thread.is_alive():
            return
        self._velocity_stop.clear()
        self._velocity_paused.clear()
        self._target_vx = 0.0
        self._target_vy = 0.0
        self._target_vyaw = 0.0
        self._velocity_thread = threading.Thread(
            target=self._velocity_loop, daemon=True, name="go2_velocity_spam"
        )
        self._velocity_thread.start()
        self._log("velocity 线程启动 (%dHz)", self.velocity_hz)

    def _stop_velocity_thread(self) -> None:
        if self._velocity_thread is None:
            return
        self._velocity_stop.set()
        self._velocity_thread.join(timeout=2.0)
        self._velocity_thread = None

    def _velocity_loop(self) -> None:
        interval = 1.0 / self.velocity_hz
        sent = 0
        while not self._velocity_stop.is_set():
            if not self._velocity_paused.is_set():
                with self._velocity_lock:
                    vx = self._target_vx
                    vy = self._target_vy
                    vyaw = self._target_vyaw
                try:
                    with self._sport_lock:
                        self._sport.Move(vx, vy, vyaw)
                    sent += 1
                except Exception as e:
                    self._log("velocity Move 异常 (忽略): %s", e)
            time.sleep(interval)
        self._log("velocity 线程退出 (共发 %d 条 Move)", sent)

    def pause_velocity(self) -> None:
        """暂停后台 Move spam, 让 Euler/StandDown 等动作不被覆盖. 同时清零目标速度."""
        self._velocity_paused.set()
        with self._velocity_lock:
            self._target_vx = 0.0
            self._target_vy = 0.0
            self._target_vyaw = 0.0
        if self.dry_run:
            return
        if self._sport is not None:
            try:
                with self._sport_lock:
                    self._sport.StopMove()
            except Exception:
                pass

    def resume_velocity(self) -> None:
        """恢复后台 Move spam (目标速度重置为 0)."""
        with self._velocity_lock:
            self._target_vx = 0.0
            self._target_vy = 0.0
            self._target_vyaw = 0.0
        self._velocity_paused.clear()

    # -------- 运动指令 --------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """设置目标速度. 后台线程会以 velocity_hz spam Move(vx,vy,vyaw).
        dry_run 模式下: 仅记录, 不发任何指令."""
        self._require_init()
        vx = _clamp(vx, -self.max_vx, self.max_vx)
        vy = _clamp(vy, -self.max_vy, self.max_vy)
        vyaw = _clamp(vyaw, -self.max_vyaw, self.max_vyaw)
        with self._velocity_lock:
            self._target_vx = float(vx)
            self._target_vy = float(vy)
            self._target_vyaw = float(vyaw)

    # 兼容老接口
    def safe_move(self, vx: float, vy: float, vyaw: float) -> None:
        self.set_velocity(vx, vy, vyaw)

    def stop_move(self) -> None:
        """立刻停下: 速度清零 + 调 StopMove (双保险)."""
        with self._velocity_lock:
            self._target_vx = 0.0
            self._target_vy = 0.0
            self._target_vyaw = 0.0
        if self.dry_run:
            return
        if self._sport is not None:
            try:
                with self._sport_lock:
                    self._sport.StopMove()
            except Exception:
                pass

    def stand_up(self) -> None:
        self._require_init()
        if self.dry_run: return self._log_dry("stand_up")
        self._call_sport("StandUp")
        time.sleep(1.5)
        self._call_sport("BalanceStand")
        time.sleep(0.5)
        self._log("StandUp + BalanceStand 完成")

    def stand_down(self) -> None:
        self._require_init()
        if self.dry_run: return self._log_dry("stand_down")
        self._call_sport("StandDown")
        time.sleep(1.0)

    def recovery_stand(self) -> None:
        self._require_init()
        if self.dry_run: return self._log_dry("recovery_stand")
        self._call_sport("RecoveryStand")
        time.sleep(1.0)

    def balance_stand(self) -> None:
        self._require_init()
        if self.dry_run: return self._log_dry("balance_stand")
        self._call_sport("BalanceStand")

    def damp(self) -> None:
        self._require_init()
        if self.dry_run: return self._log_dry("damp")
        self._call_sport("Damp")

    def set_speed_level(self, level: int) -> None:
        """level 取 0/1/2 三档 (慢/正常/快)."""
        self._require_init()
        if self.dry_run: return self._log_dry(f"set_speed_level({level})")
        self._call_sport("SpeedLevel", int(level))

    def euler(self, roll: float, pitch: float, yaw: float) -> None:
        """姿态控制 (rad). 此 API 只生效一帧, 持续调用才能保持姿态.
        ⚠ 调用前必须先 pause_velocity(), 否则会被 Move spam 覆盖."""
        self._require_init()
        if self.dry_run: return
        self._call_sport("Euler", float(roll), float(pitch), float(yaw), warn_nonzero=False)

    def sit(self) -> None:
        self._require_init()
        if self.dry_run: return self._log_dry("sit")
        self._call_sport("Sit")

    def rise_sit(self) -> None:
        self._require_init()
        if self.dry_run: return self._log_dry("rise_sit")
        self._call_sport("RiseSit")

    def stretch(self) -> None:
        """伸懒腰动作 (前低后高)."""
        self._require_init()
        if self.dry_run: return self._log_dry("stretch")
        self._call_sport("Stretch")

    def hello(self) -> None:
        self._require_init()
        if self.dry_run: return self._log_dry("hello")
        self._call_sport("Hello")

    def scrape(self) -> None:
        """官方 Scrape 预置动作. 是否适合卸料需实测."""
        self._require_init()
        if self.dry_run: return self._log_dry("scrape")
        self._call_sport("Scrape")

    def pose(self, enable: bool) -> None:
        """切换姿态控制开关. Go2 mcf 下可能不接受, 返回码会记录."""
        self._require_init()
        if self.dry_run: return self._log_dry(f"pose({enable})")
        self._call_sport("Pose", bool(enable))

    def classic_walk(self, enable: bool) -> None:
        """切换到 / 退出经典步态 (更稳, 适合上下台阶)."""
        self._require_init()
        if self.dry_run: return self._log_dry(f"classic_walk({enable})")
        self._call_sport("ClassicWalk", bool(enable))

    def cross_step(self, enable: bool) -> None:
        """官方 CrossStep 预置动作. 是否适合卸料需实测."""
        self._require_init()
        if self.dry_run: return self._log_dry(f"cross_step({enable})")
        self._call_sport("CrossStep", bool(enable))

    def static_walk(self) -> None:
        """静步行 (更稳但更慢)."""
        self._require_init()
        if self.dry_run: return self._log_dry("static_walk")
        self._call_sport("StaticWalk")

    def trot_run(self) -> None:
        """trot 跑步步态 (高速)."""
        self._require_init()
        if self.dry_run: return self._log_dry("trot_run")
        self._call_sport("TrotRun")

    # -------- 视频 --------

    def get_image_sample(self) -> Tuple[int, Optional[bytes]]:
        """返回 (code, data); code==0 成功; data 是 JPEG 字节流."""
        self._require_init()
        return self._video.GetImageSample()

    # -------- 内部 --------

    def _require_init(self) -> None:
        if not self._initialized:
            raise Go2ClientError("Go2Client.init() 未调用")

    def _log(self, msg: str, *args) -> None:
        if self.log is not None:
            self.log.info(msg, *args)

    def _log_dry(self, action: str) -> None:
        if self.log is not None:
            self.log.debug("[dry_run] %s noop", action)

    def _call_sport(self, method: str, *args, warn_nonzero: bool = True):
        """调用 SportClient API 并记录非 0 返回码.

        官方高层 API 多数会返回 code。之前忽略返回码会把"命令被 mcf/固件拒绝"
        伪装成动作执行成功, 不利于现场排障。
        """
        with self._sport_lock:
            fn = getattr(self._sport, method)
            code = fn(*args)
        if warn_nonzero and code not in (None, 0):
            self._log("SportClient.%s%r return code=%s", method, args, code)
        return code


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
