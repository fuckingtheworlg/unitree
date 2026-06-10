"""Go2 EDU 道路识别赛 - 巡线主程序。

用法 (在狗的 Orin Nano 上):
    python -m src.main --network eth0 --province

可选:
    --replay path/to/video.mp4   # 离线视频回灌, 不连狗, 仅打印决策
    --no-display                 # 不开 GUI 窗口 (赛场默认)
    --headless-record path.mp4   # 把可视化叠加层录到文件
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from src.control.fsm import MissionFSM, State, choose_shortest_fork_branch
from src.control.pid import PID, PIDParams
from src.robot.go2_client import Go2Client, Go2ClientError
from src.robot.motions import (
    enter_stair_mode,
    execute_dump_action,
    exit_stair_mode,
    final_dock,
)
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.mjpeg_server import MjpegServer
from src.vision.camera import FrameSource, Go2CameraSource, VideoFileSource
from src.vision.debug_view import overlay
from src.vision.lane_follow import (
    LaneResult,
    estimate_lane_error,
    find_largest_yellow_centroid,
)
from src.vision.landmark import (
    LandmarkType,
    detect_blue_obstacle_ring,
    detect_dock_area,
    detect_dump_zone,
    detect_fork,
    detect_stair,
)
from src.vision.realsense_target import (
    ColorClassification,
    ColorLabel,
    ColorTriggerState,
    classify_color_combo,
)
from src.vision.realsense_lane import estimate_bottom_blue_ring_lane, estimate_bottom_yellow_lane
from src.vision.undistort import IPMConfig, IPMTransformer
from src.vision.yellow_mask import crop_roi, yellow_mask

try:
    from src.vision.realsense_camera import RealsenseCamera, RealsenseFrame
except Exception:
    RealsenseCamera = None  # type: ignore
    RealsenseFrame = None  # type: ignore


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _REPO_ROOT / "config" / "params.yaml"


class MissionRunner:
    def __init__(
        self,
        cfg,
        source: FrameSource,
        robot: Optional[Go2Client],
        logger,
        realsense_cam: Optional["RealsenseCamera"] = None,
        mjpeg_server: Optional[MjpegServer] = None,
    ):
        self.cfg = cfg
        self.source = source
        self.robot = robot
        self.log = logger
        self.rs_cam = realsense_cam
        self.mjpeg = mjpeg_server
        self._last_rs_frame: Optional["RealsenseFrame"] = None
        self._last_color_cls: Optional[ColorClassification] = None
        self._last_rs_lane: Optional[LaneResult] = None
        self._last_rs_blue_ring_lane: Optional[LaneResult] = None
        self._last_blue_ring_log_t: float = 0.0
        self._lane_control_source: str = "front"
        self._last_lane_control_source: Optional[str] = None
        self._last_rs_log_t: float = 0.0
        self._color_triggers: Dict[str, ColorTriggerState] = {}
        self._post_dump_recover_until_t: float = 0.0
        self._dump_done_at: float = 0.0
        self._blind_walk_started_at: float = 0.0
        self._post_stair_straight_ready_at: float = 0.0
        self._post_stair_settle_logged: bool = False
        self._dump_post_trigger_active: bool = False
        self._dump_ring_confirmed: bool = False
        self._dock_post_trigger_active: bool = False
        self._obstacle_seq_active: bool = False
        self._obstacle_phase: Optional[str] = None
        self._obstacle_phase_t: float = 0.0
        self._obstacle_red_count: int = 0
        self._cmd_vy: float = 0.0

        pid_cfg = cfg.control.lateral_pid
        self.pid = PID(PIDParams(
            kp=pid_cfg.kp, ki=pid_cfg.ki, kd=pid_cfg.kd,
            out_min=pid_cfg.out_min, out_max=pid_cfg.out_max,
            integral_limit=pid_cfg.integral_limit,
        ))

        ipm_cfg_raw = cfg.camera.ipm_src_ratio
        self.ipm = IPMTransformer(IPMConfig(
            src_tl=tuple(ipm_cfg_raw.tl),
            src_tr=tuple(ipm_cfg_raw.tr),
            src_br=tuple(ipm_cfg_raw.br),
            src_bl=tuple(ipm_cfg_raw.bl),
            dst_size=tuple(cfg.camera.ipm_dst_size),
        ))

        self.fsm = MissionFSM(logger=logger)
        self.fork_choice: Optional[str] = None
        self.last_lane_seen_at: float = time.time()
        self.start_time: float = time.time()
        self._last_frame_time = time.time()
        self._fps = 0.0
        self._last_lane_found: Optional[bool] = None
        self._last_frame: Optional["np.ndarray"] = None
        self._last_warped: Optional["np.ndarray"] = None
        self._last_mask: Optional["np.ndarray"] = None
        self._last_debug: Optional["np.ndarray"] = None
        self._max_prox_seen: float = 0.0
        self._prox_track_state: Optional[State] = None
        self._stair_mode_entered: bool = False
        self._approach_start_time: float = 0.0
        self._approach_distance_m: float = 0.0
        self._approach_last_tick: float = 0.0
        self._approach_track_state: Optional[State] = None
        # passive 模式: dry-run 或 replay 都不该让 EMERGENCY_STOP / DONE 终止程序,
        # 否则视频流会断开. 改成自动重置回 FOLLOW_LANE 持续展示视频, 方便调参.
        self._is_passive = (robot is None) or (robot is not None and getattr(robot, "dry_run", False))

    def run(self, display: bool = True, video_writer=None) -> None:
        cfg = self.cfg
        loop_hz = max(1, int(cfg.control.loop_hz))
        loop_dt = 1.0 / loop_hz

        if self.robot is not None:
            self.fsm.transit(State.STAND_UP)
            self.robot.stand_up()
            try:
                self.robot.set_speed_level(int(cfg.robot.speed_level))
            except Exception as e:
                self.log.warning("SpeedLevel 失败 (忽略): %s", e)
            time.sleep(float(cfg.robot.startup_grace_sec))
            self.fsm.transit(State.FOLLOW_LANE)
        else:
            self.fsm.transit(State.FOLLOW_LANE)

        self.last_lane_seen_at = time.time()
        self.start_time = time.time()

        try:
            while True:
                tick = time.time()
                if (tick - self.start_time) > float(cfg.mission.max_duration_sec):
                    self.log.warning("任务超时，进入 EMERGENCY_STOP")
                    self.fsm.transit(State.EMERGENCY_STOP)

                frame = self.source.read()
                if frame is None:
                    self.log.warning("无图像帧, 退出")
                    break

                self._update_fps(tick)
                vx, vyaw, debug_img = self._step(frame)

                if self.robot is not None and self.fsm.state not in (
                    State.DUMP_ACTION,
                    State.CLIMB_STAIR,
                    State.DOCK,
                    State.DONE,
                    State.EMERGENCY_STOP,
                ):
                    self.robot.set_velocity(vx, float(getattr(self, "_cmd_vy", 0.0)), vyaw)

                if display:
                    title = "go2 patrol [PASSIVE/DRY-RUN]" if self._is_passive else "go2 patrol [LIVE]"
                    cv2.imshow(title, debug_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        self.log.info("用户按 q/ESC 退出")
                        break
                    elif key == ord("s"):
                        self._save_debug_snapshot()
                    elif key == ord("n"):
                        self._manual_advance_state()
                if video_writer is not None:
                    video_writer.write(debug_img)

                if self.fsm.state == State.DONE:
                    if self._is_passive:
                        self.log.info("[passive] DONE 状态, 但保持窗口运行 (按 q 退出)")
                        self._reset_to_follow_lane()
                    else:
                        self.log.info("任务完成 ✓ (用时 %.1fs)", time.time() - self.start_time)
                        break
                if self.fsm.state == State.EMERGENCY_STOP:
                    if self.robot is not None:
                        self.robot.stop_move()
                    if self._is_passive:
                        self.log.warning("[passive] EMERGENCY_STOP, 但保持窗口运行 (按 q 退出); 重置 FSM")
                        self._reset_to_follow_lane()
                    else:
                        break

                elapsed = time.time() - tick
                if elapsed < loop_dt:
                    time.sleep(loop_dt - elapsed)
        finally:
            if self.robot is not None:
                self.robot.shutdown()
            if display:
                cv2.destroyAllWindows()

    # ===== D435i 辅助 =====
    def _read_realsense(self) -> Optional["RealsenseFrame"]:
        if self.rs_cam is None:
            self._last_rs_frame = None
            return None
        f = self.rs_cam.read_latest(max_age_s=0.5)
        self._last_rs_frame = f
        return f

    def _classify_color(self) -> Optional[ColorClassification]:
        if self._last_rs_frame is None or self._last_rs_frame.rgb is None:
            self._last_color_cls = None
            return None
        rs_cfg = getattr(self.cfg, "realsense", None)
        ccfg = getattr(rs_cfg, "color_stat", None) if rs_cfg else None
        kw = {}
        if ccfg is not None:
            for key in (
                "roi_w_ratio", "roi_h_ratio", "depth_min_m", "depth_max_m",
                "min_valid_depth_ratio",
                "white_min_ratio", "black_min_ratio",
                "blue_min_ratio", "other_max_ratio", "yellow_max_ratio",
                "night_white_min_ratio", "night_black_min_ratio",
                "night_blue_min_ratio",
                "night_other_max_ratio_blackwhite",
                "night_other_max_ratio_bluewhite",
                "night_blue_dominant_sum",
                "adaptive_white_v_base",
                "adaptive_v_ref",
            ):
                v = getattr(ccfg, key, None)
                if v is not None:
                    kw[key] = float(v)
            ne = getattr(ccfg, "night_enabled", None)
            if ne is not None:
                kw["night_enabled"] = bool(ne)
            ae = getattr(ccfg, "adaptive_enabled", None)
            if ae is not None:
                kw["adaptive_enabled"] = bool(ae)
        cls = classify_color_combo(
            self._last_rs_frame.rgb,
            depth_raw=self._last_rs_frame.depth_raw,
            depth_scale=self._last_rs_frame.depth_scale,
            require_depth=True,
            **kw,
        )
        self._last_color_cls = cls
        now = time.time()
        show = cls.label != ColorLabel.NONE or float(getattr(cls, "red_ratio", 0.0)) >= 0.05
        if now - self._last_rs_log_t > 0.5 and show:
            self.log.info(
                "[RS] %s conf=%.2f w=%.2f k=%.2f b=%.2f y=%.2f r=%.2f z=%.2fm",
                cls.label.value, cls.confidence,
                cls.white_ratio, cls.black_ratio, cls.blue_ratio,
                cls.yellow_ratio, getattr(cls, "red_ratio", 0.0), cls.depth_median_m,
            )
            self._last_rs_log_t = now
        return cls

    def _estimate_realsense_lane(self) -> Optional[LaneResult]:
        """Estimate coarse yellow-line error from the bottom D435i."""
        self._last_rs_lane = None
        if self._last_rs_frame is None or self._last_rs_frame.rgb is None:
            return None
        rs_cfg = getattr(self.cfg, "realsense", None)
        assist_cfg = getattr(rs_cfg, "lane_assist", None) if rs_cfg else None
        if assist_cfg is not None and not bool(getattr(assist_cfg, "enabled", True)):
            return None

        ranges_cfg = getattr(self.cfg.vision, "yellow_hsv_ranges", None)
        ranges_arg = None
        if ranges_cfg:
            ranges_arg = [(tuple(it[0]), tuple(it[1])) for it in ranges_cfg]

        def _get(name: str, default):
            return getattr(assist_cfg, name, default) if assist_cfg is not None else default

        lane = estimate_bottom_yellow_lane(
            self._last_rs_frame.rgb,
            depth_raw=self._last_rs_frame.depth_raw,
            depth_scale=self._last_rs_frame.depth_scale,
            depth_min_m=float(_get("depth_min_m", getattr(getattr(rs_cfg, "color_stat", None), "depth_min_m", 0.05))),
            depth_max_m=float(_get("depth_max_m", getattr(getattr(rs_cfg, "color_stat", None), "depth_max_m", 0.50))),
            roi_w_ratio=float(_get("roi_w_ratio", 0.80)),
            roi_h_ratio=float(_get("roi_h_ratio", 0.70)),
            min_depth_valid_ratio=float(_get("min_depth_valid_ratio", 0.10)),
            min_yellow_ratio=float(_get("min_yellow_ratio", 0.015)),
            min_pixels_per_strip=int(_get("min_pixels_per_strip", 30)),
            n_strips=int(_get("n_strips", 6)),
            error_sign=float(_get("error_sign", 1.0)),
            lower_hsv=tuple(self.cfg.vision.yellow_hsv_lower) if ranges_arg is None else None,
            upper_hsv=tuple(self.cfg.vision.yellow_hsv_upper) if ranges_arg is None else None,
            ranges=ranges_arg,
            open_kernel=int(self.cfg.vision.morph_open_kernel),
            close_kernel=int(self.cfg.vision.morph_close_kernel),
            adaptive=bool(getattr(self.cfg.vision, "yellow_adaptive_enabled", True)),
            adaptive_h_range=tuple(getattr(self.cfg.vision, "yellow_adaptive_h_range", [12, 48])),
            adaptive_s_min=int(getattr(self.cfg.vision, "yellow_adaptive_s_min", 12)),
            adaptive_v_min=int(getattr(self.cfg.vision, "yellow_adaptive_v_min", 45)),
            adaptive_lab_b_min=int(getattr(self.cfg.vision, "yellow_adaptive_lab_b_min", 138)),
            adaptive_lab_b_delta=int(getattr(self.cfg.vision, "yellow_adaptive_lab_b_delta", 8)),
            adaptive_rg_delta_min=int(getattr(self.cfg.vision, "yellow_adaptive_rg_delta_min", 12)),
        )
        self._last_rs_lane = lane
        return lane

    def _estimate_realsense_blue_ring(self) -> Optional[LaneResult]:
        """Estimate right-side blue-ring tracking from the bottom D435i."""
        self._last_rs_blue_ring_lane = None
        if self._last_rs_frame is None or self._last_rs_frame.rgb is None:
            return None
        rs_cfg = getattr(self.cfg, "realsense", None)
        assist_cfg = getattr(rs_cfg, "blue_ring_assist", None) if rs_cfg else None
        if assist_cfg is not None and not bool(getattr(assist_cfg, "enabled", True)):
            return None

        def _get(name: str, default):
            return getattr(assist_cfg, name, default) if assist_cfg is not None else default

        lane = estimate_bottom_blue_ring_lane(
            self._last_rs_frame.rgb,
            depth_raw=self._last_rs_frame.depth_raw,
            depth_scale=self._last_rs_frame.depth_scale,
            depth_min_m=float(_get("depth_min_m", 0.05)),
            depth_max_m=float(_get("depth_max_m", 0.55)),
            roi_w_ratio=float(_get("roi_w_ratio", 0.90)),
            roi_h_ratio=float(_get("roi_h_ratio", 0.80)),
            min_depth_valid_ratio=float(_get("min_depth_valid_ratio", 0.08)),
            min_blue_ratio=float(_get("min_blue_ratio", 0.010)),
            min_pixels_per_strip=int(_get("min_pixels_per_strip", 20)),
            n_strips=int(_get("n_strips", 6)),
            direction=str(getattr(getattr(self.cfg, "landmark", None), "obstacle_ring_direction", "right")),
        )
        self._last_rs_blue_ring_lane = lane
        now = time.time()
        if lane.found and now - self._last_blue_ring_log_t > 1.0:
            self.log.info("[ring] D435i blue ring conf=%.2f err=%+.3f", lane.confidence, lane.error)
            self._last_blue_ring_log_t = now
        return lane

    def _select_control_lane(self, front_lane: LaneResult) -> LaneResult:
        """Use front lane normally; use D435i only as a low-confidence fallback."""
        self._lane_control_source = "front"
        rs_cfg = getattr(self.cfg, "realsense", None)
        assist_cfg = getattr(rs_cfg, "lane_assist", None) if rs_cfg else None
        if assist_cfg is not None and not bool(getattr(assist_cfg, "enabled", True)):
            return front_lane

        min_front_conf = float(
            getattr(assist_cfg, "front_confidence_min", 0.45)
            if assist_cfg is not None else 0.45
        )
        min_rs_conf = float(
            getattr(assist_cfg, "bottom_confidence_min", 0.35)
            if assist_cfg is not None else 0.35
        )
        front_ok = front_lane.found and front_lane.confidence >= min_front_conf
        if front_ok:
            return front_lane

        rs_lane = self._last_rs_lane
        if rs_lane is not None and rs_lane.found and rs_lane.confidence >= min_rs_conf:
            self._lane_control_source = "realsense"
            self.last_lane_seen_at = time.time()
            if self._last_lane_control_source != "realsense":
                self.log.info(
                    "[lane] 前置黄线低置信/丢失, 切到底部 D435i 兜底: "
                    "front(found=%s conf=%.2f), bottom(conf=%.2f err=%+.3f)",
                    front_lane.found, front_lane.confidence,
                    rs_lane.confidence, rs_lane.error,
                )
            return rs_lane
        return front_lane

    def _ensure_color_triggers(self) -> None:
        if self._color_triggers:
            return
        rs_cfg = getattr(self.cfg, "realsense", None)
        if rs_cfg is None:
            return
        n_stable = int(getattr(rs_cfg, "n_stable_frames", 5))
        min_hits = int(getattr(rs_cfg, "min_hits_in_window", 4))
        cd_dump_s = float(getattr(rs_cfg, "cooldown_dump_s", 5.0))
        cd_obstacle_s = float(getattr(rs_cfg, "cooldown_obstacle_s", 5.0))
        cd_stair_s = float(getattr(rs_cfg, "cooldown_stair_s", 8.0))
        cd_dock_s = float(getattr(rs_cfg, "cooldown_dock_s", 8.0))
        self._color_triggers = {
            "obstacle": ColorTriggerState(
                name="obstacle", target_label=ColorLabel.BLUE_WHITE,
                trigger_depth_m=float(getattr(rs_cfg, "obstacle_trigger_depth_m", 0.40)),
                n_stable=n_stable, min_hits=min_hits,
                cooldown_after_time_s=cd_obstacle_s,
            ),
            "dump": ColorTriggerState(
                name="dump", target_label=ColorLabel.BLACK_WHITE,
                trigger_depth_m=float(getattr(rs_cfg, "dump_trigger_depth_m", 0.40)),
                n_stable=n_stable, min_hits=min_hits,
                cooldown_after_time_s=cd_dump_s,
            ),
            "stair": ColorTriggerState(
                name="stair", target_label=ColorLabel.BLUE_WHITE,
                trigger_depth_m=float(getattr(rs_cfg, "stair_trigger_depth_m", 0.40)),
                n_stable=n_stable, min_hits=min_hits,
                cooldown_after_time_s=cd_stair_s,
            ),
            "dock": ColorTriggerState(
                name="dock", target_label=ColorLabel.BLUE_WHITE,
                trigger_depth_m=float(getattr(rs_cfg, "dock_trigger_depth_m", 0.40)),
                n_stable=n_stable, min_hits=min_hits,
                cooldown_after_time_s=cd_dock_s,
            ),
        }

    def _save_trigger_snapshot(self, tag: str) -> None:
        """D435i trigger 命中时保存现场 jpg, 用于事后核对是不是真到位."""
        try:
            out_dir = Path("logs/triggers")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            ms = int((time.time() % 1) * 1000)
            cls = self._last_color_cls
            rs = self._last_rs_frame
            panels = []
            H = 360
            def fit(img):
                hh, ww = img.shape[:2]
                s = H / float(hh)
                return cv2.resize(img, (int(ww * s), H))
            if self._last_frame is not None:
                fish = self._last_frame.copy()
                cfg = self.cfg
                yt = int(fish.shape[0] * cfg.camera.roi_top_ratio)
                yb = int(fish.shape[0] * cfg.camera.roi_bottom_ratio)
                cv2.rectangle(fish, (0, yt), (fish.shape[1] - 1, yb), (0, 255, 0), 4)
                cv2.putText(fish, f"FISHEYE | tag={tag}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                panels.append(fit(fish))
            if rs is not None and rs.rgb is not None and cls is not None:
                rgb = rs.rgb.copy()
                x, y, w, h = cls.roi_bbox
                roi_color = (0, 255, 255) if cls.label == ColorLabel.BLACK_WHITE else \
                            (255, 200, 0) if cls.label == ColorLabel.BLUE_WHITE else \
                            (120, 120, 120)
                cv2.rectangle(rgb, (x, y), (x + w, y + h), roi_color, 3)
                cv2.putText(rgb, f"D435i RGB | {cls.label.value}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)
                cv2.putText(rgb, f"w={cls.white_ratio:.2f} k={cls.black_ratio:.2f} "
                                  f"b={cls.blue_ratio:.2f} y={cls.yellow_ratio:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(rgb, f"z_med={cls.depth_median_m:.2f}m valid={cls.depth_valid_ratio:.2f}",
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(rgb, f"rule: {cls.rule_explain[:60]}",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
                panels.append(fit(rgb))
                if rs.depth_raw is not None and rs.depth_raw.size > 0:
                    valid = rs.depth_raw[rs.depth_raw > 0]
                    if valid.size > 0:
                        lo = float(np.percentile(valid, 5)) * rs.depth_scale
                        hi = float(np.percentile(valid, 95)) * rs.depth_scale
                    else:
                        lo, hi = 0.2, 1.5
                    if hi - lo < 0.05:
                        hi = lo + 0.05
                    d_m = rs.depth_raw.astype(np.float32) * rs.depth_scale
                    d_norm = np.clip((d_m - lo) / (hi - lo), 0.0, 1.0)
                    d_u8 = (d_norm * 255).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
                    depth_vis[rs.depth_raw == 0] = 0
                    cv2.rectangle(depth_vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv2.putText(depth_vis, f"DEPTH {lo:.2f}~{hi:.2f}m", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    panels.append(fit(depth_vis))
            if not panels:
                return
            cat = np.hstack(panels)
            out_path = out_dir / f"{tag}-{ts}-{ms:03d}.jpg"
            cv2.imwrite(str(out_path), cat, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            self.log.info("[SNAP] trigger 现场存到 %s", out_path.name)
        except Exception as e:
            self.log.warning("[SNAP] save_trigger_snapshot 失败: %s", e)

    def _draw_rs_thumbnail(self, canvas) -> None:
        rs = self._last_rs_frame
        if rs is None or rs.rgb is None:
            return
        thumb_w = 240
        h_src, w_src = rs.rgb.shape[:2]
        thumb_h = int(h_src * (thumb_w / float(w_src)))
        thumb = cv2.resize(rs.rgb, (thumb_w, thumb_h))
        sx = thumb_w / float(w_src)
        sy = thumb_h / float(h_src)
        cls = self._last_color_cls
        if cls is not None:
            x, y, bw, bh = cls.roi_bbox
            x1 = int(x * sx); y1 = int(y * sy)
            x2 = int((x + bw) * sx); y2 = int((y + bh) * sy)
            roi_color = {
                ColorLabel.BLACK_WHITE: (0, 255, 255),
                ColorLabel.BLUE_WHITE: (255, 200, 0),
                ColorLabel.NONE: (120, 120, 120),
            }[cls.label]
            cv2.rectangle(thumb, (x1, y1), (x2, y2), roi_color, 2)
            label = f"{cls.label.value} z={cls.depth_median_m:.2f}m"
            cv2.putText(thumb, label, (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, roi_color, 1)
        dh, dw = canvas.shape[:2]
        x0 = dw - thumb_w - 10
        y0 = dh - thumb_h - 10
        if x0 < 0 or y0 < 0:
            return
        canvas[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb
        cv2.rectangle(canvas, (x0, y0), (x0 + thumb_w, y0 + thumb_h), (255, 0, 255), 2)
        cv2.putText(canvas, "D435i color", (x0 + 4, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        if cls is not None and cls.label != ColorLabel.NONE:
            txt = (f"w={cls.white_ratio:.2f} k={cls.black_ratio:.2f} "
                   f"b={cls.blue_ratio:.2f} y={cls.yellow_ratio:.2f}")
            cv2.putText(canvas, txt, (x0, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    def _build_rs_debug_frame(self) -> Optional["np.ndarray"]:
        rs = self._last_rs_frame
        if rs is None or rs.rgb is None:
            return None
        rgb = rs.rgb.copy()
        cls = self._last_color_cls
        if cls is not None:
            x, y, bw, bh = cls.roi_bbox
            roi_color = {
                ColorLabel.BLACK_WHITE: (0, 255, 255),
                ColorLabel.BLUE_WHITE: (255, 200, 0),
                ColorLabel.NONE: (120, 120, 120),
            }[cls.label]
            cv2.rectangle(rgb, (x, y), (x + bw, y + bh), roi_color, 2)
            cv2.putText(rgb, f"D435i {cls.label.value} z={cls.depth_median_m:.2f}m",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)
            cv2.putText(rgb, f"w={cls.white_ratio:.2f} k={cls.black_ratio:.2f} "
                              f"b={cls.blue_ratio:.2f} y={cls.yellow_ratio:.2f}",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(rgb, cls.rule_explain[:80],
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
        if self._last_rs_lane is not None:
            lane = self._last_rs_lane
            cv2.putText(rgb, f"yellow_lane found={lane.found} conf={lane.confidence:.2f} err={lane.error:+.3f}",
                        (10, rgb.shape[0] - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if self._last_rs_blue_ring_lane is not None:
            lane = self._last_rs_blue_ring_lane
            cv2.putText(rgb, f"blue_ring found={lane.found} conf={lane.confidence:.2f} err={lane.error:+.3f}",
                        (10, rgb.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)

        if rs.depth_raw is None or rs.depth_raw.size == 0:
            return rgb
        valid = rs.depth_raw[rs.depth_raw > 0]
        if valid.size > 0:
            lo = float(np.percentile(valid, 5)) * rs.depth_scale
            hi = float(np.percentile(valid, 95)) * rs.depth_scale
        else:
            lo, hi = 0.2, 1.5
        if hi - lo < 0.05:
            hi = lo + 0.05
        d_m = rs.depth_raw.astype(np.float32) * rs.depth_scale
        d_norm = np.clip((d_m - lo) / (hi - lo), 0.0, 1.0)
        d_u8 = (d_norm * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
        depth_vis[rs.depth_raw == 0] = 0
        cv2.putText(depth_vis, f"DEPTH {lo:.2f}~{hi:.2f}m",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if cls is not None:
            x, y, bw, bh = cls.roi_bbox
            cv2.rectangle(depth_vis, (x, y), (x + bw, y + bh), (255, 255, 255), 2)
        return np.hstack([rgb, depth_vis])

    def _step(self, frame):
        cfg = self.cfg
        self._read_realsense()
        self._classify_color()
        self._estimate_realsense_lane()
        self._estimate_realsense_blue_ring()
        self._ensure_color_triggers()
        roi, _ = crop_roi(
            frame,
            float(cfg.camera.roi_top_ratio),
            float(cfg.camera.roi_bottom_ratio),
        )
        # 优先用 yellow_hsv_ranges 多组并集 (覆盖不同光照).
        ranges_cfg = getattr(cfg.vision, "yellow_hsv_ranges", None)
        ranges_arg = None
        if ranges_cfg:
            ranges_arg = [(tuple(it[0]), tuple(it[1])) for it in ranges_cfg]
        mask_roi = yellow_mask(
            roi,
            lower_hsv=tuple(cfg.vision.yellow_hsv_lower) if ranges_arg is None else None,
            upper_hsv=tuple(cfg.vision.yellow_hsv_upper) if ranges_arg is None else None,
            ranges=ranges_arg,
            open_kernel=int(cfg.vision.morph_open_kernel),
            close_kernel=int(cfg.vision.morph_close_kernel),
            adaptive=bool(getattr(cfg.vision, "yellow_adaptive_enabled", True)),
            adaptive_h_range=tuple(getattr(cfg.vision, "yellow_adaptive_h_range", [12, 48])),
            adaptive_s_min=int(getattr(cfg.vision, "yellow_adaptive_s_min", 12)),
            adaptive_v_min=int(getattr(cfg.vision, "yellow_adaptive_v_min", 45)),
            adaptive_lab_b_min=int(getattr(cfg.vision, "yellow_adaptive_lab_b_min", 138)),
            adaptive_lab_b_delta=int(getattr(cfg.vision, "yellow_adaptive_lab_b_delta", 8)),
            adaptive_rg_delta_min=int(getattr(cfg.vision, "yellow_adaptive_rg_delta_min", 12)),
        )
        # 巡线在 ROI mask 上跑 (不再 warp, 否则黄道在 ROI 中上部会被 IPM 扔出画面).
        mask = mask_roi
        warped = self.ipm.warp(roi)  # 仅用于可视化
        front_lane = estimate_lane_error(mask, n_strips=8, min_pixels_per_strip=60)
        lane = self._select_control_lane(front_lane)
        if lane.found:
            self.last_lane_seen_at = time.time()

        # landmark 检测在原图 ROI 上跑 (颜色饱和 + 远距离地标早识别).
        # IPM warp 后的 mask 仅给巡线 (估横向误差) 用.
        landmarks = []
        dump = detect_dump_zone(
            mask_roi,
            roi,
            min_radius_ratio=float(cfg.landmark.dump_min_radius_ratio),
            min_area_ratio=float(cfg.landmark.dump_min_area_ratio),
            circularity_min=float(cfg.landmark.dump_circularity_min),
        )
        if dump is not None:
            landmarks.append(dump)

        fork = detect_fork(
            mask_roi,
            upper_band_ratio=float(cfg.landmark.fork_split_y_ratio),
            min_branch_area_ratio=float(cfg.landmark.fork_branch_min_area_ratio),
        )
        if fork is not None:
            landmarks.append(fork)

        blue_ring = None
        if bool(getattr(cfg.landmark, "obstacle_ring_enabled", False)):
            blue_ring = detect_blue_obstacle_ring(
                roi,
                min_area_ratio=float(getattr(cfg.landmark, "blue_ring_min_area_ratio", 0.012)),
                min_circularity=float(getattr(cfg.landmark, "blue_ring_min_circularity", 0.45)),
                inner_white_min=float(getattr(cfg.landmark, "blue_ring_inner_white_min", 0.25)),
                inner_black_min=float(getattr(cfg.landmark, "blue_ring_inner_black_min", 0.01)),
            )
            if blue_ring is not None:
                landmarks.append(blue_ring)

        # 蓝色矩形检测: 障碍区标记/台阶区/充电区视觉特征一样, 由 FSM 顺序区分语义.
        # 起点未 dump 前首次见蓝色矩形环 -> obstacle marker; dump 后第一次见 -> stair; stair 后第二次见 -> dock.
        from src.vision.landmark import detect_blue_rect
        blue_rect = detect_blue_rect(
            roi,
            min_area_ratio=float(getattr(cfg.landmark, "blue_rect_min_area_ratio", 0.02)),
        )
        obstacle = None
        if (bool(getattr(cfg.landmark, "obstacle_enabled", False))
                and not self.fsm.flags.obstacle_avoided
                and not self.fsm.flags.dumped):
            obstacle = blue_rect
            stair = None
            dock = None
        elif not self.fsm.flags.stair_climbed:
            stair = blue_rect
            dock = None
        else:
            stair = None
            dock = blue_rect
        if blue_rect is not None:
            from src.vision.landmark import LandmarkType
            blue_rect.type = LandmarkType.DOCK_AREA if self.fsm.flags.stair_climbed else LandmarkType.STAIR
            landmarks.append(blue_rect)

        vx, vyaw = self._dispatch(
            lane, dump, fork, stair, dock,
            obstacle=obstacle, blue_ring=blue_ring,
        )

        # debug 画布用 ROI 而不是 warped, 让 landmark bbox 坐标和画布对齐.
        # mask_roi 同样在 ROI 坐标系, lane.debug_centroid 用 warped 坐标系所以不画.
        debug = overlay(
            roi,
            mask=mask_roi,
            lane=None,
            landmarks=landmarks,
            state_text=self.fsm.state.value,
            error=front_lane.error if front_lane.found else 0.0,
            vyaw=vyaw,
            fps=self._fps,
        )
        # 把巡线 debug_centroid 单独画在右上角的 IPM 缩略图上 (可选)
        if front_lane.found and front_lane.debug_centroid is not None:
            mini_h = 200
            mini_w = 200
            mini = cv2.resize(warped, (mini_w, mini_h))
            mh, mw = mini.shape[:2]
            cx_w = int(front_lane.debug_centroid[0] * mw / mask.shape[1])
            cy_w = int(front_lane.debug_centroid[1] * mh / mask.shape[0])
            cv2.circle(mini, (cx_w, cy_w), 6, (0, 255, 0), -1)
            cv2.line(mini, (mw // 2, 0), (mw // 2, mh), (255, 255, 255), 1)
            # 贴到 debug 右上角
            dh, dw = debug.shape[:2]
            x_offset = dw - mw - 10
            y_offset = 35
            if x_offset >= 0 and y_offset + mh <= dh:
                debug[y_offset:y_offset + mh, x_offset:x_offset + mw] = mini
                cv2.rectangle(debug, (x_offset, y_offset), (x_offset + mw, y_offset + mh), (255, 255, 0), 2)

        if self._last_lane_found != front_lane.found:
            n_valid = sum(1 for c in front_lane.centerline_x if c >= 0)
            yellow_px = int((mask > 0).sum())
            self.log.info(
                "[vision] lane.found=%s, valid_strips=%d/8, yellow_px=%d (mask shape=%s), conf=%.2f, err=%+.3f",
                front_lane.found, n_valid, yellow_px, mask.shape, front_lane.confidence, front_lane.error,
            )
            self._last_lane_found = front_lane.found

        if self._last_lane_control_source != self._lane_control_source:
            self.log.info("[lane] control source = %s", self._lane_control_source)
            self._last_lane_control_source = self._lane_control_source

        front_debug = debug.copy()

        if self._last_rs_frame is not None and self._last_rs_frame.rgb is not None:
            try:
                self._draw_rs_thumbnail(debug)
            except Exception:
                pass

        if self.mjpeg is not None:
            try:
                self.mjpeg.push_frame(front_debug, stream="front")
                rs_debug = self._build_rs_debug_frame()
                if rs_debug is not None:
                    self.mjpeg.push_frame(rs_debug, stream="realsense")
                self.mjpeg.push_frame(debug, stream="debug")
            except Exception:
                pass

        self._last_frame = frame
        self._last_warped = warped
        self._last_mask = mask
        self._last_debug = debug
        return vx, vyaw, debug

    def _dispatch(self, lane: LaneResult, dump, fork, stair, dock, obstacle=None, blue_ring=None):
        cfg = self.cfg
        f = cfg.control.forward_speed
        st = self.fsm.state

        vx = 0.0
        vyaw = 0.0
        self._cmd_vy = 0.0

        if st == State.FOLLOW_LANE:
            rs_cfg = getattr(cfg, "realsense", None)
            rs_enabled = self.rs_cam is not None
            cls = self._last_color_cls
            color_active = rs_enabled and cls is not None and bool(self._color_triggers)

            # ---- stair 完成后: 强制终点直行锁定 ----
            # 这段路容易看到右侧黄线/场地图案。台阶完成后不再用黄线控制转向,
            # D435i/鱼眼只负责判断终点是否到达, 运动命令始终 vx>0, vyaw=0。
            # ⚠ 只看 stair_climbed, 不要求 dumped: 万一 dump 当场漏触发,
            # 锁定也必须激活, 否则狗会继续巡线跟着右侧黄线跑 (静默失效).
            if self.fsm.flags.stair_climbed:
                now = time.time()
                if now < self._post_stair_straight_ready_at:
                    if not self._post_stair_settle_logged:
                        self.log.info(
                            "[FSM] 台阶后停稳 %.1fs, 清掉步态/转向惯性后再终点直行",
                            self._post_stair_straight_ready_at - now,
                        )
                        self._post_stair_settle_logged = True
                    self.last_lane_seen_at = now
                    return 0.0, 0.0

                cruise_speed = float(getattr(rs_cfg, "dock_blind_walk_speed", 0.25))
                if self._blind_walk_started_at <= 0.0:
                    self._blind_walk_started_at = time.time()
                    self.log.info(
                        "[FSM] 进入终点直行锁定 (vx=%.2fm/s vyaw=0, "
                        "不再用黄线转向, 等终点触发)",
                        cruise_speed,
                    )
                vx = cruise_speed
                vyaw = 0.0
                self.last_lane_seen_at = time.time()  # 抑制 SEARCH_LANE

                if color_active:
                    trig = self._color_triggers["dock"]
                    if trig.update(cls):
                        trig.disarm()
                        self._save_trigger_snapshot("dock")
                        self._dock_post_trigger_active = True
                        self.log.info(
                            "[FSM] D435i BLUE_WHITE 2nd @ z=%.2fm -> APPROACH_DOCK",
                            cls.depth_median_m,
                        )
                        self.fsm.transit(State.APPROACH_DOCK)
                        return 0.0, 0.0
                else:
                    dock_close = self._landmark_close_enough(
                        dock,
                        float(getattr(cfg.landmark, "dock_proximity_threshold", 0.82)),
                    )
                    if self.fsm.vote_dock(
                        dock_close,
                        int(cfg.landmark.dock_consecutive_frames),
                    ):
                        self._save_trigger_snapshot("dock")
                        self._dock_post_trigger_active = False
                        self.log.info("[FSM] 鱼眼终点触发 -> APPROACH_DOCK (继续直行锁定)")
                        self.fsm.transit(State.APPROACH_DOCK)
                        return 0.0, 0.0
                return vx, vyaw

            vx, vyaw = self._follow(lane, float(f.follow))

            # ---- 起步保护 (前 N 秒禁用 D435i 触发) ----
            startup_protect_s = float(getattr(rs_cfg, "startup_protect_s", 3.0)) if rs_cfg else 0.0
            elapsed_since_start = time.time() - self.start_time
            in_startup_protect = elapsed_since_start < startup_protect_s
            if in_startup_protect and not getattr(self, "_startup_protect_logged", False):
                self.log.info(
                    "[FSM] 起步保护期 (elapsed=%.1fs < %.1fs), D435i 触发禁用",
                    elapsed_since_start, startup_protect_s,
                )
                self._startup_protect_logged = True
            if (not in_startup_protect and getattr(self, "_startup_protect_logged", False)
                    and not getattr(self, "_startup_protect_exit_logged", False)):
                self.log.info("[FSM] 起步保护期结束 (elapsed=%.1fs)", elapsed_since_start)
                self._startup_protect_exit_logged = True

            if color_active and not in_startup_protect:
                # ---- D435i 主路径 ----
                # (0) 起点障碍区: 启动保护结束后, 看到蓝色入口才进入脚本避障.
                if (bool(getattr(cfg.landmark, "obstacle_enabled", False))
                        and not self.fsm.flags.obstacle_avoided
                        and not self.fsm.flags.blue_ring_done
                        and not self.fsm.flags.dumped):
                    if not self._obstacle_entry_allowed():
                        self.fsm.vote_obstacle(
                            False,
                            int(getattr(cfg.landmark, "obstacle_consecutive_frames", 8)),
                        )
                    elif self._blue_ring_hit(blue_ring):
                        if self.fsm.vote_obstacle(
                            True,
                            int(getattr(cfg.landmark, "obstacle_consecutive_frames", 8)),
                        ):
                            self._save_trigger_snapshot("blue_ring")
                            self.log.info("[FSM] blue obstacle ring -> FOLLOW_OBSTACLE_RING")
                            self.fsm.transit(State.FOLLOW_OBSTACLE_RING)
                            return 0.0, 0.0
                    elif not bool(getattr(cfg.landmark, "obstacle_ring_enabled", True)):
                        trig = self._color_triggers["obstacle"]
                        if trig.update(cls):
                            trig.disarm()
                            self._save_trigger_snapshot("obstacle")
                            self.log.info(
                                "[FSM] D435i BLUE_WHITE obstacle @ z=%.2fm -> AVOID_OBSTACLE",
                                cls.depth_median_m,
                            )
                            self.fsm.transit(State.AVOID_OBSTACLE)
                            return 0.0, 0.0
                    else:
                        self.fsm.vote_obstacle(
                            False,
                            int(getattr(cfg.landmark, "obstacle_consecutive_frames", 8)),
                        )

                # (1) dump: 比赛现场以底部 D435i 黑白区为准, 命中后直接卸料.
                # 前置鱼眼黄色圆环只作为 D435i 不可用时的兜底.
                if not self.fsm.flags.dumped:
                    dump_allowed = self._dump_entry_allowed()
                    trig = self._color_triggers["dump"]
                    if dump_allowed and trig.update(cls):
                        trig.disarm()
                        self._save_trigger_snapshot("dump")
                        self._dump_ring_confirmed = True
                        self._dump_post_trigger_active = False
                        self.log.info(
                            "[FSM] D435i BLACK_WHITE dump @ z=%.2fm -> DUMP_ACTION",
                            cls.depth_median_m,
                        )
                        self.fsm.transit(State.DUMP_ACTION)
                        return 0.0, 0.0

                    fisheye_hit = False
                    if dump_allowed and bool(getattr(cfg.landmark, "dump_fisheye_backup_enabled", False)):
                        dump_close = self._landmark_close_enough(
                            dump,
                            float(getattr(cfg.landmark, "dump_proximity_threshold", 0.58)),
                        )
                        fisheye_hit = self.fsm.vote_dump(
                            dump_close, int(cfg.landmark.dump_consecutive_frames),
                        )
                    else:
                        self.fsm.vote_dump(False, int(cfg.landmark.dump_consecutive_frames))

                    if fisheye_hit:
                        trig.disarm()
                        self._save_trigger_snapshot("dump")
                        self._dump_ring_confirmed = False
                        self.log.info(
                            "[FSM] 鱼眼 yellow dump ring -> FOLLOW_DUMP_RING",
                        )
                        self.fsm.transit(State.FOLLOW_DUMP_RING)
                        return 0.0, 0.0

                # (2) stair: dump 完成后, D435i 白蓝区直接切 ClassicWalk 爬楼状态.
                if (self.fsm.flags.dumped
                        and not self.fsm.flags.stair_climbed):
                    trig = self._color_triggers["stair"]
                    if trig.update(cls):
                        trig.disarm()
                        self._save_trigger_snapshot("stair")
                        self.log.info(
                            "[FSM] D435i BLUE_WHITE stair @ z=%.2fm -> APPROACH_STAIR",
                            cls.depth_median_m,
                        )
                        self.fsm.transit(State.APPROACH_STAIR)
                        return 0.0, 0.0

                # (3) fork (dumped 之后, stair 之前, 鱼眼检测)
                if (self.fsm.flags.dumped
                        and not self.fsm.flags.fork_chosen
                        and not self.fsm.flags.stair_climbed):
                    if self.fsm.vote_fork(
                        fork is not None,
                        int(cfg.landmark.fork_consecutive_frames),
                    ):
                        if fork is not None and fork.extra:
                            image_w = int(fork.extra.get("image_w", self._roi_width()))
                            self.fork_choice = choose_shortest_fork_branch(
                                float(fork.extra["left_x"]),
                                float(fork.extra["right_x"]),
                                image_w,
                            )
                            self.log.info("[FORK] 选 %s 支", self.fork_choice)
                        self.fsm.flags.fork_chosen = True
                        self.fsm.transit(State.CHOOSE_FORK)
                        return 0.0, 0.0
            elif not color_active:
                # ---- 兜底: 鱼眼 + vote (D435i 不可用时) ----
                if (bool(getattr(cfg.landmark, "obstacle_enabled", False))
                        and not self.fsm.flags.obstacle_avoided
                        and not self.fsm.flags.blue_ring_done
                        and not self.fsm.flags.dumped):
                    if not self._obstacle_entry_allowed():
                        self.fsm.vote_obstacle(
                            False,
                            int(getattr(cfg.landmark, "obstacle_consecutive_frames", 8)),
                        )
                    elif bool(getattr(cfg.landmark, "obstacle_ring_enabled", True)):
                        if self.fsm.vote_obstacle(
                            self._blue_ring_hit(blue_ring),
                            int(getattr(cfg.landmark, "obstacle_consecutive_frames", 8)),
                        ):
                            self.log.info("[OBSTACLE] 检测到蓝色圆环, 进入右侧沿环绕行")
                            self.fsm.transit(State.FOLLOW_OBSTACLE_RING)
                            return 0.0, 0.0
                    elif self.fsm.vote_obstacle(
                        obstacle is not None,
                        int(getattr(cfg.landmark, "obstacle_consecutive_frames", 8)),
                    ):
                        self.log.info("[OBSTACLE] 检测到起点障碍区蓝色矩形标记, 右绕避障")
                        self.fsm.transit(State.AVOID_OBSTACLE)
                        return 0.0, 0.0
                elif not self.fsm.flags.dumped and self._dump_entry_allowed():
                    dump_close = self._landmark_close_enough(
                        dump,
                        float(getattr(cfg.landmark, "dump_proximity_threshold", 0.58)),
                    )
                    if self.fsm.vote_dump(dump_close, int(cfg.landmark.dump_consecutive_frames)):
                        self._dump_ring_confirmed = False
                        self.fsm.transit(State.FOLLOW_DUMP_RING)
                        return 0.0, 0.0
                elif not self.fsm.flags.fork_chosen:
                    if self.fsm.vote_fork(fork is not None, int(cfg.landmark.fork_consecutive_frames)):
                        if fork is not None and fork.extra:
                            image_w = int(fork.extra.get("image_w", self._roi_width()))
                            self.fork_choice = choose_shortest_fork_branch(
                                float(fork.extra["left_x"]),
                                float(fork.extra["right_x"]),
                                image_w,
                            )
                            self.log.info("[FORK] 选 %s 支", self.fork_choice)
                        self.fsm.flags.fork_chosen = True
                        self.fsm.transit(State.CHOOSE_FORK)
                elif not self.fsm.flags.stair_climbed:
                    stair_close = self._landmark_close_enough(
                        stair,
                        float(getattr(cfg.landmark, "stair_proximity_threshold", 0.72)),
                    )
                    if self.fsm.vote_stair(stair_close, int(cfg.landmark.stair_consecutive_frames)):
                        self.fsm.transit(State.APPROACH_STAIR)
                else:
                    dock_close = self._landmark_close_enough(
                        dock,
                        float(getattr(cfg.landmark, "dock_proximity_threshold", 0.82)),
                    )
                    if self.fsm.vote_dock(dock_close, int(cfg.landmark.dock_consecutive_frames)):
                        self.fsm.transit(State.APPROACH_DOCK)

        elif st == State.FOLLOW_DUMP_RING:
            return self._follow_dump_ring(lane, dump)

        elif st == State.APPROACH_DUMP:
            entered = self._ensure_approach_distance_state(st)
            rs_cfg = getattr(cfg, "realsense", None)
            if entered and self._dump_post_trigger_active:
                target_preview = float(getattr(
                    rs_cfg,
                    "dump_post_trigger_distance_m",
                    getattr(cfg.landmark, "dump_walk_distance_m", 1.2),
                ))
                self.log.info(
                    "[DUMP] D435i 已确认黑白倾倒区, 继续沿黄线 %.2fm 后再卸料",
                    target_preview,
                )

            vx, vyaw = self._follow_or_landmark(lane, float(f.dump_approach), dump)
            dist = self._approach_distance_update(vx)
            if self._dump_post_trigger_active:
                target_dist = float(getattr(
                    rs_cfg,
                    "dump_post_trigger_distance_m",
                    getattr(cfg.landmark, "dump_walk_distance_m", 1.2),
                ))
                timeout_s = float(getattr(rs_cfg, "dump_post_trigger_timeout_s", 8.0))
            else:
                target_dist = float(getattr(cfg.landmark, "dump_walk_distance_m", 1.2))
                timeout_s = 15.0

            if dist >= target_dist:
                self.log.info("[DUMP] 路程 %.2fm >= %.2fm, 到达倾倒动作点, 卸料", dist, target_dist)
                self._dump_post_trigger_active = False
                self.fsm.transit(State.DUMP_ACTION)
                return 0.0, 0.0
            if self.fsm.time_in_state() > timeout_s:
                self.log.warning("APPROACH_DUMP 超时 %.1fs, 强制卸料", timeout_s)
                self._dump_post_trigger_active = False
                self.fsm.transit(State.DUMP_ACTION)
                return 0.0, 0.0

        elif st == State.DUMP_ACTION:
            if self.robot is not None:
                self.log.info("[DUMP] 开始卸料动作")
                execute_dump_action(
                    self.robot,
                    method=str(getattr(cfg.dump_action, "method", "stretch")),
                    roll_rad=float(cfg.dump_action.roll_rad),
                    hold_sec=float(cfg.dump_action.hold_sec),
                    spam_hz=int(cfg.dump_action.spam_hz),
                    repeat=int(getattr(cfg.dump_action, "repeat", 2)),
                )
                self.log.info("[DUMP] 卸料完成, 站稳 1s 恢复姿态")
                self.robot.stop_move()
                time.sleep(1.0)
            self.fsm.flags.dumped = True
            self._post_dump_recover_until_t = time.time() + 2.0
            self._dump_done_at = time.time()
            self.last_lane_seen_at = time.time()
            self.fsm.transit(State.FOLLOW_LANE)
            return 0.0, 0.0

        elif st == State.CHOOSE_FORK:
            bias = -0.45 if self.fork_choice == "left" else 0.45
            vx = float(f.fork)
            vyaw = self._lateral_to_vyaw(bias)
            if self.fsm.time_in_state() > 1.5:
                self.fsm.transit(State.FOLLOW_LANE)

        elif st == State.FOLLOW_OBSTACLE_RING:
            mode = str(getattr(cfg.landmark, "obstacle_ring_mode", "scripted"))
            if mode == "scripted":
                return self._obstacle_scripted_step(lane)
            vx, vyaw = self._follow_blue_ring(blue_ring, lane)
            return vx, vyaw

        elif st == State.AVOID_OBSTACLE:
            t = self.fsm.time_in_state()
            avoid_s = float(getattr(cfg.landmark, "obstacle_avoid_duration_sec", 2.2))
            recover_s = float(getattr(cfg.landmark, "obstacle_recover_duration_sec", 1.2))
            vx = float(getattr(f, "obstacle", f.follow))
            if t < avoid_s:
                vyaw = float(getattr(cfg.landmark, "obstacle_right_yaw", -0.55))
            elif t < avoid_s + recover_s:
                vyaw = float(getattr(cfg.landmark, "obstacle_recover_yaw", 0.35))
            else:
                self.fsm.flags.obstacle_avoided = True
                self.last_lane_seen_at = time.time()
                self.log.info("[OBSTACLE] 右绕完成, 恢复巡线")
                self.fsm.transit(State.FOLLOW_LANE)
                return 0.0, 0.0

        elif st == State.APPROACH_STAIR:
            # D435i 触发已经把狗放在台阶前 ~0.4m, 立刻停步切步态; 走够路程退出.
            if self._ensure_approach_distance_state(st):
                self._stair_mode_entered = False

            target_through = float(getattr(cfg.landmark, "stair_through_distance_m", 2.5))

            if not self._stair_mode_entered:
                self.log.info("[STAIR] 进入 APPROACH_STAIR, 立即停步切 ClassicWalk")
                if self.robot is not None:
                    self.robot.stop_move()
                    time.sleep(0.3)
                    enter_stair_mode(self.robot)
                self._stair_mode_entered = True
                self._approach_distance_reset()
                return 0.0, 0.0
            else:
                # ⚠ 台阶穿越期间不用黄线转向:
                # 狗在台阶上时视野里可能出现右侧黄线/场地图案, 用 lane.error 转向
                # 会把航向带歪 (省赛实际翻车点), 且台阶上激进转向有摔落风险.
                # 穿越台阶就是纯直行, vyaw 强制压到极小.
                vx = float(f.stair)
                stair_yaw_cap = float(getattr(
                    getattr(cfg, "realsense", None),
                    "stair_cross_max_yaw", 0.0,
                ))
                if stair_yaw_cap > 0.0 and lane.found:
                    raw_vyaw = self._lateral_to_vyaw(lane.error)
                    vyaw = max(-stair_yaw_cap, min(stair_yaw_cap, raw_vyaw))
                else:
                    vyaw = 0.0
                self.last_lane_seen_at = time.time()  # 台阶上不触发 SEARCH_LANE
                dist = self._approach_distance_update(vx)
                if dist >= target_through:
                    self.log.info("[STAIR] 穿过台阶 (路程 %.2fm)", dist)
                    if self.robot is not None:
                        exit_stair_mode(self.robot)
                    self.fsm.flags.stair_climbed = True
                    self._stair_mode_entered = False
                    settle_s = float(getattr(
                        getattr(cfg, "realsense", None),
                        "dock_post_stair_settle_sec",
                        0.8,
                    ))
                    self._post_stair_straight_ready_at = time.time() + max(0.0, settle_s)
                    self._post_stair_settle_logged = False
                    self._blind_walk_started_at = 0.0
                    self.fsm.transit(State.FOLLOW_LANE)
                    return 0.0, 0.0

            if self.fsm.time_in_state() > 30.0:
                self.log.warning("APPROACH_STAIR 超时 30s, 强制退出")
                if self.robot is not None and self._stair_mode_entered:
                    exit_stair_mode(self.robot)
                self.fsm.flags.stair_climbed = True
                self._stair_mode_entered = False
                settle_s = float(getattr(
                    getattr(cfg, "realsense", None),
                    "dock_post_stair_settle_sec",
                    0.8,
                ))
                self._post_stair_straight_ready_at = time.time() + max(0.0, settle_s)
                self._post_stair_settle_logged = False
                self._blind_walk_started_at = 0.0
                self.fsm.transit(State.FOLLOW_LANE)
                return 0.0, 0.0

        elif st == State.CLIMB_STAIR:
            # 已废弃: 改用 APPROACH_STAIR 单状态包含步态切换+穿越判定.
            # 万一旧逻辑误进入这个状态, 直接切回 FOLLOW_LANE 兜底.
            self.log.warning("CLIMB_STAIR 已废弃, 切回 FOLLOW_LANE")
            self.fsm.flags.stair_climbed = True
            settle_s = float(getattr(
                getattr(cfg, "realsense", None),
                "dock_post_stair_settle_sec",
                0.8,
            ))
            self._post_stair_straight_ready_at = time.time() + max(0.0, settle_s)
            self._post_stair_settle_logged = False
            self._blind_walk_started_at = 0.0
            self.fsm.transit(State.FOLLOW_LANE)
            return 0.0, 0.0

        elif st == State.APPROACH_DOCK:
            entered = self._ensure_approach_distance_state(st)
            rs_cfg = getattr(cfg, "realsense", None)

            if self._dock_post_trigger_active:
                target_dist = float(getattr(
                    rs_cfg,
                    "dock_post_trigger_distance_m",
                    getattr(cfg.landmark, "dock_walk_distance_m", 0.5),
                ))
                timeout_s = float(getattr(rs_cfg, "dock_post_trigger_timeout_s", 5.0))
                vx = float(getattr(rs_cfg, "dock_blind_walk_speed", f.dock))
                vyaw = 0.0
                self.last_lane_seen_at = time.time()
                if entered:
                    self.log.info(
                        "[DOCK] D435i 已确认终点蓝白区, 继续直行 %.2fm 后停车",
                        target_dist,
                    )
                dist = self._approach_distance_update(vx)
                if dist >= target_dist:
                    self.log.info("[DOCK] 终点后移 %.2fm >= %.2fm, 停车", dist, target_dist)
                    self._dock_post_trigger_active = False
                    self.fsm.transit(State.DOCK)
                    return 0.0, 0.0
                if self.fsm.time_in_state() > timeout_s:
                    self.log.info("[DOCK] 终点后移超时 %.1fs, 停车", timeout_s)
                    self._dock_post_trigger_active = False
                    self.fsm.transit(State.DOCK)
                    return 0.0, 0.0
                return vx, vyaw

            if self.fsm.flags.stair_climbed:
                vx = float(getattr(rs_cfg, "dock_blind_walk_speed", f.dock)) if rs_cfg else float(f.dock)
                vyaw = 0.0
                self.last_lane_seen_at = time.time()
            else:
                vx, vyaw = self._follow_or_landmark(lane, float(f.dock), dock)
            dist = self._approach_distance_update(vx)
            target_dist = float(getattr(cfg.landmark, "dock_walk_distance_m", 1.0))

            if dist >= target_dist:
                self.log.info("[DOCK] 路程 %.2fm >= %.2fm, 到达充电区中心, 停车", dist, target_dist)
                self.fsm.transit(State.DOCK)
                return 0.0, 0.0
            if self.fsm.time_in_state() > 12.0:
                self.log.info("[DOCK] 超时 12s, 停车")
                self.fsm.transit(State.DOCK)
                return 0.0, 0.0

        elif st == State.DOCK:
            if self.robot is not None:
                final_dock(self.robot)
            self.fsm.transit(State.DONE)
            return 0.0, 0.0

        elif st == State.SEARCH_LANE:
            vx = 0.0
            vyaw = float(cfg.control.lost_lane_search_yaw)
            if lane.found:
                self.fsm.transit(State.FOLLOW_LANE)
            elif self.fsm.time_in_state() > float(getattr(cfg.control, "search_lane_max_sec", 8.0)):
                self.log.error("SEARCH_LANE 超时仍未找到黄道, EMERGENCY_STOP")
                self.fsm.transit(State.EMERGENCY_STOP)

        has_active_landmark = (
            (st == State.FOLLOW_DUMP_RING and dump is not None)
            or (st == State.APPROACH_DUMP and dump is not None)
            or (st == State.APPROACH_STAIR and stair is not None)
            or (st == State.APPROACH_DOCK and dock is not None)
        )
        if not lane.found and not has_active_landmark and st in (
            State.FOLLOW_LANE, State.FOLLOW_DUMP_RING,
            State.APPROACH_DUMP, State.APPROACH_STAIR, State.APPROACH_DOCK
        ):
            if (time.time() - self.last_lane_seen_at) > float(cfg.control.lost_lane_timeout_sec):
                self.log.warning("丢线超时, 进入 SEARCH_LANE")
                self.fsm.transit(State.SEARCH_LANE)

        return vx, vyaw

    def _obstacle_entry_allowed(self) -> bool:
        """启动保护窗口内不允许进入避障脚本.

        这同时约束 D435i 触发和前置鱼眼兜底触发, 防止刚放到起点时把
        起点/附近蓝色区域误当成障碍区入口。
        """
        rs_cfg = getattr(self.cfg, "realsense", None)
        protect_s = float(getattr(rs_cfg, "startup_protect_s", 3.0)) if rs_cfg else 3.0
        return (time.time() - self.start_time) >= protect_s

    def _dump_entry_allowed(self) -> bool:
        """是否允许进入倾倒区。

        比赛顺序是先避障、再倾倒。D435i 看到黑白区域可能来自起点白块、
        展板/阴影或障碍附近地面, 所以默认要求障碍区已完成后才允许倾倒。
        """
        obstacle_enabled = bool(getattr(self.cfg.landmark, "obstacle_enabled", False))
        require_obstacle = bool(getattr(self.cfg.landmark, "dump_require_obstacle_done", True))
        if not obstacle_enabled or not require_obstacle:
            return True
        return bool(self.fsm.flags.obstacle_avoided or self.fsm.flags.blue_ring_done)

    def _obstacle_red_hit(self, cls) -> bool:
        """D435i 是否稳定看到红色 (按帧投票)."""
        cfg = self.cfg
        rs_cfg = getattr(cfg, "realsense", None)
        need = int(getattr(cfg.landmark, "obstacle_seq_red_consecutive", 3))
        rmin = float(getattr(rs_cfg, "obstacle_red_min_ratio", 0.15)) if rs_cfg else 0.15
        depth_max = float(getattr(rs_cfg, "obstacle_red_trigger_depth_m", 0.60)) if rs_cfg else 0.60
        hit = False
        if cls is not None and float(getattr(cls, "red_ratio", 0.0)) >= rmin:
            z = float(getattr(cls, "depth_median_m", 0.0))
            if z <= 0.0 or z <= depth_max:
                hit = True
        self._obstacle_red_count = self._obstacle_red_count + 1 if hit else 0
        return self._obstacle_red_count >= need

    def _exit_obstacle_seq(self, lane: LaneResult):
        self.fsm.flags.obstacle_avoided = True
        self.fsm.flags.blue_ring_done = True
        self.last_lane_seen_at = time.time()
        self._obstacle_seq_active = False
        self._obstacle_phase = None
        self.log.info("[OBSTACLE_SEQ] 避障区完成, 恢复 FOLLOW_LANE 用鱼眼循黄线")
        self.fsm.transit(State.FOLLOW_LANE)
        return 0.0, 0.0

    def _obstacle_scripted_step(self, lane: LaneResult):
        """避障区脚本化动作 (蓝色进入后, 内部可纯定时):
        识别到蓝色进入 ->
        right90:   右转 90°            -> straight1
        straight1: 直行, 定时/见红 -> left90a
        left90a:   左转 90°            -> straight2
        straight2: 直行, 定时/见红 -> left90b
        left90b:   左转 90°            -> straight3
        straight3: 直行                -> 可选 strafe_right
        strafe_right: 横向右移          -> right_final
        right_final: 右转 90°          -> 可选 final_forward -> 退出循黄线
        转弯角度靠 (turn_yaw × 时间) 近似, 现场用 obstacle_seq_turn90_sec 标定.
        """
        cfg = self.cfg
        cls = self._last_color_cls
        self._cmd_vy = 0.0
        turn_yaw = float(getattr(cfg.landmark, "obstacle_seq_turn_yaw", 0.6))
        turn_vx = float(getattr(cfg.landmark, "obstacle_seq_turn_vx", 0.0))
        straight_vx = float(getattr(cfg.landmark, "obstacle_seq_straight_vx", 0.22))
        turn90_sec = float(getattr(cfg.landmark, "obstacle_seq_turn90_sec", 2.6))
        right90_sec = float(getattr(cfg.landmark, "obstacle_seq_right90_sec", turn90_sec))
        left90a_sec = float(getattr(cfg.landmark, "obstacle_seq_left90a_sec", turn90_sec))
        left90b_sec = float(getattr(cfg.landmark, "obstacle_seq_left90b_sec", turn90_sec))
        right_final_sec = float(getattr(cfg.landmark, "obstacle_seq_right_final_sec", turn90_sec))
        straight1_sec = float(getattr(cfg.landmark, "obstacle_seq_straight1_sec", 2.2))
        straight2_sec = float(getattr(cfg.landmark, "obstacle_seq_straight2_sec", 3.0))
        straight3_sec = float(getattr(cfg.landmark, "obstacle_seq_straight3_sec", 3.0))
        strafe_sec = float(getattr(cfg.landmark, "obstacle_seq_strafe_after_straight3_sec", 0.0))
        strafe_vy = float(getattr(cfg.landmark, "obstacle_seq_strafe_vy", 0.0))
        final_forward_sec = float(getattr(cfg.landmark, "obstacle_seq_final_forward_sec", 0.0))
        use_red = bool(getattr(cfg.landmark, "obstacle_seq_use_red", True))
        grace = float(getattr(cfg.landmark, "obstacle_seq_straight_grace_sec", 0.6))
        phase_max_sec = float(getattr(cfg.landmark, "obstacle_seq_phase_max_sec", 8.0))
        max_sec = float(getattr(cfg.landmark, "obstacle_seq_max_sec", 40.0))

        now = time.time()
        if not self._obstacle_seq_active:
            self._obstacle_seq_active = True
            self._obstacle_phase = "right90"
            self._obstacle_phase_t = now
            self._obstacle_red_count = 0
            mode = "red-trigger" if use_red else "timed"
            self.log.info("[OBSTACLE_SEQ] 进入避障区: 右转90° (mode=%s)", mode)

        if (now - self.fsm.entered_at) > max_sec:
            self.log.warning("[OBSTACLE_SEQ] 超时 %.1fs, 强制退出避障区", max_sec)
            return self._exit_obstacle_seq(lane)

        red = self._obstacle_red_hit(cls) if use_red else False
        phase = self._obstacle_phase
        el = now - self._obstacle_phase_t

        if use_red and phase == "straight1" and el > phase_max_sec:
            self.log.warning(
                "[OBSTACLE_SEQ] %s 等红色超时 %.1fs, 退出避障区恢复循黄线",
                phase, phase_max_sec,
            )
            return self._exit_obstacle_seq(lane)
        if use_red and phase == "straight2" and el > phase_max_sec:
            self.log.warning(
                "[OBSTACLE_SEQ] straight2 等红色超时 %.1fs, 继续执行左转/直行/右转收尾",
                phase_max_sec,
            )
            self._obstacle_phase = "left90b"
            self._obstacle_phase_t = now
            self._obstacle_red_count = 0
            return turn_vx, turn_yaw

        def _to(name: str) -> None:
            self._obstacle_phase = name
            self._obstacle_phase_t = now
            self._obstacle_red_count = 0
            self.log.info("[OBSTACLE_SEQ] %s -> %s (red_ratio=%.2f)",
                          phase, name, float(getattr(cls, "red_ratio", 0.0)) if cls else 0.0)

        if phase == "right90":
            if el >= right90_sec:
                _to("straight1")
                return straight_vx, 0.0
            return turn_vx, -turn_yaw
        if phase == "straight1":
            if (not use_red and el >= straight1_sec) or (use_red and el >= grace and red):
                _to("left90a")
                return turn_vx, turn_yaw
            return straight_vx, 0.0
        if phase == "left90a":
            if el >= left90a_sec:
                _to("straight2")
                return straight_vx, 0.0
            return turn_vx, turn_yaw
        if phase == "straight2":
            if (not use_red and el >= straight2_sec) or (use_red and el >= grace and red):
                _to("left90b")
                return turn_vx, turn_yaw
            return straight_vx, 0.0
        if phase == "left90b":
            if el >= left90b_sec:
                _to("straight3")
                return straight_vx, 0.0
            return turn_vx, turn_yaw
        if phase == "straight3":
            if el >= straight3_sec:
                if strafe_sec > 0.0 and abs(strafe_vy) > 1e-6:
                    _to("strafe_right")
                    self._cmd_vy = strafe_vy
                    return 0.0, 0.0
                _to("right_final")
                return turn_vx, -turn_yaw
            return straight_vx, 0.0
        if phase == "strafe_right":
            self._cmd_vy = strafe_vy
            if el >= strafe_sec:
                self._cmd_vy = 0.0
                _to("right_final")
                return turn_vx, -turn_yaw
            return 0.0, 0.0
        if phase == "right_final":
            if el >= right_final_sec:
                if final_forward_sec > 0.0:
                    _to("final_forward")
                    return straight_vx, 0.0
                return self._exit_obstacle_seq(lane)
            return turn_vx, -turn_yaw
        if phase == "final_forward":
            if el >= final_forward_sec:
                return self._exit_obstacle_seq(lane)
            return straight_vx, 0.0
        return self._exit_obstacle_seq(lane)

    def _blue_ring_hit(self, blue_ring) -> bool:
        min_conf = float(getattr(self.cfg.landmark, "blue_ring_min_confidence", 0.45))
        if blue_ring is not None and blue_ring.confidence >= min_conf:
            return True
        rs_lane = self._last_rs_blue_ring_lane
        assist_cfg = getattr(getattr(self.cfg, "realsense", None), "blue_ring_assist", None)
        rs_min_conf = float(getattr(assist_cfg, "min_confidence", 0.35)) if assist_cfg else 0.35
        return bool(rs_lane is not None and rs_lane.found and rs_lane.confidence >= rs_min_conf)

    def _blue_ring_front_error(self, blue_ring) -> Optional[float]:
        if blue_ring is None or not blue_ring.extra:
            return None
        image_w = float(blue_ring.extra.get("image_w", self._roi_width()))
        if image_w <= 1:
            return None
        direction = str(getattr(self.cfg.landmark, "obstacle_ring_direction", "right"))
        key = "left_band_x" if direction == "left" else "right_band_x"
        target_x = float(blue_ring.extra.get(key, image_w / 2.0))
        return float(np.clip((target_x - image_w / 2.0) / (image_w / 2.0), -1.0, 1.0))

    def _follow_blue_ring(self, blue_ring, lane: LaneResult):
        cfg = self.cfg
        speed = float(getattr(cfg.landmark, "obstacle_ring_speed", 0.20))
        yaw_gain = float(getattr(cfg.landmark, "obstacle_ring_yaw_gain", 0.65))
        exit_frames = int(getattr(cfg.landmark, "obstacle_ring_exit_yellow_frames", 8))
        exit_conf = float(getattr(cfg.landmark, "obstacle_ring_exit_yellow_conf_min", 0.45))
        lost_frames = int(getattr(cfg.landmark, "obstacle_ring_lost_frames", 10))
        max_sec = float(getattr(cfg.landmark, "obstacle_ring_max_sec", 18.0))

        ring_visible = self._blue_ring_hit(blue_ring)
        if ring_visible:
            self.last_lane_seen_at = time.time()
            self.fsm.vote_landmark_lost(False, lost_frames)

        rs_lane = self._last_rs_blue_ring_lane
        assist_cfg = getattr(getattr(cfg, "realsense", None), "blue_ring_assist", None)
        rs_min_conf = float(getattr(assist_cfg, "min_confidence", 0.35)) if assist_cfg else 0.35
        if rs_lane is not None and rs_lane.found and rs_lane.confidence >= rs_min_conf:
            err = rs_lane.error
            source = "d435i"
        else:
            front_err = self._blue_ring_front_error(blue_ring)
            if front_err is None:
                front_err = 0.35 if str(getattr(cfg.landmark, "obstacle_ring_direction", "right")) == "right" else -0.35
            err = front_err
            source = "front"

        vyaw = self._lateral_to_vyaw(err) * yaw_gain
        if self.fsm.time_in_state() < 0.2:
            self.log.info("[OBSTACLE_RING] 固定右侧沿蓝环绕行 speed=%.2f", speed)
        now = time.time()
        if now - float(getattr(self, "_last_obstacle_ring_log_t", 0.0)) > 1.0:
            self.log.info(
                "[OBSTACLE_RING] source=%s err=%+.3f vyaw=%+.3f visible=%s yellow=(%s %.2f)",
                source, err, vyaw, ring_visible, lane.found, lane.confidence,
            )
            self._last_obstacle_ring_log_t = now

        yellow_recovered = lane.found and lane.confidence >= exit_conf
        ring_lost = self.fsm.vote_landmark_lost(not ring_visible, lost_frames)
        if ring_lost and self.fsm.vote_lane_recovered(yellow_recovered, exit_frames):
            self.fsm.flags.obstacle_avoided = True
            self.fsm.flags.blue_ring_done = True
            self.last_lane_seen_at = time.time()
            self.log.info("[OBSTACLE_RING] 蓝环消失且黄线稳定, 恢复 FOLLOW_LANE")
            self.fsm.transit(State.FOLLOW_LANE)
            return 0.0, 0.0

        if self.fsm.time_in_state() > max_sec:
            self.fsm.flags.obstacle_avoided = True
            self.fsm.flags.blue_ring_done = True
            self.log.warning("[OBSTACLE_RING] 超时 %.1fs, 退出蓝环模式", max_sec)
            if lane.found:
                self.last_lane_seen_at = time.time()
                self.fsm.transit(State.FOLLOW_LANE)
            else:
                self.fsm.transit(State.SEARCH_LANE)
            return 0.0, 0.0

        return speed, vyaw

    def _dump_blackwhite_confirmed(self) -> bool:
        cls = self._last_color_cls
        if cls is None or cls.label != ColorLabel.BLACK_WHITE:
            return False
        rs_cfg = getattr(self.cfg, "realsense", None)
        max_depth = float(getattr(rs_cfg, "dump_trigger_depth_m", 0.40)) if rs_cfg else 0.40
        min_conf = float(getattr(self.cfg.landmark, "dump_ring_blackwhite_min_confidence", 0.50))
        z = float(getattr(cls, "depth_median_m", 0.0))
        return (
            float(getattr(cls, "confidence", 0.0)) >= min_conf
            and z > 0.0
            and z <= max_depth
        )

    def _dump_ring_lane(self, dump) -> LaneResult:
        mask = self._last_mask
        if mask is None or dump is None or not getattr(dump, "extra", None):
            return LaneResult(False, 0.0, 0.0, [])

        extra = dump.extra or {}
        center = extra.get("center")
        radius = float(extra.get("radius", 0.0))
        if not center or radius <= 1.0:
            return LaneResult(False, 0.0, 0.0, [])

        cx, cy = float(center[0]), float(center[1])
        h, w = mask.shape[:2]
        yy, xx = np.indices((h, w))
        band = max(10.0, radius * float(getattr(self.cfg.landmark, "dump_ring_band_ratio", 0.24)))
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        annulus = (dist >= radius - band) & (dist <= radius + band)

        side = str(getattr(self.cfg.landmark, "dump_ring_side", "lower")).lower()
        if side == "upper":
            side_ok = yy <= cy
        elif side == "left":
            side_ok = xx <= cx
        elif side == "right":
            side_ok = xx >= cx
        else:
            side_ok = yy >= cy

        ring_mask = np.zeros_like(mask)
        ring_mask[(mask > 0) & annulus & side_ok] = 255
        return estimate_lane_error(
            ring_mask,
            n_strips=int(getattr(self.cfg.landmark, "dump_ring_n_strips", 6)),
            min_pixels_per_strip=int(getattr(self.cfg.landmark, "dump_ring_min_pixels_per_strip", 20)),
            weight_strategy="near",
        )

    def _follow_dump_ring(self, lane: LaneResult, dump):
        cfg = self.cfg
        entered = self._ensure_approach_distance_state(State.FOLLOW_DUMP_RING)
        if entered:
            self.log.info("[DUMP_RING] 进入黄色圆环一侧跟随, 等待合适位置卸料")

        if self._dump_blackwhite_confirmed():
            self._dump_ring_confirmed = True

        ring_lane = self._dump_ring_lane(dump)
        speed = float(getattr(cfg.landmark, "dump_ring_speed", cfg.control.forward_speed.dump_approach))
        if ring_lane.found:
            vx, vyaw = self._follow(ring_lane, speed)
        else:
            vx, vyaw = self._follow_or_landmark(lane, speed, dump)

        dist = self._approach_distance_update(vx)
        elapsed = self.fsm.time_in_state()
        min_sec = float(getattr(cfg.landmark, "dump_ring_min_follow_sec", 1.0))
        min_dist = float(getattr(cfg.landmark, "dump_ring_min_follow_distance_m", 0.25))
        max_sec = float(getattr(cfg.landmark, "dump_ring_max_sec", 8.0))
        # 倾倒动作执行点在黄色圆环上, 底部 D435i 通常看不到中心白底黑字区域。
        # 黑白识别只作为辅助日志信号, 不能作为比赛版倾倒动作的必要条件。
        ready = elapsed >= min_sec and dist >= min_dist

        if ready:
            self.log.info(
                "[DUMP_RING] 到达卸料点: dist=%.2fm elapsed=%.1fs blackwhite=%s",
                dist, elapsed, self._dump_ring_confirmed,
            )
            self._dump_post_trigger_active = False
            self.fsm.transit(State.DUMP_ACTION)
            return 0.0, 0.0

        if elapsed > max_sec:
            self.log.warning(
                "[DUMP_RING] 超时 %.1fs, 强制进入卸料动作 (dist=%.2fm blackwhite=%s)",
                max_sec, dist, self._dump_ring_confirmed,
            )
            self._dump_post_trigger_active = False
            self.fsm.transit(State.DUMP_ACTION)
            return 0.0, 0.0

        self.last_lane_seen_at = time.time()
        return vx, vyaw

    def _follow(self, lane: LaneResult, vx_target: float):
        if not lane.found:
            return 0.0, 0.0
        vyaw = self._lateral_to_vyaw(lane.error)
        if self._lane_control_source == "realsense":
            assist_cfg = getattr(getattr(self.cfg, "realsense", None), "lane_assist", None)
            vx_cap = float(getattr(assist_cfg, "fallback_speed", 0.20)) if assist_cfg else 0.20
            yaw_scale = float(getattr(assist_cfg, "yaw_scale", 0.65)) if assist_cfg else 0.65
            vx_target = min(vx_target, vx_cap)
            vyaw *= yaw_scale
        # dump 后 2s 限幅, 给 mcf + IMU 时间稳住
        if time.time() < self._post_dump_recover_until_t:
            cap = 0.5
            if vyaw > cap:
                vyaw = cap
            elif vyaw < -cap:
                vyaw = -cap
            vx_target = min(vx_target, 0.20)
        return vx_target, vyaw

    def _follow_or_landmark(self, lane: LaneResult, vx_target: float, landmark):
        """优先沿黄道巡线；黄道被地标遮挡时，用地标横向中心维持慢速前进.

        台阶区/充电区的蓝色标志会短暂遮住黄道。如果这时直接返回 vx=0，
        路程型状态机会停在区域前方。只要 landmark 仍可见，就允许低速向前，
        并用 landmark 的 cx 粗略对准。
        """
        if lane.found:
            return self._follow(lane, vx_target)
        if landmark is None:
            return 0.0, 0.0
        cx_err = self._landmark_cx_error(landmark)
        # cx_err 为正表示目标在右侧；Go2 vyaw 负值向右转。
        vyaw = self._lateral_to_vyaw(-cx_err) * 0.5
        return vx_target, vyaw

    def _lateral_to_vyaw(self, error: float) -> float:
        out = self.pid.step(-error)
        return out

    def _reset_to_follow_lane(self) -> None:
        """passive 模式下用: 把 FSM 重置回 FOLLOW_LANE 让程序继续转, 不让窗口断开."""
        self.fsm.state = State.FOLLOW_LANE
        self.fsm.entered_at = time.time()
        self.fsm.flags.dumped = False
        self.fsm.flags.obstacle_avoided = False
        self.fsm.flags.blue_ring_done = False
        self.fsm.flags.fork_chosen = False
        self.fsm.flags.stair_climbed = False
        self.last_lane_seen_at = time.time()
        self.start_time = time.time()
        self.fork_choice = None
        self.pid.reset()
        self._stair_mode_entered = False
        self._max_prox_seen = 0.0
        self._approach_distance_m = 0.0
        self._approach_track_state = None
        self._dump_post_trigger_active = False
        self._dump_ring_confirmed = False
        self._dock_post_trigger_active = False
        self._post_stair_straight_ready_at = 0.0
        self._post_stair_settle_logged = False
        self._blind_walk_started_at = 0.0
        self._last_rs_blue_ring_lane = None
        self._obstacle_seq_active = False
        self._obstacle_phase = None
        self._obstacle_red_count = 0

    def _roi_height(self) -> float:
        """ROI 实际高度 (像素), 用于按比例判断 landmark 距离."""
        cfg = self.cfg
        if self._last_frame is not None:
            return float(self._last_frame.shape[0]) * (
                float(cfg.camera.roi_bottom_ratio) - float(cfg.camera.roi_top_ratio)
            )
        return float(
            cfg.camera.height
            * (cfg.camera.roi_bottom_ratio - cfg.camera.roi_top_ratio)
        )

    def _roi_width(self) -> float:
        """ROI 实际宽度 (像素), 用于 fork 坐标系兜底."""
        if self._last_frame is not None:
            return float(self._last_frame.shape[1])
        return float(getattr(self.cfg.camera, "width", 1280))

    def _proximity_score(self, det) -> float:
        """估计地标接近度 [0, 1].

        ⚠ 重点: 用 min(y_ratio, size_ratio) 而不是平均 -- 必须**两个指标同时达标**
        才算真接近. 否则:
          - 远处大圆环顶部被遮挡 (size 高 + cy 低) → 平均也会过线 → 误判
          - 近处底部小斑点 (cy 高 + size 低) → 平均也会过线 → 误判
        """
        if det is None or det.bbox is None:
            return 0.0
        roi_h = self._roi_height()
        if roi_h <= 0:
            return 0.0
        _, y, _, h = det.bbox
        cy = y + h / 2.0
        y_ratio = min(cy / roi_h, 1.0)              # 0=ROI 顶, 1=ROI 底 (越靠下越近)
        size_ratio = min((h / roi_h) * 1.5, 1.0)    # *1.5: 实拍 bbox 高度很难超 60%
        return min(y_ratio, size_ratio)             # AND 逻辑: 两个都要够大

    def _landmark_close_enough(self, det, threshold: float) -> bool:
        if det is None:
            return False
        prox = self._proximity_score(det)
        ok = prox >= float(threshold)
        if ok:
            return True
        # 只在检测到了但太远时偶尔打日志, 方便现场知道不是没识别, 是被近距门槛拦住.
        now = time.time()
        key = f"_last_far_{getattr(det.type, 'value', 'det')}_log_t"
        if now - float(getattr(self, key, 0.0)) > 1.0:
            self.log.info(
                "[vision] %s detected but too far: prox=%.2f < %.2f",
                getattr(det.type, "value", "landmark"), prox, threshold,
            )
            setattr(self, key, now)
        return False

    def _landmark_cx_error(self, det) -> float:
        """地标中心 cx 相对画面中心的归一化偏差 [-1, +1].
        正 = 地标在画面右侧, 狗需要向右转 (vyaw < 0).
        0 = 地标在画面正中央.
        返回 0 表示没检测到."""
        if det is None or det.bbox is None:
            return 0.0
        x, _, w, _ = det.bbox
        cx = x + w / 2.0
        roi_w = float(det.extra.get("image_w", 0)) if det.extra else 0
        if roi_w <= 0:
            if self._last_frame is not None:
                roi_w = float(self._last_frame.shape[1])
            else:
                return 0.0
        return (cx - roi_w / 2.0) / (roi_w / 2.0)

    def _is_centered(self, det, cx_tolerance: float = 0.15) -> bool:
        """地标 cx 是否在画面中心 ±tolerance 以内."""
        return abs(self._landmark_cx_error(det)) <= cx_tolerance

    def _approach_distance_reset(self) -> None:
        """APPROACH_X 进入时调用, 重置路程计."""
        self._approach_start_time = time.time()
        self._approach_distance_m = 0.0
        self._approach_last_tick = time.time()

    def _ensure_approach_distance_state(self, state: State) -> bool:
        """进入新的 APPROACH 状态时可靠重置路程计.

        旧逻辑用 time_in_state()<0.05 判断进入帧。实际取图/SDK 调用稍慢时,
        第一帧可能已经超过 50ms, 导致 _approach_last_tick 仍为 0, 路程积分
        直接加上 epoch 秒数, 触发几亿米的假距离。
        """
        if self._approach_track_state != state:
            self._approach_track_state = state
            self._approach_distance_reset()
            return True
        return False

    def _approach_distance_update(self, vx: float) -> float:
        """每帧调用, 累加路程. 返回累计行走距离 (米)."""
        now = time.time()
        dt = max(now - self._approach_last_tick, 0.001)
        self._approach_last_tick = now
        self._approach_distance_m += abs(vx) * dt
        return self._approach_distance_m

    def _track_proximity(self, det) -> "tuple[float, float]":
        """跟踪当前 APPROACH_X 状态期间地标接近度.
        返回 (当前帧接近度, 本状态期间最大接近度).
        切换到新 FSM 状态时自动重置."""
        cur_st = self.fsm.state
        if cur_st != self._prox_track_state:
            self._max_prox_seen = 0.0
            self._prox_track_state = cur_st
        cur = self._proximity_score(det)
        if cur > self._max_prox_seen:
            self._max_prox_seen = cur
        return cur, self._max_prox_seen

    def _manual_advance_state(self) -> None:
        """dry-run 调试用: 强制把当前 APPROACH/CLIMB 状态推进到下一状态.
        实战中绝对不应该用这个. dry-run 下狗不动, 否则会卡 30s 超时."""
        s = self.fsm.state
        msg = f"[manual-advance] {s.value} -> "
        if s == State.APPROACH_DUMP:
            self.fsm.transit(State.DUMP_ACTION)
            self.log.warning(msg + "DUMP_ACTION")
        elif s == State.APPROACH_STAIR:
            if self.robot is not None:
                exit_stair_mode(self.robot)
            self.fsm.flags.stair_climbed = True
            self.fsm.transit(State.FOLLOW_LANE)
            self.log.warning(msg + "FOLLOW_LANE (stair_climbed=True)")
        elif s == State.CLIMB_STAIR:
            if self.robot is not None:
                exit_stair_mode(self.robot)
            self.fsm.flags.stair_climbed = True
            self.fsm.transit(State.FOLLOW_LANE)
            self.log.warning(msg + "FOLLOW_LANE")
        elif s == State.APPROACH_DOCK:
            self.fsm.transit(State.DOCK)
            self.log.warning(msg + "DOCK")
        elif s == State.SEARCH_LANE:
            self.last_lane_seen_at = time.time()
            self.fsm.transit(State.FOLLOW_LANE)
            self.log.warning(msg + "FOLLOW_LANE (假装找到黄道)")
        else:
            self.log.info("[manual-advance] %s 无下一步, 忽略", s.value)

    def _save_debug_snapshot(self) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.cfg.logging.save_dir) / f"snap_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, img in [
            ("frame", self._last_frame),
            ("warped", self._last_warped),
            ("mask", self._last_mask),
            ("debug", self._last_debug),
        ]:
            if img is not None:
                cv2.imwrite(str(out_dir / f"{name}.png"), img)
        self.log.info("已保存调试快照到 %s", out_dir)

    def _update_fps(self, now: float) -> None:
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst


def _build_source(args, cfg, robot, logger) -> FrameSource:
    if args.replay:
        logger.info("使用离线视频源: %s", args.replay)
        return VideoFileSource(args.replay, loop=False)
    if robot is None:
        raise RuntimeError("既没有 --replay 也没有连狗，无源可读")
    return Go2CameraSource(robot, logger=logger)


def _setup_signal(robot: Optional[Go2Client]) -> None:
    def _handler(sig, _frame):
        if robot is not None:
            try:
                robot.stop_move()
                robot.balance_stand()
            except Exception:
                pass
        sys.exit(130)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _apply_start_stage(runner: MissionRunner, start_stage: str, log) -> None:
    """Apply manual mission-progress flags for segmented field tests."""
    stage = (start_stage or "normal").strip().lower()
    if stage in ("", "normal"):
        return
    if stage in ("after-obstacle", "dump-entry"):
        runner.fsm.flags.obstacle_avoided = True
        runner.fsm.flags.blue_ring_done = True
        log.info("[START_STAGE] 从倾倒区前开始: 已跳过避障区 flags")
        return
    raise ValueError(f"unknown start stage: {start_stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_DEFAULT_CFG))
    parser.add_argument("--network", default=None, help="DDS 网卡, 默认读 config")
    parser.add_argument("--province", action="store_true", help="省赛模式 (默认)")
    parser.add_argument("--replay", default=None, help="离线视频回灌, 不连狗")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--headless-record", default=None, help="把叠加图录到文件")
    parser.add_argument("--dry-run", action="store_true",
                        help="连狗只为读视频流, 任何动狗指令都 noop. 用于 Mac 端实时调参")
    parser.add_argument("--realsense", action="store_true",
                        help="启用 D435i 辅助")
    parser.add_argument("--no-realsense", action="store_true",
                        help="强制关闭 D435i")
    parser.add_argument("--mjpeg-port", type=int, default=0,
                        help="MJPEG 流端口 (默认 0=不启). e.g. --mjpeg-port 8088")
    parser.add_argument(
        "--start-stage",
        choices=("normal", "after-obstacle", "dump-entry"),
        default="normal",
        help="分段测试入口: after-obstacle/dump-entry 会跳过起点避障, 从倾倒区前开始",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = get_logger("go2patrol", level=cfg.logging.level, save_dir=cfg.logging.save_dir)

    iface = args.network or cfg.network.interface

    robot: Optional[Go2Client] = None
    if not args.replay:
        try:
            robot = Go2Client(
                network_iface=iface,
                domain_id=int(cfg.network.domain_id),
                max_vx=float(getattr(cfg.robot, "max_vx", 0.5)),
                max_vy=float(getattr(cfg.robot, "max_vy", 0.3)),
                max_vyaw=float(getattr(cfg.robot, "max_vyaw", 1.0)),
                velocity_hz=int(getattr(cfg.robot, "velocity_hz", 20)),
                dry_run=bool(args.dry_run),
                logger=log,
            )
            robot.init()
        except Go2ClientError as e:
            log.error("连狗失败: %s", e)
            log.error("如果你只想离线调试，请加 --replay <video>")
            sys.exit(2)

    _setup_signal(robot)

    source = _build_source(args, cfg, robot, log)

    video_writer = None
    if args.headless_record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.headless_record, fourcc, 20.0,
            tuple(cfg.camera.ipm_dst_size),
        )

    display = not args.no_display
    # 仅在 Linux 上检查 DISPLAY (macOS 用 Cocoa, Windows 用 Win32, 都不需要 DISPLAY)
    if display and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        log.warning("Linux 无 DISPLAY 环境变量 (狗端无 GUI), 自动切到 no-display")
        display = False

    rs_cam = None
    rs_cfg = getattr(cfg, "realsense", None)
    rs_enabled_cfg = bool(getattr(rs_cfg, "enabled", False)) if rs_cfg else False
    rs_wanted = (args.realsense or rs_enabled_cfg) and not args.no_realsense
    if rs_wanted:
        if RealsenseCamera is None:
            log.error("--realsense 但 RealsenseCamera import 失败")
        else:
            rs_cam = RealsenseCamera(
                width=int(getattr(rs_cfg, "width", 640)),
                height=int(getattr(rs_cfg, "height", 480)),
                fps=int(getattr(rs_cfg, "fps", 30)),
                logger=log,
            )
            if not rs_cam.open():
                log.error("D435i 启动失败, 继续 (无 D435i 辅助)")
                rs_cam = None

    mjpeg = None
    if args.mjpeg_port and args.mjpeg_port > 0:
        mjpeg = MjpegServer(port=int(args.mjpeg_port), jpeg_quality=70, max_width=960)
        try:
            mjpeg.start()
            log.info("MJPEG live stream 启动: http://<robot-ip>:%d/", args.mjpeg_port)
        except Exception as e:
            log.error("MJPEG server 启动失败: %s", e)
            mjpeg = None

    runner = MissionRunner(cfg, source, robot, log,
                           realsense_cam=rs_cam, mjpeg_server=mjpeg)
    _apply_start_stage(runner, args.start_stage, log)
    try:
        runner.run(display=display, video_writer=video_writer)
    finally:
        source.release()
        if video_writer is not None:
            video_writer.release()
        if rs_cam is not None:
            rs_cam.close()
        if mjpeg is not None:
            mjpeg.stop()


if __name__ == "__main__":
    main()
