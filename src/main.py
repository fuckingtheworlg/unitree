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
        self._last_rs_log_t: float = 0.0
        self._color_triggers: Dict[str, ColorTriggerState] = {}
        self._post_dump_recover_until_t: float = 0.0
        self._dump_done_at: float = 0.0
        self._blind_walk_started_at: float = 0.0

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
                    self.robot.set_velocity(vx, 0.0, vyaw)

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
            ):
                v = getattr(ccfg, key, None)
                if v is not None:
                    kw[key] = float(v)
            ne = getattr(ccfg, "night_enabled", None)
            if ne is not None:
                kw["night_enabled"] = bool(ne)
        cls = classify_color_combo(
            self._last_rs_frame.rgb,
            depth_raw=self._last_rs_frame.depth_raw,
            depth_scale=self._last_rs_frame.depth_scale,
            require_depth=True,
            **kw,
        )
        self._last_color_cls = cls
        now = time.time()
        if now - self._last_rs_log_t > 0.5 and cls.label != ColorLabel.NONE:
            self.log.info(
                "[RS] %s conf=%.2f w=%.2f k=%.2f b=%.2f y=%.2f z=%.2fm",
                cls.label.value, cls.confidence,
                cls.white_ratio, cls.black_ratio, cls.blue_ratio,
                cls.yellow_ratio, cls.depth_median_m,
            )
            self._last_rs_log_t = now
        return cls

    def _ensure_color_triggers(self) -> None:
        if self._color_triggers:
            return
        rs_cfg = getattr(self.cfg, "realsense", None)
        if rs_cfg is None:
            return
        n_stable = int(getattr(rs_cfg, "n_stable_frames", 5))
        min_hits = int(getattr(rs_cfg, "min_hits_in_window", 4))
        cd_dump_s = float(getattr(rs_cfg, "cooldown_dump_s", 5.0))
        cd_stair_s = float(getattr(rs_cfg, "cooldown_stair_s", 8.0))
        cd_dock_s = float(getattr(rs_cfg, "cooldown_dock_s", 8.0))
        self._color_triggers = {
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

    def _step(self, frame):
        cfg = self.cfg
        self._read_realsense()
        self._classify_color()
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
        )
        # 巡线在 ROI mask 上跑 (不再 warp, 否则黄道在 ROI 中上部会被 IPM 扔出画面).
        mask = mask_roi
        warped = self.ipm.warp(roi)  # 仅用于可视化
        lane = estimate_lane_error(mask, n_strips=8, min_pixels_per_strip=60)
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

        # 蓝色矩形检测: 充电区/台阶区视觉特征一样, 由 FSM stair_climbed 标志区分语义.
        # 第一次见 -> stair, 第二次见 -> dock.
        from src.vision.landmark import detect_blue_rect
        blue_rect = detect_blue_rect(
            roi,
            min_area_ratio=float(getattr(cfg.landmark, "blue_rect_min_area_ratio", 0.02)),
        )
        if not self.fsm.flags.stair_climbed:
            stair = blue_rect
            dock = None
        else:
            stair = None
            dock = blue_rect
        if blue_rect is not None:
            from src.vision.landmark import LandmarkType
            blue_rect.type = LandmarkType.DOCK_AREA if self.fsm.flags.stair_climbed else LandmarkType.STAIR
            landmarks.append(blue_rect)

        vx, vyaw = self._dispatch(lane, dump, fork, stair, dock)

        # debug 画布用 ROI 而不是 warped, 让 landmark bbox 坐标和画布对齐.
        # mask_roi 同样在 ROI 坐标系, lane.debug_centroid 用 warped 坐标系所以不画.
        debug = overlay(
            roi,
            mask=mask_roi,
            lane=None,
            landmarks=landmarks,
            state_text=self.fsm.state.value,
            error=lane.error if lane.found else 0.0,
            vyaw=vyaw,
            fps=self._fps,
        )
        # 把巡线 debug_centroid 单独画在右上角的 IPM 缩略图上 (可选)
        if lane.found and lane.debug_centroid is not None:
            mini_h = 200
            mini_w = 200
            mini = cv2.resize(warped, (mini_w, mini_h))
            mh, mw = mini.shape[:2]
            cx_w = int(lane.debug_centroid[0] * mw / mask.shape[1])
            cy_w = int(lane.debug_centroid[1] * mh / mask.shape[0])
            cv2.circle(mini, (cx_w, cy_w), 6, (0, 255, 0), -1)
            cv2.line(mini, (mw // 2, 0), (mw // 2, mh), (255, 255, 255), 1)
            # 贴到 debug 右上角
            dh, dw = debug.shape[:2]
            x_offset = dw - mw - 10
            y_offset = 35
            if x_offset >= 0 and y_offset + mh <= dh:
                debug[y_offset:y_offset + mh, x_offset:x_offset + mw] = mini
                cv2.rectangle(debug, (x_offset, y_offset), (x_offset + mw, y_offset + mh), (255, 255, 0), 2)

        if self._last_lane_found != lane.found:
            n_valid = sum(1 for c in lane.centerline_x if c >= 0)
            yellow_px = int((mask > 0).sum())
            self.log.info(
                "[vision] lane.found=%s, valid_strips=%d/8, yellow_px=%d (mask shape=%s), conf=%.2f, err=%+.3f",
                lane.found, n_valid, yellow_px, mask.shape, lane.confidence, lane.error,
            )
            self._last_lane_found = lane.found

        if self._last_rs_frame is not None and self._last_rs_frame.rgb is not None:
            try:
                self._draw_rs_thumbnail(debug)
            except Exception:
                pass

        if self.mjpeg is not None:
            try:
                self.mjpeg.push_frame(debug)
            except Exception:
                pass

        self._last_frame = frame
        self._last_warped = warped
        self._last_mask = mask
        self._last_debug = debug
        return vx, vyaw, debug

    def _dispatch(self, lane: LaneResult, dump, fork, stair, dock):
        cfg = self.cfg
        f = cfg.control.forward_speed
        st = self.fsm.state

        vx = 0.0
        vyaw = 0.0

        if st == State.FOLLOW_LANE:
            rs_cfg = getattr(cfg, "realsense", None)
            rs_enabled = self.rs_cam is not None
            cls = self._last_color_cls
            color_active = rs_enabled and cls is not None and bool(self._color_triggers)

            # ---- stair 完成后: 纯直行, 完全不看黄线, 等 D435i 终点触发 ----
            if (color_active and self.fsm.flags.dumped
                    and self.fsm.flags.stair_climbed):
                cruise_speed = float(getattr(rs_cfg, "dock_blind_walk_speed", 0.25))
                if self._blind_walk_started_at <= 0.0:
                    self._blind_walk_started_at = time.time()
                    self.log.info(
                        "[FSM] 进入终点接近模式 (纯直行 vx=%.2fm/s vyaw=0, "
                        "完全不看黄线, 等 D435i 终点触发)",
                        cruise_speed,
                    )
                vx = cruise_speed
                vyaw = 0.0
                self.last_lane_seen_at = time.time()  # 抑制 SEARCH_LANE
                trig = self._color_triggers["dock"]
                if trig.update(cls):
                    trig.disarm()
                    self._save_trigger_snapshot("dock")
                    self.log.info(
                        "[FSM] D435i BLUE_WHITE 2nd @ z=%.2fm -> DOCK",
                        cls.depth_median_m,
                    )
                    self.fsm.transit(State.DOCK)
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
                # (1) dump
                if not self.fsm.flags.dumped:
                    trig = self._color_triggers["dump"]
                    if trig.update(cls):
                        trig.disarm()
                        self._save_trigger_snapshot("dump")
                        self.log.info(
                            "[FSM] D435i BLACK_WHITE @ z=%.2fm -> DUMP_ACTION",
                            cls.depth_median_m,
                        )
                        self.fsm.transit(State.DUMP_ACTION)
                        return 0.0, 0.0

                # (2) stair (dump 后 dump_to_stair_min_s 秒才允许)
                if (self.fsm.flags.dumped
                        and not self.fsm.flags.stair_climbed):
                    dump_to_stair_min_s = float(
                        getattr(rs_cfg, "dump_to_stair_min_s", 15.0)
                    )
                    elapsed_since_dump = (
                        time.time() - self._dump_done_at
                        if self._dump_done_at > 0 else 0.0
                    )
                    if elapsed_since_dump < dump_to_stair_min_s:
                        if not getattr(self, "_dump_to_stair_protect_logged", False):
                            self.log.info(
                                "[FSM] dump→stair 保护期 (%.1fs < %.1fs), stair 触发禁用",
                                elapsed_since_dump, dump_to_stair_min_s,
                            )
                            self._dump_to_stair_protect_logged = True
                    else:
                        if (getattr(self, "_dump_to_stair_protect_logged", False)
                                and not getattr(self, "_dump_to_stair_protect_exit_logged", False)):
                            self.log.info(
                                "[FSM] dump→stair 保护期结束 (%.1fs)",
                                elapsed_since_dump,
                            )
                            self._dump_to_stair_protect_exit_logged = True
                        trig = self._color_triggers["stair"]
                        if trig.update(cls):
                            trig.disarm()
                            self._save_trigger_snapshot("stair")
                            self.log.info(
                                "[FSM] D435i BLUE_WHITE 1st @ z=%.2fm -> APPROACH_STAIR",
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
                            image_w = int(cfg.camera.ipm_dst_size[0])
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
                if not self.fsm.flags.dumped:
                    if self.fsm.vote_dump(dump is not None, int(cfg.landmark.dump_consecutive_frames)):
                        self.fsm.transit(State.APPROACH_DUMP)
                        return self._follow(lane, float(f.dump_approach))
                elif not self.fsm.flags.fork_chosen:
                    if self.fsm.vote_fork(fork is not None, int(cfg.landmark.fork_consecutive_frames)):
                        if fork is not None and fork.extra:
                            image_w = int(cfg.camera.ipm_dst_size[0])
                            self.fork_choice = choose_shortest_fork_branch(
                                float(fork.extra["left_x"]),
                                float(fork.extra["right_x"]),
                                image_w,
                            )
                            self.log.info("[FORK] 选 %s 支", self.fork_choice)
                        self.fsm.flags.fork_chosen = True
                        self.fsm.transit(State.CHOOSE_FORK)
                elif not self.fsm.flags.stair_climbed:
                    if self.fsm.vote_stair(stair is not None, int(cfg.landmark.stair_consecutive_frames)):
                        self.fsm.transit(State.APPROACH_STAIR)
                else:
                    if self.fsm.vote_dock(dock is not None, int(cfg.landmark.dock_consecutive_frames)):
                        self.fsm.transit(State.APPROACH_DOCK)

        elif st == State.APPROACH_DUMP:
            if self.fsm.time_in_state() < 0.05:
                self._approach_distance_reset()

            vx, vyaw = self._follow_or_landmark(lane, float(f.dump_approach), dump)
            dist = self._approach_distance_update(vx)
            target_dist = float(getattr(cfg.landmark, "dump_walk_distance_m", 1.2))

            if dist >= target_dist:
                self.log.info("[DUMP] 路程 %.2fm >= %.2fm, 到达倾倒区中心, 卸料", dist, target_dist)
                self.fsm.transit(State.DUMP_ACTION)
                return 0.0, 0.0
            if self.fsm.time_in_state() > 15.0:
                self.log.warning("APPROACH_DUMP 超时 15s, 强制卸料")
                self.fsm.transit(State.DUMP_ACTION)
                return 0.0, 0.0

        elif st == State.DUMP_ACTION:
            if self.robot is not None:
                self.log.info("[DUMP] 开始卸料动作")
                execute_dump_action(
                    self.robot,
                    roll_rad=float(cfg.dump_action.roll_rad),
                    hold_sec=float(cfg.dump_action.hold_sec),
                    spam_hz=int(cfg.dump_action.spam_hz),
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

        elif st == State.APPROACH_STAIR:
            # D435i 触发已经把狗放在台阶前 ~0.4m, 立刻停步切步态; 走够路程退出.
            if self.fsm.time_in_state() < 0.05:
                self._approach_distance_reset()
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
                vx, vyaw = self._follow_or_landmark(lane, float(f.stair), stair)
                dist = self._approach_distance_update(vx)
                if dist >= target_through:
                    self.log.info("[STAIR] 穿过台阶 (路程 %.2fm)", dist)
                    if self.robot is not None:
                        exit_stair_mode(self.robot)
                    self.fsm.flags.stair_climbed = True
                    self._stair_mode_entered = False
                    self.fsm.transit(State.FOLLOW_LANE)
                    return 0.0, 0.0

            if self.fsm.time_in_state() > 30.0:
                self.log.warning("APPROACH_STAIR 超时 30s, 强制退出")
                if self.robot is not None and self._stair_mode_entered:
                    exit_stair_mode(self.robot)
                self.fsm.flags.stair_climbed = True
                self._stair_mode_entered = False
                self.fsm.transit(State.FOLLOW_LANE)
                return 0.0, 0.0

        elif st == State.CLIMB_STAIR:
            # 已废弃: 改用 APPROACH_STAIR 单状态包含步态切换+穿越判定.
            # 万一旧逻辑误进入这个状态, 直接切回 FOLLOW_LANE 兜底.
            self.log.warning("CLIMB_STAIR 已废弃, 切回 FOLLOW_LANE")
            self.fsm.flags.stair_climbed = True
            self.fsm.transit(State.FOLLOW_LANE)
            return 0.0, 0.0

        elif st == State.APPROACH_DOCK:
            if self.fsm.time_in_state() < 0.05:
                self._approach_distance_reset()

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
            (st == State.APPROACH_DUMP and dump is not None)
            or (st == State.APPROACH_STAIR and stair is not None)
            or (st == State.APPROACH_DOCK and dock is not None)
        )
        if not lane.found and not has_active_landmark and st in (
            State.FOLLOW_LANE, State.APPROACH_DUMP, State.APPROACH_STAIR, State.APPROACH_DOCK
        ):
            if (time.time() - self.last_lane_seen_at) > float(cfg.control.lost_lane_timeout_sec):
                self.log.warning("丢线超时, 进入 SEARCH_LANE")
                self.fsm.transit(State.SEARCH_LANE)

        return vx, vyaw

    def _follow(self, lane: LaneResult, vx_target: float):
        if not lane.found:
            return 0.0, 0.0
        vyaw = self._lateral_to_vyaw(lane.error)
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
        self.fsm.flags.fork_chosen = False
        self.fsm.flags.stair_climbed = False
        self.last_lane_seen_at = time.time()
        self.start_time = time.time()
        self.fork_choice = None
        self.pid.reset()
        self._stair_mode_entered = False
        self._max_prox_seen = 0.0
        self._approach_distance_m = 0.0

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
