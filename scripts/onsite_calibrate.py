"""国赛现场快速标定 (Mac 端跑, 全自动: 测量→调参→写yaml→部署→重启).

15 分钟现场流程, 每步一条命令:
  .venv/bin/python scripts/onsite_calibrate.py lane     # 黄线: 自动选 hsv/lab 并部署
  .venv/bin/python scripts/onsite_calibrate.py dump     # 狗鼻贴倾倒区白圆 ~25cm
  .venv/bin/python scripts/onsite_calibrate.py stair    # 狗鼻贴台阶牌 ~25cm
  .venv/bin/python scripts/onsite_calibrate.py dock     # 狗鼻贴充电牌 ~25cm

每条命令自动完成: 采集 → 分析 → (必要时)改 config/params.yaml → rsync 上狗 → 杀 main.
全程不需要人判断. 输出最后一行只看 [PASS] / [FIXED] / [FAIL]:
  [PASS]  = 不用动, 直接下一步
  [FIXED] = 已自动调参并部署, 直接下一步
  [FAIL]  = 硬件问题 (USB/网线), 按提示处理

加 --no-apply 只看结果不改配置.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROBOT = "unitree@192.168.123.18"
PASS = "123"
YAML = Path(__file__).resolve().parent.parent / "config" / "params.yaml"
DDS_ENV = ("export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds && "
           "export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH")


def _ssh(cmd: str, tty: bool = False, timeout: int = 60) -> str:
    args = ["sshpass", "-p", PASS, "ssh"]
    if tty:
        args.append("-tt")
    args += ["-o", "StrictHostKeyChecking=no", ROBOT, cmd]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"


def _pull(remote: str, local: str) -> bool:
    try:
        subprocess.run(["sshpass", "-p", PASS, "rsync", "-a",
                        f"{ROBOT}:{remote}", local], check=True, timeout=60)
        return True
    except Exception:
        return False


def _set_yaml(key: str, value: str) -> None:
    """改 config/params.yaml 里 'key:' 开头那行的值 (保留缩进和注释)."""
    text = YAML.read_text(encoding="utf-8")
    pattern = rf"^(\s*{re.escape(key)}:\s*)[^\s#]+"
    new_text, n = re.subn(pattern, rf"\g<1>{value}", text, count=1, flags=re.M)
    if n == 0:
        raise RuntimeError(f"yaml 里找不到 {key}:")
    YAML.write_text(new_text, encoding="utf-8")
    print(f"  yaml: {key} -> {value}")


def _deploy_and_restart() -> None:
    print("  部署 yaml 上狗 + 杀 main (狗端需重新启动 run_full_robot.sh)...")
    subprocess.run(["sshpass", "-p", PASS, "rsync", "-a",
                    str(YAML), f"{ROBOT}:~/go2-patrol/config/"],
                   check=True, timeout=60)
    _ssh("pkill -9 -f src.main || true")


# ============ lane ============

def calibrate_lane(apply: bool) -> int:
    import cv2
    import numpy as np
    from src.utils.config import load_config
    from src.vision.lane_follow import estimate_lane_error
    from src.vision.yellow_mask import crop_roi, yellow_mask, yellow_mask_lab

    print(">>> [1/3] 狗端录 5 秒 (狗保持静止, 视野含黄道)...")
    out = _ssh(f"cd ~/go2-patrol && {DDS_ENV} && "
               "python3 scripts/record_camera.py --duration 5 --network eth0 "
               "--output recordings/onsite_lane.avi 2>&1 | tail -1", tty=True)
    if "输出" not in out:
        print(out.strip()[-300:])
        print("[FAIL] 录像失败: 查网线 / 杀掉占用 DDS 的进程: "
              "sshpass -p 123 ssh unitree@192.168.123.18 'pkill -9 -f src.main'")
        return 2
    if not _pull("~/go2-patrol/recordings/onsite_lane.avi", "recordings/"):
        print("[FAIL] rsync 拉取失败, 查网线")
        return 2

    print(">>> [2/3] 分析 HSV / LAB 两个检测器...")
    cfg = load_config(str(YAML))
    ranges = [(tuple(it[0]), tuple(it[1])) for it in cfg.vision.yellow_hsv_ranges]
    cap = cv2.VideoCapture("recordings/onsite_lane.avi")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        print("[FAIL] 视频空")
        return 2

    out_dir = Path("logs/onsite")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%H%M%S")
    scores = {}
    for det in ("hsv", "lab"):
        pcts, confs, founds = [], [], []
        mid_blend = None
        for idx in np.linspace(0, len(frames) - 1, 6, dtype=int):
            roi, _ = crop_roi(frames[idx], cfg.camera.roi_top_ratio,
                              cfg.camera.roi_bottom_ratio)
            if det == "hsv":
                m = yellow_mask(roi, ranges=ranges, open_kernel=3, close_kernel=7)
            else:
                m = yellow_mask_lab(roi, open_kernel=3, close_kernel=7)
            lane = estimate_lane_error(m, n_strips=8, min_pixels_per_strip=60)
            pcts.append(100.0 * (m > 0).sum() / m.size)
            confs.append(lane.confidence)
            founds.append(lane.found)
            if idx == len(frames) // 2 or mid_blend is None:
                ov = roi.copy()
                ov[m > 0] = (0, 255, 255)
                mid_blend = cv2.addWeighted(roi, 0.45, ov, 0.55, 0)
        pct, conf = float(np.mean(pcts)), float(np.mean(confs))
        found_ratio = float(np.mean(founds))
        # 合格: 黄量 3~45% (太少=丢线, 太多=误判大片), 全帧 found, conf 不太差
        ok_flag = (3.0 <= pct <= 45.0) and found_ratio >= 0.99 and conf >= 0.4
        scores[det] = (ok_flag, pct, conf)
        p = out_dir / f"lane_{det}_{ts}.jpg"
        cv2.putText(mid_blend, f"{det.upper()}: {pct:.1f}% conf={conf:.2f} "
                                f"{'OK' if ok_flag else 'BAD'}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imwrite(str(p), mid_blend)
        print(f"  {det.upper():4s}: 黄量 {pct:5.1f}%  conf {conf:.2f}  "
              f"{'合格' if ok_flag else '不合格'}   图: {p}")

    print(">>> [3/3] 决策...")
    hsv_ok = scores["hsv"][0]
    lab_ok = scores["lab"][0]
    cur = str(getattr(cfg.vision, "lane_detector", "hsv")).lower()
    if hsv_ok:
        choice = "hsv"
    elif lab_ok:
        choice = "lab"
    else:
        print("[FAIL] 两个检测器都不合格! 打开上面两张图人工看哪个接近能用,")
        print("       黄量太低 -> 黄道没在视野里, 挪狗重跑;")
        print("       黄量太高 -> 大片误判, 选误判少的那个手动改 lane_detector")
        return 1

    if choice == cur:
        print(f"[PASS] 当前 lane_detector={cur} 即可, 无需改动")
        return 0
    if not apply:
        print(f"[建议] lane_detector: {cur} -> {choice} (--no-apply 模式未改)")
        return 0
    _set_yaml("lane_detector", choice)
    _deploy_and_restart()
    print(f"[FIXED] lane_detector 已切到 {choice} 并部署. 狗端重启 run_full_robot.sh")
    return 0


# ============ landmark ============

def calibrate_landmark(kind: str, apply: bool) -> int:
    import numpy as np
    from src.utils.config import load_config
    from src.vision.realsense_target import classify_color_combo

    print(f">>> [1/3] 抓 D435i ({kind}: 狗鼻贴牌 ~25cm)...")
    snap_code = (
        "import sys; sys.path.insert(0, '.');"
        "from src.vision.realsense_camera import RealsenseCamera;"
        "import numpy as np, time;"
        "cam = RealsenseCamera(640, 480, 30, logger=None);"
        "ok = cam.open();"
        "assert ok, 'D435i open fail';"
        "time.sleep(1.0);"
        "f = cam.read_latest(2.0);"
        "assert f is not None, 'no frame';"
        "np.savez_compressed('/tmp/onsite_snap.npz', rgb=f.rgb, depth_raw=f.depth_raw,"
        " depth_scale=f.depth_scale);"
        "cam.close();"
        "print('SNAP OK')"
    )
    out = _ssh(f'cd ~/go2-patrol && python3 -c "{snap_code}" 2>&1 | tail -1')
    if "SNAP OK" not in out:
        print(out.strip()[-200:])
        print("[FAIL] D435i 抓帧失败: 1) main 在跑? pkill -9 -f src.main  "
              "2) USB 掉了? lsusb | grep -i intel, 重插蓝色 USB3 口")
        return 2
    if not _pull("/tmp/onsite_snap.npz", "/tmp/"):
        print("[FAIL] rsync 失败, 查网线")
        return 2

    print(">>> [2/3] 颜色分析...")

    def classify_with(cfg_kw):
        data = np.load("/tmp/onsite_snap.npz")
        return classify_color_combo(
            data["rgb"], depth_raw=data["depth_raw"],
            depth_scale=float(data["depth_scale"]), require_depth=True, **cfg_kw,
        )

    def load_kw():
        cfg = load_config(str(YAML))
        ccfg = cfg.realsense.color_stat
        kw = {}
        for key in ("white_min_ratio", "black_min_ratio", "blue_min_ratio",
                    "other_max_ratio", "night_white_min_ratio",
                    "night_black_min_ratio", "night_blue_min_ratio",
                    "night_other_max_ratio_blackwhite",
                    "night_other_max_ratio_bluewhite",
                    "night_blue_dominant_sum", "depth_min_m", "depth_max_m"):
            v = getattr(ccfg, key, None)
            if v is not None:
                kw[key] = float(v)
        kw["night_enabled"] = bool(getattr(ccfg, "night_enabled", True))
        return kw

    expected = {"dump": "black_white", "stair": "blue_white",
                "dock": "blue_white"}[kind]
    cls = classify_with(load_kw())
    print(f"  label={cls.label.value} (期望 {expected})  "
          f"w={cls.white_ratio:.2f} k={cls.black_ratio:.2f} "
          f"b={cls.blue_ratio:.2f} z={cls.depth_median_m:.2f}m")

    print(">>> [3/3] 决策...")
    if cls.label.value == expected:
        print(f"[PASS] {kind} 能正确触发, 无需改动")
        return 0

    if cls.depth_median_m <= 0 or cls.depth_valid_ratio < 0.3:
        print("[FAIL] 深度无效 (狗离牌太近 <5cm 或太远 >50cm), 挪到 ~25cm 重跑")
        return 1

    # ---- 自动调参: 按实测值带余量放宽 night 档 ----
    fixes = []
    if expected == "black_white":
        if cls.white_ratio < load_kw()["night_white_min_ratio"]:
            fixes.append(("night_white_min_ratio",
                          round(max(0.15, cls.white_ratio * 0.75), 2)))
        if cls.black_ratio < load_kw()["night_black_min_ratio"]:
            fixes.append(("night_black_min_ratio",
                          round(max(0.10, cls.black_ratio * 0.7), 2)))
        if cls.blue_ratio > load_kw()["night_other_max_ratio_blackwhite"]:
            fixes.append(("night_other_max_ratio_blackwhite",
                          round(min(0.40, cls.blue_ratio + 0.05), 2)))
    else:  # blue_white
        if cls.blue_ratio < load_kw()["night_blue_min_ratio"]:
            fixes.append(("night_blue_min_ratio",
                          round(max(0.03, cls.blue_ratio * 0.6), 2)))
        if (cls.white_ratio < load_kw()["night_white_min_ratio"]
                and (cls.white_ratio + cls.blue_ratio)
                < load_kw()["night_blue_dominant_sum"]):
            fixes.append(("night_white_min_ratio",
                          round(max(0.15, cls.white_ratio * 0.75), 2)))
        if cls.black_ratio > load_kw()["night_other_max_ratio_bluewhite"]:
            fixes.append(("night_other_max_ratio_bluewhite",
                          round(min(0.45, cls.black_ratio + 0.05), 2)))

    if not fixes:
        print("[FAIL] 没找到可调的阈值 (颜色特征完全不符), 把上面 w/k/b 记下来")
        return 1
    if not apply:
        print(f"[建议] {fixes} (--no-apply 模式未改)")
        return 0

    for key, val in fixes:
        _set_yaml(key, str(val))

    cls2 = classify_with(load_kw())
    if cls2.label.value != expected:
        print(f"[FAIL] 调参后复验仍是 {cls2.label.value}, 阈值救不了, 记下 w/k/b")
        return 1
    _deploy_and_restart()
    print(f"[FIXED] {kind} 阈值已自动调整并部署 (复验通过). 狗端重启 run_full_robot.sh")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["lane", "stair", "dump", "dock"])
    parser.add_argument("--no-apply", action="store_true",
                        help="只看结果不改配置")
    args = parser.parse_args()
    apply = not args.no_apply
    if args.what == "lane":
        return calibrate_lane(apply)
    return calibrate_landmark(args.what, apply)


if __name__ == "__main__":
    sys.exit(main())
