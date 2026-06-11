"""国赛现场快速标定 (Mac 端跑, 自动连狗).

用法 (Mac, 项目根目录):
  .venv/bin/python scripts/onsite_calibrate.py lane     # 黄线标定: 狗摆在能看到黄道+地板的位置
  .venv/bin/python scripts/onsite_calibrate.py stair    # 台阶牌标定: 狗鼻贴台阶牌 ~25cm
  .venv/bin/python scripts/onsite_calibrate.py dump     # 倾倒区标定: 狗鼻贴倾倒区白圆 ~25cm
  .venv/bin/python scripts/onsite_calibrate.py dock     # 充电区标定: 狗鼻贴充电牌 ~25cm

干什么:
  lane:  录5s鱼眼 → 跑 HSV/LAB 两个检测器 → 出叠加图 + 命中HSV统计 → 给建议
  stair/dump/dock: 抓 D435i 一帧 → 跑 color combo → 打印 w/k/b 占比 + 是否能触发

依赖: sshpass (brew install sshpass), 狗 IP 192.168.123.18, 密码 123
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROBOT = "unitree@192.168.123.18"
PASS = "123"
DDS_ENV = ("export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds && "
           "export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH")


def _ssh(cmd: str, tty: bool = False, timeout: int = 60) -> str:
    args = ["sshpass", "-p", PASS, "ssh"]
    if tty:
        args.append("-tt")
    args += ["-o", "StrictHostKeyChecking=no", ROBOT, cmd]
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def _pull(remote: str, local: str) -> None:
    subprocess.run(["sshpass", "-p", PASS, "rsync", "-a",
                    f"{ROBOT}:{remote}", local], check=True, timeout=60)


def calibrate_lane() -> int:
    import cv2
    import numpy as np
    from src.utils.config import load_config
    from src.vision.yellow_mask import crop_roi, yellow_mask, yellow_mask_lab

    print(">>> 狗端录 5 秒鱼眼 (保持狗静止, 视野里有黄道最好)...")
    out = _ssh(f"cd ~/go2-patrol && {DDS_ENV} && "
               "python3 scripts/record_camera.py --duration 5 --network eth0 "
               "--output recordings/onsite_lane.avi 2>&1 | tail -2", tty=True)
    print(out.strip()[-200:])
    _pull("~/go2-patrol/recordings/onsite_lane.avi", "recordings/")
    print(">>> 拉回完成, 分析中...")

    cfg = load_config("config/params.yaml")
    ranges = [(tuple(it[0]), tuple(it[1])) for it in cfg.vision.yellow_hsv_ranges]
    cap = cv2.VideoCapture("recordings/onsite_lane.avi")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 40
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, f = cap.read()
    cap.release()
    if not ok:
        print("[ERR] 视频读取失败")
        return 2
    roi, _ = crop_roi(f, cfg.camera.roi_top_ratio, cfg.camera.roi_bottom_ratio)

    out_dir = Path("logs/onsite")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%H%M%S")

    for det in ("hsv", "lab"):
        if det == "hsv":
            m = yellow_mask(roi, ranges=ranges, open_kernel=3, close_kernel=7)
        else:
            m = yellow_mask_lab(roi, open_kernel=3, close_kernel=7)
        yp = int((m > 0).sum())
        pct = 100.0 * yp / m.size
        ov = roi.copy()
        ov[m > 0] = (0, 255, 255)
        blend = cv2.addWeighted(roi, 0.45, ov, 0.55, 0)
        cv2.putText(blend, f"{det.upper()}: {pct:.1f}%", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        p = out_dir / f"lane_{det}_{ts}.jpg"
        cv2.imwrite(str(p), blend)
        print(f"  {det.upper():4s}: {pct:5.1f}%  -> {p}")

        hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        px = hsv_img[m > 0]
        if px.size:
            H, S, V = px[:, 0], px[:, 1], px[:, 2]
            print(f"        命中 HSV: H p50={np.percentile(H,50):.0f} "
                  f"S p50={np.percentile(S,50):.0f} V p50={np.percentile(V,50):.0f}")

    print()
    print("人工判断 (打开上面两张 jpg):")
    print("  - 黄道整条亮 + 地板/其他区域干净  -> 该检测器可用")
    print("  - 都有问题 -> 发图给 AI 重新调阈值")
    print("  - 切换检测器: 改 config/params.yaml 的 vision.lane_detector: hsv|lab")
    print("    然后 sshpass -p 123 rsync -a config/params.yaml "
          "unitree@192.168.123.18:~/go2-patrol/config/  并重启 main")
    return 0


def calibrate_landmark(kind: str) -> int:
    import numpy as np
    from src.utils.config import load_config
    from src.vision.realsense_target import classify_color_combo, ColorLabel

    print(f">>> 抓 D435i 一帧 ({kind}: 狗鼻贴牌 ~25cm)...")
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
    out = _ssh(f'cd ~/go2-patrol && python3 -c "{snap_code}" 2>&1 | tail -2')
    print(out.strip()[-200:])
    if "SNAP OK" not in out:
        print("[ERR] D435i 抓帧失败 (检查 USB)")
        return 2
    _pull("/tmp/onsite_snap.npz", "/tmp/")

    cfg = load_config("config/params.yaml")
    ccfg = cfg.realsense.color_stat
    kw = {}
    for key in ("white_min_ratio", "black_min_ratio", "blue_min_ratio",
                "other_max_ratio", "night_white_min_ratio", "night_black_min_ratio",
                "night_blue_min_ratio", "night_other_max_ratio_blackwhite",
                "night_other_max_ratio_bluewhite", "night_blue_dominant_sum",
                "depth_min_m", "depth_max_m"):
        v = getattr(ccfg, key, None)
        if v is not None:
            kw[key] = float(v)
    kw["night_enabled"] = bool(getattr(ccfg, "night_enabled", True))

    data = np.load("/tmp/onsite_snap.npz")
    cls = classify_color_combo(
        data["rgb"], depth_raw=data["depth_raw"],
        depth_scale=float(data["depth_scale"]), require_depth=True, **kw,
    )
    expected = {"dump": "black_white", "stair": "blue_white",
                "dock": "blue_white"}[kind]
    print()
    print(f"  label    = {cls.label.value}   (期望 {expected})")
    print(f"  w={cls.white_ratio:.2f} k={cls.black_ratio:.2f} "
          f"b={cls.blue_ratio:.2f} y={cls.yellow_ratio:.2f}  z={cls.depth_median_m:.2f}m")
    print(f"  rule: {cls.rule_explain}")
    print()
    if cls.label.value == expected:
        print("  ✓ 能正确触发, 不用调")
    else:
        print("  ✗ 识别不到! 把上面 w/k/b 数字发给 AI, 或按规则手动调 yaml:")
        print("    - BLACK_WHITE 需要: w≥night_white_min(0.35) 且 k≥night_black_min(0.30)")
        print("    - BLUE_WHITE  需要: w≥0.35 且 b≥night_blue_min(0.12)")
        print("    - 改完 rsync yaml 上狗 + 重启 main")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["lane", "stair", "dump", "dock"])
    args = parser.parse_args()
    if args.what == "lane":
        return calibrate_lane()
    return calibrate_landmark(args.what)


if __name__ == "__main__":
    sys.exit(main())
