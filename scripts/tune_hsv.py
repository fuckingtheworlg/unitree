"""HSV 阈值现场调参 GUI：滑动条改 H/S/V 上下限，实时看二值化结果。

用法 (Mac 端推荐):
    python -m scripts.tune_hsv tests/data/sample.mp4
    # 调好后, 把控制台打印出来的阈值粘到 config/params.yaml 的 vision 段

也支持图片:
    python -m scripts.tune_hsv tests/data/clip01.jpg

按键:
    [SPACE] 暂停/继续    [s] 打印当前 HSV    [q/ESC] 退出
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _nothing(_x):
    return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    args = parser.parse_args()

    src = args.source
    if not Path(src).exists():
        print(f"找不到: {src}", file=sys.stderr)
        sys.exit(1)

    is_video = src.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

    cv2.namedWindow("ctrl", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ctrl", 480, 280)
    for name, default, max_val in [
        ("H_low", 18, 179), ("H_high", 38, 179),
        ("S_low", 90, 255), ("S_high", 255, 255),
        ("V_low", 90, 255), ("V_high", 255, 255),
        ("open", 3, 21), ("close", 7, 31),
    ]:
        cv2.createTrackbar(name, "ctrl", default, max_val, _nothing)

    cap = cv2.VideoCapture(src) if is_video else None
    static_img = None if is_video else cv2.imread(src)
    paused = False
    last_frame = None

    while True:
        if is_video and not paused:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            last_frame = frame
        else:
            frame = last_frame if last_frame is not None else static_img
        if frame is None:
            break

        h_lo = cv2.getTrackbarPos("H_low", "ctrl")
        h_hi = cv2.getTrackbarPos("H_high", "ctrl")
        s_lo = cv2.getTrackbarPos("S_low", "ctrl")
        s_hi = cv2.getTrackbarPos("S_high", "ctrl")
        v_lo = cv2.getTrackbarPos("V_low", "ctrl")
        v_hi = cv2.getTrackbarPos("V_high", "ctrl")
        ok = max(1, cv2.getTrackbarPos("open", "ctrl") | 1)
        ck = max(1, cv2.getTrackbarPos("close", "ctrl") | 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, np.array([h_lo, s_lo, v_lo], np.uint8),
            np.array([h_hi, s_hi, v_hi], np.uint8),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok)),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)),
        )

        m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        side = np.hstack([frame, m3])
        cv2.putText(
            side, f"H[{h_lo},{h_hi}] S[{s_lo},{s_hi}] V[{v_lo},{v_hi}] o{ok} c{ck}",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
        cv2.imshow("tune_hsv", side)
        key = cv2.waitKey(15) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("s"):
            print(
                f"yellow_hsv_lower: [{h_lo}, {s_lo}, {v_lo}]\n"
                f"yellow_hsv_upper: [{h_hi}, {s_hi}, {v_hi}]\n"
                f"morph_open_kernel: {ok}\nmorph_close_kernel: {ck}"
            )

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
