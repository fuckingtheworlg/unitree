"""极简 MJPEG HTTP server 用于 Mac 浏览器实时看狗端识别画面."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


_INDEX_HTML = b"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>Go2 Patrol Live</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:sans-serif; }
  .wrap { padding:8px; }
  .meta { font-size:12px; color:#aaa; padding:4px 0; }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap:10px; }
  .card { background:#181818; border:1px solid #333; padding:6px; }
  .title { font-size:14px; color:#fff; margin-bottom:4px; }
  img { display:block; width:100%; height:auto; }
</style></head>
<body><div class='wrap'>
  <div class='meta'>Go2 Patrol live streams: /debug /front /realsense</div>
  <div class='grid'>
    <div class='card'><div class='title'>debug combined</div><img src='/debug' /></div>
    <div class='card'><div class='title'>front fisheye recognition</div><img src='/front' /></div>
    <div class='card'><div class='title'>D435i recognition</div><img src='/realsense' /></div>
  </div>
</div></body></html>
"""


class _State:
    lock = threading.Lock()
    streams: Dict[str, Tuple[Optional[bytes], float]] = {
        "debug": (None, 0.0),
        "front": (None, 0.0),
        "realsense": (None, 0.0),
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "Go2MJPEG/1.0"

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(_INDEX_HTML)))
            self.end_headers()
            self.wfile.write(_INDEX_HTML)
            return

        if path.startswith("/snapshot"):
            stream = "debug"
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 2:
                stream = _normalize_stream_name(parts[1])
            with _State.lock:
                jpeg, _ = _State.streams.get(stream, (None, 0.0))
            if jpeg is None:
                self.send_error(503, "no frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
            return

        stream = _path_to_stream(path)
        if stream is None:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Cache-Control", "no-store, no-cache")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()

        last_sent_ts = 0.0
        try:
            while True:
                with _State.lock:
                    jpeg, ts = _State.streams.get(stream, (None, 0.0))
                if jpeg is None or ts <= last_sent_ts:
                    time.sleep(0.02)
                    continue
                last_sent_ts = ts
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
        except Exception:
            return


def _normalize_stream_name(name: str) -> str:
    name = (name or "debug").strip("/").lower()
    if name in ("stream", "debug"):
        return "debug"
    if name in ("front", "fisheye"):
        return "front"
    if name in ("realsense", "rs", "d435i"):
        return "realsense"
    return name


def _path_to_stream(path: str) -> Optional[str]:
    if path == "/stream":
        return "debug"
    if path in ("/debug", "/front", "/realsense"):
        return _normalize_stream_name(path)
    return None


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MjpegServer:
    def __init__(self, port: int = 8088, host: str = "0.0.0.0",
                 jpeg_quality: int = 75, max_width: int = 0):
        self.port = port
        self.host = host
        self.jpeg_quality = max(10, min(95, int(jpeg_quality)))
        self.max_width = max(0, int(max_width))
        self._httpd: Optional[_ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._httpd is not None:
            return
        self._httpd = _ThreadingHTTPServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="mjpeg-server", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is None:
            return
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:
            pass
        self._httpd = None

    def push_frame(self, img: np.ndarray, stream: str = "debug") -> None:
        if img is None:
            return
        stream = _normalize_stream_name(stream)
        out = img
        if self.max_width > 0 and img.shape[1] > self.max_width:
            ratio = self.max_width / float(img.shape[1])
            out = cv2.resize(
                img, (self.max_width, int(img.shape[0] * ratio)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(
            ".jpg", out,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return
        with _State.lock:
            _State.streams[stream] = (buf.tobytes(), time.monotonic())
