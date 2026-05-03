"""极简 MJPEG HTTP server 用于 Mac 浏览器实时看狗端 debug 画面."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional

import cv2
import numpy as np


_INDEX_HTML = b"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>Go2 Patrol Live</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:sans-serif; }
  .wrap { padding:8px; }
  .meta { font-size:12px; color:#aaa; padding:4px 0; }
  img { display:block; max-width:100%; height:auto; }
</style></head>
<body><div class='wrap'>
  <div class='meta'>Go2 Patrol live stream</div>
  <img src='/stream' />
</div></body></html>
"""


class _State:
    lock = threading.Lock()
    latest_jpeg: Optional[bytes] = None
    last_update_ts: float = 0.0


class _Handler(BaseHTTPRequestHandler):
    server_version = "Go2MJPEG/1.0"

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(_INDEX_HTML)))
            self.end_headers()
            self.wfile.write(_INDEX_HTML)
            return

        if self.path.startswith("/snapshot"):
            with _State.lock:
                jpeg = _State.latest_jpeg
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

        if self.path != "/stream":
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
                    jpeg = _State.latest_jpeg
                    ts = _State.last_update_ts
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

    def push_frame(self, img: np.ndarray) -> None:
        if img is None:
            return
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
            _State.latest_jpeg = buf.tobytes()
            _State.last_update_ts = time.monotonic()
