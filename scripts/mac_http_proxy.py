"""
极简 HTTP / HTTPS 代理 — 给狗端 (Orin Nano) 借 Mac 出网用

无需任何第三方依赖，纯标准库实现。
- 监听本地 :8888
- 支持 HTTP GET/POST/...（明文请求）
- 支持 CONNECT (HTTPS 隧道)
- 多线程，每连接一个线程

启动：
    python3 scripts/mac_http_proxy.py             # 监听 0.0.0.0:8888
    python3 scripts/mac_http_proxy.py --port 1080
    python3 scripts/mac_http_proxy.py --bind 127.0.0.1   # 仅本机访问

退出：Ctrl+C

配合 ssh_robot_proxy.sh 用：ssh -R 8888:localhost:8888 转发到狗端，
狗端 export http_proxy=http://localhost:8888 即可借 Mac 出网。
"""

from __future__ import annotations

import argparse
import logging
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlsplit


CHUNK = 64 * 1024


class ThreadingServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        logging.info("%s - %s", self.client_address[0], fmt % args)

    def _tunnel(self, sock_a: socket.socket, sock_b: socket.socket) -> None:
        # 纯 TCP 双向转发, 用于 CONNECT 隧道 (HTTPS / git+ssl)
        # 注意: select 超时只能用来兜底, 不能因为短暂无数据就断隧道,
        # 否则 git clone 大文件遇到 GitHub 节流时 TLS 会话会被切断 (GnuTLS recv error -9)
        sock_a.setblocking(False)
        sock_b.setblocking(False)
        idle_limit = 600  # 10 分钟两边都没动静才认为对方挂了
        last_active = time.monotonic()
        try:
            while True:
                r, _, x = select.select([sock_a, sock_b], [], [sock_a, sock_b], 30)
                if x:
                    return
                now = time.monotonic()
                if not r:
                    if now - last_active > idle_limit:
                        return
                    continue
                last_active = now
                for s in r:
                    other = sock_b if s is sock_a else sock_a
                    try:
                        data = s.recv(CHUNK)
                    except (BlockingIOError,):
                        continue
                    except (ConnectionResetError, OSError):
                        return
                    if not data:
                        return
                    try:
                        other.sendall(data)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
        finally:
            for s in (sock_a, sock_b):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    s.close()
                except OSError:
                    pass

    def do_CONNECT(self) -> None:
        host, _, port = self.path.partition(":")
        port = int(port or 443)
        try:
            upstream = socket.create_connection((host, port), timeout=15)
        except (socket.gaierror, OSError) as e:
            self.send_error(502, f"upstream connect failed: {e}")
            return
        self.send_response(200, "Connection Established")
        self.send_header("Connection", "close")
        self.end_headers()
        self._tunnel(self.connection, upstream)

    def _do_method(self) -> None:
        url = self.path
        parts = urlsplit(url if "://" in url else "http://" + url)
        host = parts.hostname
        if not host:
            self.send_error(400, "no host in request")
            return
        port = parts.port or 80
        try:
            upstream = socket.create_connection((host, port), timeout=15)
        except (socket.gaierror, OSError) as e:
            self.send_error(502, f"upstream connect failed: {e}")
            return

        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        req_line = f"{self.command} {path} HTTP/1.1\r\n".encode()
        upstream.sendall(req_line)
        for k, v in self.headers.items():
            if k.lower() in ("proxy-connection",):
                continue
            upstream.sendall(f"{k}: {v}\r\n".encode())
        upstream.sendall(b"\r\n")
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            remaining = length
            while remaining > 0:
                buf = self.rfile.read(min(CHUNK, remaining))
                if not buf:
                    break
                upstream.sendall(buf)
                remaining -= len(buf)
        self._tunnel(self.connection, upstream)

    def do_GET(self) -> None:
        self._do_method()

    def do_POST(self) -> None:
        self._do_method()

    def do_PUT(self) -> None:
        self._do_method()

    def do_DELETE(self) -> None:
        self._do_method()

    def do_HEAD(self) -> None:
        self._do_method()

    def do_OPTIONS(self) -> None:
        self._do_method()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bind", default="127.0.0.1",
                   help="监听地址 (默认 127.0.0.1，配 ssh -R 用就够；要让别的机器直连用 0.0.0.0)")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    server = ThreadingServer((args.bind, args.port), ProxyHandler)
    addr = f"{args.bind}:{args.port}"
    print(f"[mac_http_proxy] 监听 {addr}  (Ctrl+C 退出)")
    print(f"[mac_http_proxy] 配合: ssh -R {args.port}:localhost:{args.port} unitree@192.168.123.18")
    print(f"[mac_http_proxy] 狗端: export http_proxy=http://localhost:{args.port} https_proxy=http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mac_http_proxy] 退出")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
