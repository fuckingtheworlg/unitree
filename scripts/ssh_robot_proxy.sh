#!/usr/bin/env bash
#
# 一键 ssh 进狗 + 把 Mac 的 HTTP 代理通过反向端口转发给狗
#
# 工作原理:
#   1) 检查 Mac 本地 :8888 是否已经在跑 mac_http_proxy.py
#      没跑就 fork 一个后台进程跑起来
#   2) ssh 进狗时加 -R 8888:localhost:8888, 把狗的 :8888 反向转给 Mac 的 :8888
#   3) ssh 进去后自动 export http_proxy / https_proxy / no_proxy
#
# 用法:
#   bash scripts/ssh_robot_proxy.sh
#   ROBOT_HOST=192.168.123.18 PROXY_PORT=8888 bash scripts/ssh_robot_proxy.sh
#
# 退出 ssh 后, 后台代理进程也会自动停 (trap)

set -e

ROBOT_USER="${ROBOT_USER:-unitree}"
ROBOT_HOST="${ROBOT_HOST:-192.168.123.18}"
PROXY_PORT="${PROXY_PORT:-8888}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROXY_SCRIPT="${ROOT_DIR}/scripts/mac_http_proxy.py"
PROXY_PID_FILE="/tmp/mac_http_proxy_${PROXY_PORT}.pid"
PROXY_LOG="/tmp/mac_http_proxy_${PROXY_PORT}.log"

PYTHON="${ROOT_DIR}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

started_proxy=0
existing_pid=""
if [ -f "$PROXY_PID_FILE" ] && kill -0 "$(cat "$PROXY_PID_FILE")" 2>/dev/null; then
  existing_pid="$(cat "$PROXY_PID_FILE")"
  echo ">>> 复用已有代理进程 (pid=$existing_pid, port=$PROXY_PORT)"
elif lsof -nP -iTCP:"$PROXY_PORT" -sTCP:LISTEN 2>/dev/null | grep -q LISTEN; then
  echo ">>> 端口 $PROXY_PORT 已被占用 (非本脚本启动); 假设是别的代理, 直接复用"
else
  echo ">>> 启动 Mac HTTP 代理 (监听 127.0.0.1:$PROXY_PORT)"
  nohup "$PYTHON" "$PROXY_SCRIPT" --port "$PROXY_PORT" --quiet >"$PROXY_LOG" 2>&1 &
  echo $! >"$PROXY_PID_FILE"
  started_proxy=1
  sleep 1
  if ! kill -0 "$(cat "$PROXY_PID_FILE")" 2>/dev/null; then
    echo "[ERR] 代理启动失败, 看日志: $PROXY_LOG"
    exit 1
  fi
  echo "    pid=$(cat "$PROXY_PID_FILE")  log=$PROXY_LOG"
fi

cleanup() {
  if [ "$started_proxy" = "1" ] && [ -f "$PROXY_PID_FILE" ]; then
    pid="$(cat "$PROXY_PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      echo
      echo ">>> 停止 Mac HTTP 代理 (pid=$pid)"
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$PROXY_PID_FILE"
  fi
}
trap cleanup EXIT INT TERM

REMOTE_INIT=$(cat <<EOF
export http_proxy="http://localhost:${PROXY_PORT}"
export https_proxy="http://localhost:${PROXY_PORT}"
export HTTP_PROXY="\$http_proxy"
export HTTPS_PROXY="\$https_proxy"
export no_proxy="localhost,127.0.0.1,192.168.123.0/24,::1"
export NO_PROXY="\$no_proxy"
echo "[robot] http_proxy=\$http_proxy"
echo "[robot] 测试: curl -sS https://www.google.com -o /dev/null -w 'google=%{http_code}\\\n' --max-time 5"
exec bash -l
EOF
)

echo
echo ">>> ssh 进狗 (反向转发 ${PROXY_PORT} → Mac 的 ${PROXY_PORT})"
echo "    狗端会自动 export http_proxy=http://localhost:${PROXY_PORT}"
echo "    退出 ssh 后会自动停掉代理"
echo

ssh -t \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R "${PROXY_PORT}:127.0.0.1:${PROXY_PORT}" \
  "${ROBOT_USER}@${ROBOT_HOST}" \
  "$REMOTE_INIT"
