#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/home/unitree/cyclonedds_ws/install/cyclonedds}"
export LD_LIBRARY_PATH="${CYCLONEDDS_HOME}/lib:${LD_LIBRARY_PATH:-}"

network="${GO2_NETWORK:-eth0}"
mjpeg_port="${MJPEG_PORT:-8088}"

exec python3 -m src.main \
  --network "${network}" \
  --province \
  --no-display \
  --realsense \
  --mjpeg-port "${mjpeg_port}" \
  "$@"
