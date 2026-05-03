#!/usr/bin/env bash
set -e

# 部署本工程到 Go2 EDU 的 Orin Nano
#
# 注意:
#   - 默认部署到 ~/go2-patrol/, 跟狗上其他工程隔离
#   - 不使用 rsync --delete, 防止误删狗上其他文件
#   - 用户可通过环境变量覆盖: ROBOT_HOST=x.x.x.x ROBOT_DIR=~/foo bash deploy.sh

ROBOT_USER="${ROBOT_USER:-unitree}"
ROBOT_HOST="${ROBOT_HOST:-192.168.123.18}"
ROBOT_DIR="${ROBOT_DIR:-/home/${ROBOT_USER}/go2-patrol}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> 部署 ${ROOT_DIR}"
echo "    -> ${ROBOT_USER}@${ROBOT_HOST}:${ROBOT_DIR}"
echo "    (不会删除目标目录里其他文件; 仅同步本工程)"
echo

ssh "${ROBOT_USER}@${ROBOT_HOST}" "mkdir -p ${ROBOT_DIR}"

rsync -av \
  --exclude '.venv' \
  --exclude '.deps' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude '.pytest_cache' \
  --exclude 'tests/data' \
  --exclude 'recordings' \
  --exclude 'logs' \
  --exclude '.DS_Store' \
  -e ssh \
  "${ROOT_DIR}/" "${ROBOT_USER}@${ROBOT_HOST}:${ROBOT_DIR}/"

# 单独同步 RealSense 源码 bundle (大件, 单独控制)
if [ -d "${ROOT_DIR}/.deps/realsense" ]; then
  echo
  echo ">>> 检测到 .deps/realsense, 同步 librealsense 源码到狗"
  ssh "${ROBOT_USER}@${ROBOT_HOST}" "mkdir -p ${ROBOT_DIR}/.deps"
  rsync -av --delete \
    --exclude 'build' \
    --exclude '.git' \
    -e ssh \
    "${ROOT_DIR}/.deps/realsense/" "${ROBOT_USER}@${ROBOT_HOST}:${ROBOT_DIR}/.deps/realsense/"
fi

cat <<EOF

>>> 部署完成 ✓ 下一步:

  ssh ${ROBOT_USER}@${ROBOT_HOST}
  cd ${ROBOT_DIR}
  bash setup_robot.sh           # 仅首次需要 (装 cyclonedds + sdk)
  source .venv/bin/activate
  python -m src.main --network eth0 --province

如果要装 RealSense (D435i):
  bash scripts/ssh_robot_proxy.sh                        # Mac 端: 带反向代理 ssh 进狗
  bash ~/go2-patrol/scripts/setup_realsense_robot.sh     # 狗端: 编译安装

EOF
