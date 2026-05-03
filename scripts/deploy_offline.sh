#!/usr/bin/env bash
# 在 Mac 上跑: 推送项目代码 + offline_bundle 到狗, 然后远程触发狗端离线安装
set -e

ROBOT_USER="${ROBOT_USER:-unitree}"
ROBOT_HOST="${ROBOT_HOST:-192.168.123.18}"
ROBOT_DIR="${ROBOT_DIR:-/home/${ROBOT_USER}/go2-patrol}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${ROOT_DIR}/.deps/offline_bundle"

if [ ! -d "${BUNDLE_DIR}/cyclonedds-python" ]; then
    echo "❌ 未找到 ${BUNDLE_DIR}/cyclonedds-python"
    echo "   请先在 Mac 上跑: bash scripts/prepare_offline_bundle.sh"
    exit 1
fi

echo ">>> 部署目标: ${ROBOT_USER}@${ROBOT_HOST}:${ROBOT_DIR}"

ssh "${ROBOT_USER}@${ROBOT_HOST}" "mkdir -p ${ROBOT_DIR}"

echo ">>> [1/3] 同步项目代码 (~76KB)"
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

echo ""
echo ">>> [2/3] 同步 offline_bundle (--delete: 让狗端跟 Mac 完全一致, 清掉旧版本 wheel)"
rsync -av --delete --exclude='__pycache__' \
    -e ssh \
    "${BUNDLE_DIR}/" "${ROBOT_USER}@${ROBOT_HOST}:${ROBOT_DIR}/offline_bundle/"

echo ""
echo ">>> [3/3] 远程触发狗端离线安装"
ssh -t "${ROBOT_USER}@${ROBOT_HOST}" "cd ${ROBOT_DIR} && bash scripts/_install_on_robot.sh"
