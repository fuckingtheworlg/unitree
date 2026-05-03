#!/usr/bin/env bash
# 在 Mac 上跑, 准备狗端纯离线安装所需的全部资源
# 输出到 .deps/offline_bundle/, 之后用 deploy_offline.sh 推到狗
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${ROOT_DIR}/.deps/offline_bundle"

echo ">>> 检查 venv"
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${ROOT_DIR}/.venv/bin/activate"
    else
        echo "❌ 请先在 Mac 上跑: bash setup_mac.sh"
        exit 1
    fi
fi

echo ">>> 清理旧 bundle"
rm -rf "${BUNDLE_DIR}"
mkdir -p "${BUNDLE_DIR}/wheels"

echo ">>> [1/3] git clone cyclonedds-python 0.10.2 源码"
git clone --depth 1 -b 0.10.2 \
    https://github.com/eclipse-cyclonedds/cyclonedds-python.git \
    "${BUNDLE_DIR}/cyclonedds-python"
rm -rf "${BUNDLE_DIR}/cyclonedds-python/.git"

echo ">>> [2/3] 复制 unitree_sdk2_python 源码"
SDK_SRC="${ROOT_DIR}/.deps/unitree_sdk2_python"
if [ ! -d "${SDK_SRC}" ]; then
    echo "❌ 未找到 ${SDK_SRC}, 请先运行 bash setup_mac.sh"
    exit 1
fi
rsync -a --exclude='.git' --exclude='__pycache__' \
    "${SDK_SRC}/" "${BUNDLE_DIR}/unitree_sdk2_python/"

echo ">>> [3/3] 下载所有依赖 wheel (锁 Python 3.8 最后兼容版本)"
# 狗端 Python 是 3.8.10, 以下版本是各包还兼容 3.8 的最后稳定版:
#   pip 25.0+ / setuptools 76+ / rich 14+ / pygments 2.20+ / typing-extensions 4.14+ / click 8.2+
#   均已 drop Python 3.8, 必须显式锁旧版
# 另外 setuptools 68+ 内部依赖 importlib_metadata.EntryPoints (3.6+ 引入),
# 而 Ubuntu 20.04 自带的 importlib_metadata 是 1.5.x, 必须升到 4+
pip download --no-deps --only-binary :all: \
    --dest "${BUNDLE_DIR}/wheels" \
    'pip==24.3.1' \
    'setuptools==74.1.3' \
    'wheel==0.45.1' \
    'importlib_metadata==8.5.0' \
    'zipp==3.20.2' \
    'packaging==24.2' \
    'rich-click==1.8.5' \
    'click==8.1.8' \
    'rich==13.9.4' \
    'pygments==2.19.2' \
    'typing-extensions==4.13.2' \
    'markdown-it-py==3.0.0' \
    'mdurl==0.1.2'

echo ""
echo "============================================================"
echo "Offline bundle 准备完成 ✓"
echo "  位置: ${BUNDLE_DIR}"
du -sh "${BUNDLE_DIR}"
echo ""
echo "  cyclonedds-python:    $(du -sh ${BUNDLE_DIR}/cyclonedds-python | cut -f1)"
echo "  unitree_sdk2_python:  $(du -sh ${BUNDLE_DIR}/unitree_sdk2_python | cut -f1)"
echo "  wheels:               $(ls ${BUNDLE_DIR}/wheels | wc -l | tr -d ' ') 个"
echo ""
echo "下一步: bash scripts/deploy_offline.sh"
echo "============================================================"
