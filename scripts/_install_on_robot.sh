#!/usr/bin/env bash
# 在 Go2 EDU Orin Nano 上跑 (被 deploy_offline.sh 通过 ssh 远程调用)
# 纯离线安装 cyclonedds Python 绑定 + unitree_sdk2py
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${ROOT_DIR}/offline_bundle"
WHEELS="${BUNDLE_DIR}/wheels"

echo "============================================================"
echo "Go2 Orin Nano 离线安装"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "============================================================"

if [ ! -d "${WHEELS}" ] || [ ! -d "${BUNDLE_DIR}/cyclonedds-python" ]; then
    echo "❌ ${BUNDLE_DIR} 内容不全, 请先在 Mac 上跑 prepare_offline_bundle.sh + deploy_offline.sh"
    exit 1
fi

echo ">>> [0/5] 定位 cyclonedds C 库"
CYCLONEDDS_HOME_DEFAULT="/home/unitree/cyclonedds_ws/install/cyclonedds"
if [ ! -f "${CYCLONEDDS_HOME_DEFAULT}/lib/libddsc.so" ]; then
    echo "❌ 未在 ${CYCLONEDDS_HOME_DEFAULT}/lib 找到 libddsc.so"
    echo "   先确认狗上 cyclonedds 0.10.2 在哪, 改这个变量"
    exit 1
fi
export CYCLONEDDS_HOME="${CYCLONEDDS_HOME_DEFAULT}"
echo "   CYCLONEDDS_HOME=${CYCLONEDDS_HOME}"

echo ""
echo ">>> [1/5] 用本地 wheel 升级核心 build 工具 (pin Py3.8 兼容版)"
# 注意顺序: importlib_metadata 必须先于 setuptools 装好,
# 否则 setuptools 74+ 加载时找不到 metadata.EntryPoints
pip3 install --user --no-index --find-links "${WHEELS}" --upgrade \
    'pip==24.3.1' \
    'zipp==3.20.2' \
    'importlib_metadata==8.5.0' \
    'packaging==24.2' \
    'setuptools==74.1.3' \
    'wheel==0.45.1'

PIP_BIN="${HOME}/.local/bin/pip3"
[ -x "${PIP_BIN}" ] || PIP_BIN=pip3
echo "   pip 升级后版本: $(${PIP_BIN} --version)"

echo ""
echo ">>> [2/5] 装纯 Python 依赖 (rich-click 等)"
${PIP_BIN} install --user --no-index --no-deps --find-links "${WHEELS}" \
    rich-click click rich typing-extensions markdown-it-py pygments mdurl

echo ""
echo ">>> [3/5] 编译 + 装 cyclonedds Python 绑定"
echo "   (会现场编 _ddspy.so, 用狗自带的 gcc + Python 3.8 头文件)"
cd "${BUNDLE_DIR}/cyclonedds-python"
${PIP_BIN} install --user --no-index --no-deps --no-build-isolation .
cd "${ROOT_DIR}"

echo ""
echo ">>> [4/5] 装 unitree_sdk2_python (可编辑模式)"
cd "${BUNDLE_DIR}/unitree_sdk2_python"
${PIP_BIN} install --user --no-index --no-deps --no-build-isolation -e .
cd "${ROOT_DIR}"

echo ""
echo ">>> [5/5] 验证"
python3 - <<PY
import cyclonedds, unitree_sdk2py
print("cyclonedds:", getattr(cyclonedds, "__version__", "?"))
print("unitree_sdk2py 路径:", unitree_sdk2py.__file__)
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
print("SportClient.ClassicWalk:", hasattr(SportClient, "ClassicWalk"))
print("VideoClient.GetImageSample:", hasattr(VideoClient, "GetImageSample"))
print("全部接口可用 ✓")
PY

BASHRC="${HOME}/.bashrc"
if ! grep -q "CYCLONEDDS_HOME=${CYCLONEDDS_HOME_DEFAULT}" "${BASHRC}" 2>/dev/null; then
    {
      echo ""
      echo "# unitree go2 巡线工程: cyclonedds C 库路径"
      echo "export CYCLONEDDS_HOME=${CYCLONEDDS_HOME_DEFAULT}"
    } >> "${BASHRC}"
    echo ""
    echo "已写入 ${BASHRC}: export CYCLONEDDS_HOME"
fi

cat <<EOF

============================================================
狗端离线安装完成 ✓

下次新开 SSH 会话, 跑代码:

  cd ${ROOT_DIR}
  python3 -m src.main --network eth0 --province

(CYCLONEDDS_HOME 已写到 ~/.bashrc, 自动生效)
============================================================

EOF
