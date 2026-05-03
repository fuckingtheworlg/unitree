#!/usr/bin/env bash
set -e

# Go2 EDU 内置 Orin Nano (Ubuntu 20.04 aarch64) 端环境搭建
# 仅需在狗上首次部署后跑一次

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${HOME}/.unitree_deps"
mkdir -p "${DEPS_DIR}"

echo ">>> [0/4] 网络可达性检测"
if ping -c 1 -W 2 mirrors.ustc.edu.cn >/dev/null 2>&1 \
   || ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1 \
   || ping -c 1 -W 2 baidu.com >/dev/null 2>&1; then
  echo "  网络 OK"
  HAS_NETWORK=1
else
  HAS_NETWORK=0
  cat <<'EOF'
  ⚠ 狗当前无法访问外网 (DNS/ICMP 都不通).

  原因: 狗的网线接到 Mac, 但 Mac 没开"互联网共享".

  解决方法 (在 Mac 上设置):
    1. 系统设置 → 通用 → 共享 → "互联网共享" 打开
    2. "共享以下来源的连接": 选 Wi-Fi (你 Mac 当前的上网方式)
    3. "用以下端口共享给电脑": 勾上 "USB 10/100/1000 LAN" 或 "雷雳网桥"
       (具体名字看 Mac 接狗用的那条网线类型)
    4. 系统会让你确认开启 NAT, 点确认
    5. 然后狗上重新跑: ping baidu.com 应该能通

  如果你能确认狗上之前已经装过 cyclonedds 0.10.x + unitree_sdk2_python,
  可以加 --skip-network 跳过网络检测继续:
    bash setup_robot.sh --skip-network

EOF
  if [ "${1:-}" != "--skip-network" ]; then
    exit 1
  fi
  echo "  --skip-network 已指定, 继续 (将尝试用现有依赖)"
fi

echo ">>> [1/4] 检测已有 cyclonedds 是否可用"
HAS_CYCLONEDDS=0
if [ -d "${DEPS_DIR}/cyclonedds/install/lib" ]; then
  echo "  发现 ${DEPS_DIR}/cyclonedds/install, 跳过编译"
  HAS_CYCLONEDDS=1
elif command -v find >/dev/null && find / -name "libddsc.so*" 2>/dev/null | head -1 | grep -q .; then
  EXISTING=$(find / -name "libddsc.so*" 2>/dev/null | head -1)
  echo "  系统里已有 cyclonedds: ${EXISTING}"
  echo "  ⚠ 注意: 必须是 0.10.x 版本才能配 unitree_sdk2py 1.0.1"
  echo "  仍会在 ${DEPS_DIR} 编一个独立的 0.10.x 防止版本冲突"
fi

echo ">>> [2/4] 系统依赖 (cmake/python3-venv 等)"
if [ "$HAS_NETWORK" = "1" ]; then
  sudo apt-get update -y || echo "  apt update 部分失败, 尝试继续"
  sudo apt-get install -y \
      cmake g++ build-essential \
      python3-pip python3-venv python3-dev \
      libyaml-cpp-dev libeigen3-dev libboost-all-dev \
      libspdlog-dev libfmt-dev || echo "  某些 apt 包失败, 继续"
else
  echo "  跳过 apt (无网络). 假设 cmake/g++/python3-venv 已经装好."
fi

echo ">>> [3/4] 编译 CycloneDDS 0.10.x (如未装过)"
if [ "$HAS_CYCLONEDDS" = "0" ]; then
  cd "${DEPS_DIR}"
  if [ ! -d cyclonedds ]; then
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
  fi
  cd cyclonedds
  mkdir -p build install
  cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=../install
  cmake --build . --target install -j"$(nproc)"
fi
export CYCLONEDDS_HOME="${DEPS_DIR}/cyclonedds/install"

echo ">>> [4/4] 安装 unitree_sdk2_python + 项目依赖"
cd "${ROOT_DIR}"

VENV_OK=1
if [ ! -d .venv ]; then
  python3 -m venv --system-site-packages .venv 2>/dev/null || VENV_OK=0
fi
if [ "${VENV_OK}" = "1" ] && [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PIP_INSTALL="pip install"
else
  echo "  ⚠ python3-venv 不可用, 退到 pip install --user 模式"
  PIP_INSTALL="pip3 install --user"
fi

${PIP_INSTALL} --upgrade pip 2>/dev/null || true
${PIP_INSTALL} -r requirements.txt

cd "${DEPS_DIR}"
if [ ! -d unitree_sdk2_python ]; then
  git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
fi
cd unitree_sdk2_python
CYCLONEDDS_HOME="${CYCLONEDDS_HOME}" ${PIP_INSTALL} -e .

cat <<EOF

============================================================
狗端环境搭建完成 ✓

每次启动前先在 SSH 终端执行:

  cd ${ROOT_DIR}
  source .venv/bin/activate              # 如果用了 venv
  export CYCLONEDDS_HOME=${CYCLONEDDS_HOME}

或者把这两行写到 ~/.bashrc 末尾.

跑巡线:
  python -m src.main --network eth0 --province
============================================================

EOF
