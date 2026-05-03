#!/usr/bin/env bash
set -e

# Mac (Apple Silicon) 开发环境一键搭建
# 参考: https://github.com/unitreerobotics/unitree_sdk2_python/issues/98

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${ROOT_DIR}/.deps"
mkdir -p "${DEPS_DIR}"

echo ">>> [1/5] 检查 Xcode 命令行工具"
if ! xcode-select -p >/dev/null 2>&1; then
  xcode-select --install || true
  echo "请安装完 Xcode 命令行工具后重跑本脚本。"
  exit 1
fi

echo ">>> [2/5] 检查 Homebrew + 系统依赖 (cmake/git/pkg-config)"
if ! command -v brew >/dev/null 2>&1; then
  cat <<EOF
未检测到 Homebrew。请先用下面这条命令装 brew (会让你输 sudo 密码), 然后重跑本脚本:

  /bin/bash -c "\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

装完后, 把 brew 加进 PATH (Apple Silicon Mac):

  echo 'eval "\$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
  source ~/.zshrc

EOF
  exit 1
fi

MISSING=()
for tool in cmake git pkg-config; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    MISSING+=("$tool")
  fi
done
if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "缺少: ${MISSING[*]}, 自动用 brew 安装..."
  brew install "${MISSING[@]}"
fi
cmake --version | head -1
git --version

echo ">>> [3/5] 编译 CycloneDDS 0.10.x"
if [ ! -d "${DEPS_DIR}/cyclonedds/install/lib" ]; then
  cd "${DEPS_DIR}"
  if [ ! -d cyclonedds ]; then
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
  fi
  cd cyclonedds
  mkdir -p build install
  cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_OSX_ARCHITECTURES=arm64
  cmake --build . --target install -j"$(sysctl -n hw.ncpu)"
else
  echo "  跳过: 已存在 ${DEPS_DIR}/cyclonedds/install"
fi
export CYCLONEDDS_HOME="${DEPS_DIR}/cyclonedds/install"
echo "CYCLONEDDS_HOME=${CYCLONEDDS_HOME}"

echo ">>> [4/5] 创建 venv 并安装 Python 依赖"
cd "${ROOT_DIR}"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo ">>> [5/5] 安装 unitree_sdk2_python (可编辑模式)"
cd "${DEPS_DIR}"
if [ ! -d unitree_sdk2_python ]; then
  git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
fi
cd unitree_sdk2_python
CYCLONEDDS_HOME="${CYCLONEDDS_HOME}" pip install -e .

cat <<EOF

============================================================
环境搭建完成 ✓

每次开发前先在终端执行:

  cd ${ROOT_DIR}
  source .venv/bin/activate
  export CYCLONEDDS_HOME=${CYCLONEDDS_HOME}

或在 ~/.zshrc 加一个一键进环境的别名:

  alias cdunitree='cd ${ROOT_DIR} && source .venv/bin/activate && export CYCLONEDDS_HOME=${CYCLONEDDS_HOME}'

Cursor/VSCode 里把 Python 解释器选成:

  ${ROOT_DIR}/.venv/bin/python
============================================================

EOF
