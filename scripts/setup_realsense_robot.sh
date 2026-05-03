#!/usr/bin/env bash
#
# 在狗的 Orin Nano 上执行: 编译安装 librealsense + pyrealsense2 (Python 3.8 binding)
#
# 前提:
#   1) 已经从 Mac 跑过 prepare_realsense_bundle.sh, .deps/realsense/librealsense-X.Y.Z 同步过来了
#   2) 已经从 Mac 跑过 ssh_robot_proxy.sh, http_proxy 已经在当前 shell 环境里
#   3) 项目所在目录 (默认 ~/go2-patrol) 已经有 .venv (Python 3.8)
#
# 执行后会:
#   - apt 装编译依赖 (走代理, 大约 200~500MB)
#   - cmake build librealsense (~25~40 分钟, 不走网)
#   - sudo make install 到 /usr/local
#   - sudo cp udev rules
#   - 把 pyrealsense2*.so 复制到项目 venv 的 site-packages
#   - 跑 enumerate 验证
#
# 用法:
#   bash ~/go2-patrol/scripts/setup_realsense_robot.sh
#   PROJECT_DIR=/home/unitree/go2-patrol bash ./setup_realsense_robot.sh
#   JOBS=2 bash ./setup_realsense_robot.sh   # 防内存不足

set -e

PROJECT_DIR="${PROJECT_DIR:-$HOME/go2-patrol}"
BUNDLE_DIR="${PROJECT_DIR}/.deps/realsense"
JOBS="${JOBS:-4}"

if [ ! -d "$BUNDLE_DIR" ]; then
  echo "[ERR] 找不到 ${BUNDLE_DIR}"
  echo "      请先在 Mac 跑: bash scripts/prepare_realsense_bundle.sh && bash deploy.sh"
  exit 1
fi

LIBRS_DIR="$(find "$BUNDLE_DIR" -maxdepth 1 -type d -name 'librealsense-*' | head -n 1)"
if [ -z "$LIBRS_DIR" ]; then
  echo "[ERR] 找不到 librealsense 源码目录于 $BUNDLE_DIR"
  ls -la "$BUNDLE_DIR" || true
  exit 1
fi
LIBRS_VERSION="$(basename "$LIBRS_DIR" | sed 's/^librealsense-//')"
echo ">>> librealsense 源码: $LIBRS_DIR (v${LIBRS_VERSION})"

# 自动检测 Python 安装模式: venv 优先, 否则 fallback 到 --user (~/.local)
VENV_DIR="${PROJECT_DIR}/.venv"
if [ -x "$VENV_DIR/bin/python" ]; then
  PYTHON_BIN="$VENV_DIR/bin/python"
  PY_VER="$("$PYTHON_BIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
  SITE_PACKAGES="${VENV_DIR}/lib/python${PY_VER}/site-packages"
  INSTALL_MODE="venv"
  echo ">>> Python (venv): $PYTHON_BIN  v$PY_VER"
else
  PYTHON_BIN="$(command -v python3)"
  if [ -z "$PYTHON_BIN" ]; then
    echo "[ERR] 系统找不到 python3"
    exit 1
  fi
  PY_VER="$("$PYTHON_BIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
  SITE_PACKAGES="$("$PYTHON_BIN" -c 'import site;print(site.getusersitepackages())')"
  INSTALL_MODE="user"
  echo ">>> Python (system + --user): $PYTHON_BIN  v$PY_VER"
  echo "    没检测到 ${VENV_DIR}, 假设走 _install_on_robot.sh 的离线 --user 模式"
fi
echo "    site-packages: $SITE_PACKAGES"

echo
echo ">>> [1/7] 检查代理 / 网络 / 时钟"
if [ -z "$http_proxy" ] && [ -z "$HTTP_PROXY" ]; then
  echo "[WARN] 没设 http_proxy 环境变量"
  echo "       如果狗端没外网, 后续 apt 会失败"
  echo "       请确保是用 ssh_robot_proxy.sh 进来的, 或手动:"
  echo "         export http_proxy=http://localhost:8888"
  echo "         export https_proxy=http://localhost:8888"
  read -r -p "       仍然继续? [y/N] " ans
  case "$ans" in
    y|Y|yes|YES) ;;
    *) exit 1;;
  esac
else
  echo "    http_proxy=$http_proxy"
fi

# 时钟检测: 用 HTTP Date 头 (不需要证书验证) 取标准时间, 跟本机比
# 偏差超过 60 秒就提示用户校准 (HTTPS 证书验证依赖正确时钟)
CURL_PROXY_OPT=()
[ -n "$http_proxy" ] && CURL_PROXY_OPT+=(-x "$http_proxy")
WEB_DATE_STR="$(curl -sI --max-time 8 "${CURL_PROXY_OPT[@]}" http://www.baidu.com 2>/dev/null \
  | awk -F': ' '/^[Dd]ate:/ {print $2}' | tr -d '\r' | head -n 1 || true)"
if [ -n "$WEB_DATE_STR" ]; then
  WEB_TS=$(date -d "$WEB_DATE_STR" +%s 2>/dev/null || echo 0)
  LOCAL_TS=$(date +%s)
  DIFF=$((WEB_TS - LOCAL_TS))
  ABS_DIFF=${DIFF#-}
  if [ "$WEB_TS" -gt 0 ] && [ "$ABS_DIFF" -gt 60 ]; then
    echo "[WARN] 系统时钟偏差 ${DIFF}s (网络时间: $WEB_DATE_STR)"
    echo "       这会让 apt HTTPS 全部证书验证失败"
    echo "       自动校准: sudo date -s \"$WEB_DATE_STR\""
    sudo date -s "$WEB_DATE_STR"
    echo "    [ok] 时钟校准: $(date)"
  else
    echo "    [ok] 时钟正常 (差 ${DIFF}s)"
  fi
else
  echo "[WARN] 无法从 HTTP Date 头取时间; 假设时钟正常, 继续"
fi

if curl -fsS --max-time 8 -o /dev/null https://github.com; then
  echo "    [ok] 代理可达 github (https)"
else
  echo "    [WARN] HTTPS 探测失败; apt 可能仍报证书错"
fi

# git config 必须单独设, git 不读 http_proxy 环境变量
# librealsense 的 cmake 阶段会 git clone nlohmann/json 等第三方
if [ -n "$http_proxy" ]; then
  echo "    设置 git 走代理 (cmake 第三方依赖会用)"
  git config --global http.proxy "$http_proxy"
  git config --global https.proxy "${https_proxy:-$http_proxy}"
fi

echo
echo ">>> [2/7] apt 装编译依赖 (走代理)"
APT_ENV=()
[ -n "$http_proxy" ] && APT_ENV+=("Acquire::http::Proxy=$http_proxy")
[ -n "$https_proxy" ] && APT_ENV+=("Acquire::https::Proxy=$https_proxy")
APT_OPTS=()
for kv in "${APT_ENV[@]}"; do APT_OPTS+=("-o" "$kv"); done

sudo apt-get "${APT_OPTS[@]}" update
sudo apt-get "${APT_OPTS[@]}" install -y --no-install-recommends \
  build-essential cmake pkg-config \
  libusb-1.0-0-dev libssl-dev libudev-dev \
  python3-dev python3-setuptools

echo
echo ">>> [3/7] cmake 配置 (Python ${PY_VER}, 关闭 GUI/CUDA, 开启 Python binding)"

# Patch: 让 cmake 阶段的 ExternalProject 用本地 cache 而不是 git clone
# 原因: ssh 反向隧道传 GitHub 大仓库 (>10MB) 不稳, GnuTLS 经常解码失败
CACHE_DIR="${BUNDLE_DIR}/cache"
JSON_DL_FILE="${LIBRS_DIR}/CMake/json-download.cmake.in"

if [ -d "${CACHE_DIR}/nlohmann_json" ] && [ -f "$JSON_DL_FILE" ]; then
  if ! grep -q 'OFFLINE_NLOHMANN_PATCHED' "$JSON_DL_FILE"; then
    echo "    patch json-download.cmake.in -> 用本地 cache/nlohmann_json"
    cp -f "$JSON_DL_FILE" "${JSON_DL_FILE}.orig"
    cat > "$JSON_DL_FILE" <<EOF
# OFFLINE_NLOHMANN_PATCHED by setup_realsense_robot.sh
cmake_minimum_required(VERSION 3.6)
project(nlohmann-json-download NONE)
include(ExternalProject)
ExternalProject_Add(
    nlohmann_json
    PREFIX .
    DOWNLOAD_COMMAND   "\${CMAKE_COMMAND}" -E remove_directory  "\${CMAKE_BINARY_DIR}/third-party/json"
             COMMAND   "\${CMAKE_COMMAND}" -E copy_directory "${CACHE_DIR}/nlohmann_json" "\${CMAKE_BINARY_DIR}/third-party/json"
    DOWNLOAD_DIR       "\${CMAKE_BINARY_DIR}/third-party/"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
EOF
  else
    echo "    json-download.cmake.in 已经 patched (跳过)"
  fi
elif [ ! -d "${CACHE_DIR}/nlohmann_json" ]; then
  echo "[WARN] 没找到 ${CACHE_DIR}/nlohmann_json"
  echo "       cmake 仍会尝试通过代理 git clone, 大概率失败"
  echo "       请在 Mac 端跑: bash scripts/prepare_realsense_bundle.sh && bash deploy.sh"
fi

BUILD_DIR="${LIBRS_DIR}/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DBUILD_EXAMPLES=false \
  -DBUILD_GRAPHICAL_EXAMPLES=false \
  -DBUILD_WITH_OPENMP=true \
  -DBUILD_PYTHON_BINDINGS=true \
  -DPYBIND11_PYTHON_VERSION="${PY_VER}" \
  -DPYTHON_EXECUTABLE="${PYTHON_BIN}" \
  -DBUILD_WITH_CUDA=false \
  -DCHECK_FOR_UPDATES=false

echo
echo ">>> [4/7] make -j${JOBS} (估计 25~40 分钟, 可去喝杯水)"
echo "         如果中途 OOM, 重跑时设 JOBS=2"
START_TS=$(date +%s)
make -j"${JOBS}"
END_TS=$(date +%s)
echo "    编译耗时: $((END_TS - START_TS)) 秒"

echo
echo ">>> [5/7] sudo make install + ldconfig"
sudo make install
sudo ldconfig

echo
echo ">>> [6/7] 安装 udev rules (允许非 root 访问 D435i)"
RULES_FILE="${LIBRS_DIR}/config/99-realsense-libusb.rules"
if [ -f "$RULES_FILE" ]; then
  sudo cp "$RULES_FILE" /etc/udev/rules.d/
  sudo udevadm control --reload-rules
  sudo udevadm trigger
  echo "    udev rules 已安装"
else
  echo "[WARN] 找不到 $RULES_FILE, 跳过 (后续可能要 sudo 才能开相机)"
fi

echo
echo ">>> [7/7] 把 pyrealsense2 .so 复制到 site-packages (${INSTALL_MODE} 模式)"
SO_FILES=$(find "$BUILD_DIR" -name 'pyrealsense2*.so' -not -path '*/CMakeFiles/*' 2>/dev/null || true)
if [ -z "$SO_FILES" ]; then
  echo "[ERR] 没在 $BUILD_DIR 里找到 pyrealsense2*.so"
  echo "      build 可能没启用 python binding, 检查上面 cmake 输出"
  exit 1
fi
mkdir -p "$SITE_PACKAGES"
echo "$SO_FILES" | while read -r f; do
  echo "    cp $f -> $SITE_PACKAGES/"
  cp -v "$f" "$SITE_PACKAGES/"
done

PYBIND_DIR="${LIBRS_DIR}/wrappers/python"
if [ -f "${PYBIND_DIR}/pyrealsense2/__init__.py" ]; then
  cp -rv "${PYBIND_DIR}/pyrealsense2" "$SITE_PACKAGES/" || true
fi

echo
echo ">>> 验证: import + enumerate"
"$PYTHON_BIN" - <<'PY'
import pyrealsense2 as rs
print("[ok] import pyrealsense2")
ctx = rs.context()
devs = list(ctx.query_devices())
print(f"[ok] devices found: {len(devs)}")
for d in devs:
    print(" -", d.get_info(rs.camera_info.name),
          "SN:", d.get_info(rs.camera_info.serial_number),
          "USB:", d.get_info(rs.camera_info.usb_type_descriptor))
PY

cat <<EOF

>>> 全部完成 ✓
    librealsense:  /usr/local/lib/librealsense2.so.${LIBRS_VERSION%.*}
    pyrealsense2:  ${SITE_PACKAGES}/pyrealsense2*.so
    udev rules:    /etc/udev/rules.d/99-realsense-libusb.rules
    install_mode:  ${INSTALL_MODE}

下一步:
    cd ${PROJECT_DIR}
$( [ "$INSTALL_MODE" = "venv" ] && echo "    source .venv/bin/activate" )
    ${PYTHON_BIN} scripts/test_realsense.py --enum-only

如果 import 失败:
    ldconfig -p | grep realsense
    ls ${SITE_PACKAGES}/pyrealsense2*

EOF
