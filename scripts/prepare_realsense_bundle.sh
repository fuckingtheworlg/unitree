#!/usr/bin/env bash
#
# Mac 端: 下载 librealsense 源码 tarball 到 .deps/realsense/
# 之后用 deploy.sh rsync 到狗的 ~/go2-patrol/.deps/realsense/, 狗端纯本地编译, 不走代理
#
# 用法:
#   bash scripts/prepare_realsense_bundle.sh
#   LIBRS_VERSION=2.57.7 bash scripts/prepare_realsense_bundle.sh

set -e

LIBRS_VERSION="${LIBRS_VERSION:-2.57.7}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${ROOT_DIR}/.deps/realsense"
TARBALL="${BUNDLE_DIR}/librealsense-${LIBRS_VERSION}.tar.gz"
EXTRACT_DIR="${BUNDLE_DIR}/librealsense-${LIBRS_VERSION}"

URL_PRIMARY="https://github.com/realsenseai/librealsense/archive/refs/tags/v${LIBRS_VERSION}.tar.gz"
URL_FALLBACK="https://github.com/IntelRealSense/librealsense/archive/refs/tags/v${LIBRS_VERSION}.tar.gz"

mkdir -p "$BUNDLE_DIR"

if [ -f "$TARBALL" ] && [ -d "$EXTRACT_DIR" ]; then
  echo ">>> 已存在: $EXTRACT_DIR (跳过下载)"
  echo "    需要重新下载请先 rm -rf $BUNDLE_DIR"
else
  echo ">>> 下载 librealsense v${LIBRS_VERSION} 源码..."
  if ! curl -fL --retry 3 --retry-delay 2 -o "$TARBALL" "$URL_PRIMARY"; then
    echo ">>> 主源失败, 尝试 fallback: IntelRealSense org"
    curl -fL --retry 3 --retry-delay 2 -o "$TARBALL" "$URL_FALLBACK"
  fi
  echo ">>> 解压..."
  tar -xzf "$TARBALL" -C "$BUNDLE_DIR"
fi

VERSION_FILE="${BUNDLE_DIR}/VERSION"
echo "$LIBRS_VERSION" > "$VERSION_FILE"

# 预下载 cmake 阶段会 git clone 的第三方源
# 这些通过 ssh 反向隧道下载不稳定 (TLS 包损坏), Mac 直连快得多
CACHE_DIR="${BUNDLE_DIR}/cache"
mkdir -p "$CACHE_DIR"

declare -a EXTRA_REPOS=(
  # 格式: "本地目录名|git URL|tag/branch"
  "nlohmann_json|https://github.com/nlohmann/json.git|v3.12.0"
)

for repo_spec in "${EXTRA_REPOS[@]}"; do
  IFS='|' read -r local_name repo_url repo_ref <<< "$repo_spec"
  target_dir="${CACHE_DIR}/${local_name}"
  if [ -d "$target_dir/.git" ]; then
    echo ">>> 已缓存: ${local_name} (${repo_ref})"
    continue
  fi
  echo ">>> 缓存 ${local_name} (${repo_url} @ ${repo_ref})"
  rm -rf "$target_dir"
  git clone --depth 1 --branch "$repo_ref" -c advice.detachedHead=false \
    "$repo_url" "$target_dir"
done

SIZE_HUMAN="$(du -sh "$BUNDLE_DIR" | awk '{print $1}')"
echo
echo ">>> 完成 ✓"
echo "    源码:    $EXTRACT_DIR"
echo "    缓存:    $CACHE_DIR"
echo "    总大小:  $SIZE_HUMAN"
echo "    版本号:  $(cat "$VERSION_FILE")"
echo
echo ">>> 下一步:"
echo "    bash deploy.sh                                    # rsync 工程+源码+cache 到狗"
echo "    bash scripts/ssh_robot_proxy.sh                   # ssh 进狗 (带 HTTP 代理)"
echo "    # 狗端: bash ~/go2-patrol/scripts/setup_realsense_robot.sh"
