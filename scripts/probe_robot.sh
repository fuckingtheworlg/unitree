#!/usr/bin/env bash
# 在 Mac 上跑, 通过 SSH 远程探测狗 (Orin Nano) 当前的依赖现状
# 用法:
#   bash scripts/probe_robot.sh
#   ROBOT_HOST=192.168.123.18 ROBOT_USER=unitree bash scripts/probe_robot.sh
set -e

ROBOT_USER="${ROBOT_USER:-unitree}"
ROBOT_HOST="${ROBOT_HOST:-192.168.123.18}"

echo "=========================================="
echo "Probing ${ROBOT_USER}@${ROBOT_HOST}"
echo "=========================================="

ssh -o ConnectTimeout=5 "${ROBOT_USER}@${ROBOT_HOST}" 'bash -s' <<'REMOTE'
set +e

print_section() { echo; echo "### $1"; echo "------------------------------------------"; }

print_section "0. 系统 / 内核"
uname -a
cat /etc/os-release | head -3

print_section "1. 网卡 (DDS 要用的网卡名)"
ip -br addr 2>/dev/null || ifconfig

print_section "2. 外网可达性"
ping -c 1 -W 2 baidu.com >/dev/null 2>&1 && echo "ping baidu.com   : OK" || echo "ping baidu.com   : FAIL"
ping -c 1 -W 2 8.8.8.8    >/dev/null 2>&1 && echo "ping 8.8.8.8     : OK" || echo "ping 8.8.8.8     : FAIL"

print_section "3. Python 环境"
which python3 && python3 --version
which python  2>/dev/null && python --version 2>/dev/null
which pip3    && pip3 --version 2>/dev/null

print_section "4. 关键 Python 库 (系统级)"
for pkg in numpy cv2 yaml cyclonedds unitree_sdk2py; do
    python3 -c "import $pkg; print('$pkg', getattr($pkg, '__version__', 'ok'))" 2>&1 \
      | head -1 \
      | sed "s/^/  /"
done

print_section "5. cyclonedds C 库"
echo "环境变量 CYCLONEDDS_HOME = ${CYCLONEDDS_HOME:-(未设置)}"
echo "---"
echo "系统中所有 libddsc.so 位置:"
find / -name "libddsc.so*" 2>/dev/null | head -10
echo "---"
echo "如果 CYCLONEDDS_HOME 已设, 检查它的版本:"
if [ -n "${CYCLONEDDS_HOME}" ] && [ -f "${CYCLONEDDS_HOME}/lib/cyclonedds_version.h" ]; then
    grep -E "VERSION|MAJOR|MINOR|PATCH" "${CYCLONEDDS_HOME}/lib/cyclonedds_version.h" 2>/dev/null | head -10
else
    echo "  (未找到版本头文件)"
fi

print_section "6. Unitree SDK 相关"
echo "系统是否能 import unitree_sdk2py:"
python3 -c "import unitree_sdk2py; print('  OK, location:', unitree_sdk2py.__file__)" 2>&1 | head -3
echo "---"
echo "本机已 clone 的 sdk 仓库:"
find ~ -maxdepth 5 -type d -name "unitree_sdk2_python" 2>/dev/null | head -5
find ~ -maxdepth 5 -type d -name "unitree_sdk2" 2>/dev/null | head -5
echo "---"
echo "项目是否已部署:"
ls -la ~/go2-patrol 2>/dev/null | head -5
ls -la ~/unitree    2>/dev/null | head -5

print_section "7. ROS 环境 (登录时被问到的 foxy/noetic)"
echo "/opt/ros 子目录:"
ls /opt/ros 2>/dev/null
echo "---"
echo "ROS_DISTRO = ${ROS_DISTRO:-(未设置)}"

print_section "8. 关键开发工具"
for cmd in gcc g++ cmake git make rsync ssh; do
    if command -v $cmd >/dev/null 2>&1; then
        printf "  %-8s : %s\n" "$cmd" "$(command -v $cmd)"
    else
        printf "  %-8s : NOT FOUND\n" "$cmd"
    fi
done

print_section "9. 磁盘空间 (HOME)"
df -h "${HOME}" | tail -1

echo
echo "=========================================="
echo "探测完成. 把以上完整输出复制给开发者诊断."
echo "=========================================="
REMOTE
