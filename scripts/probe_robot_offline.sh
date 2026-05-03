#!/usr/bin/env bash
# 二次探测: 找狗上是否已有 cyclonedds-python 源码, 以及离线安装的关键资源
set -e

ROBOT_USER="${ROBOT_USER:-unitree}"
ROBOT_HOST="${ROBOT_HOST:-192.168.123.18}"

ssh -o ConnectTimeout=5 "${ROBOT_USER}@${ROBOT_HOST}" 'bash -s' <<'REMOTE'
set +e

echo "### A. /home/unitree 顶层"
ls -la /home/unitree/ | head -30

echo
echo "### B. cyclonedds_ws 内容 (关键)"
ls -la /home/unitree/cyclonedds_ws/ 2>/dev/null
echo "---"
echo "### B-子: cyclonedds_ws/src 内容 (寻找 cyclonedds-python)"
ls -la /home/unitree/cyclonedds_ws/src/ 2>/dev/null

echo
echo "### C. 在 ~ 整个搜 cyclonedds-python 源码"
find /home/unitree -maxdepth 5 -type d \( -name "cyclonedds-python" -o -name "cyclonedds_python" \) 2>/dev/null

echo
echo "### D. 搜 unitree_sdk2_python 源码"
find /home/unitree -maxdepth 5 -type d -name "unitree_sdk2_python" 2>/dev/null

echo
echo "### E. python3-dev 是否装好 (本地 pip install 编译需要)"
dpkg -l python3-dev 2>/dev/null | tail -1
dpkg -l python3.8-dev 2>/dev/null | tail -1
ls /usr/include/python3.8/Python.h 2>/dev/null && echo "Python.h 头文件 OK" || echo "Python.h 头文件 缺失"

echo
echo "### F. 检查 cyclonedds_ws 编译产物里的 cyclonedds-python (前任工程可能已编 .so)"
find /home/unitree/cyclonedds_ws -name "_ddspy*.so" 2>/dev/null
find /home/unitree -maxdepth 5 -name "cyclonedds*.dist-info" -type d 2>/dev/null

echo
echo "### G. 系统 site-packages 中是否藏着 cyclonedds 的部分文件"
find /usr/lib/python3 -name "cyclonedds*" 2>/dev/null
find /usr/local/lib/python3.8 -name "cyclonedds*" 2>/dev/null
find /home/unitree/.local/lib -name "cyclonedds*" 2>/dev/null

echo
echo "### H. /home/unitree/.bashrc 里有没有相关环境变量"
grep -E "CYCLONEDDS_HOME|unitree_sdk|PYTHONPATH" /home/unitree/.bashrc 2>/dev/null

echo
echo "### I. 当前 pip 已装的包 (snapshot)"
pip3 list 2>/dev/null | grep -iE "cyclonedds|unitree|numpy|opencv" 
REMOTE
