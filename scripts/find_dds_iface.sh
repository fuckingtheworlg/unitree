#!/usr/bin/env bash
# 在 Mac 上找出"接狗的那张网线网卡名" (用于 ChannelFactoryInitialize)
#
# 原理: 找一张 IP 在 192.168.123.x 段的网卡, 那就是接狗的
# 用法: bash scripts/find_dds_iface.sh

set -e

echo ">>> 寻找 Mac 上接狗 (192.168.123.x) 的网卡..."
echo

found=""
while IFS= read -r line; do
    iface=$(echo "$line" | awk '{print $1}' | tr -d ':')
    [ -z "$iface" ] && continue
    ip_info=$(ifconfig "$iface" 2>/dev/null | grep "inet " | awk '{print $2}' || true)
    if echo "$ip_info" | grep -q '^192\.168\.123\.'; then
        printf "  ✓ %-10s IP=%s  ← 这张接了狗\n" "$iface" "$ip_info"
        found="$iface"
    fi
done < <(ifconfig -l | tr ' ' '\n' | sed 's/$/:/')

echo

if [ -z "$found" ]; then
    cat <<EOF
❌ 没找到 192.168.123.x 段的网卡.
   - 检查网线是否插好
   - 系统设置 → 网络 → 接狗的网线 → 详细信息 → TCP/IP →
     IP 设成 192.168.123.222, 子网掩码 255.255.255.0
   - 然后 ping 192.168.123.18 验证连通
EOF
    exit 1
fi

echo "🎉 你的 DDS 网卡是: $found"
echo
echo "在 Mac 上跑实时巡线视图 (狗不会动):"
echo
echo "  cd /Users/binggo/unitree"
echo "  source .venv/bin/activate"
echo "  export CYCLONEDDS_HOME=\"\$(pwd)/.deps/cyclonedds/install\""
echo "  python -m src.main --network $found --province --dry-run"
echo
