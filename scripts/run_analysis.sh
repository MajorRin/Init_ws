#!/bin/bash
# =============================================================
# D-LI-Init 数据采集 + 分析 一键脚本
# 用法: ./run_analysis.sh <bag路径> [launch文件名(默认 livox_avia.launch)]
#
# 说明: RTK 速度 (/dji_osdk_ros/rtk_velocity) 已由 C++ 节点直接订阅,
#       随主日志 Li_D_init_fastlio.txt 一同记录, 无需独立录制。
# 产物 (src/D-LI-Init/Log/):
#   Li_D_init_fastlio.txt   15列主日志 (LIO速度/零偏/重力/flag/RTK最近邻)
#   Li_D_init_iter_time.txt 动态初始化每轮纯计算耗时（仪表盘耗时页数据源）
#   Li_D_init_lio_time.txt  初始化完成后LIO模式每帧分项耗时（保留日志，不纳入仪表盘）
# =============================================================
set -u

BAG="${1:-}"
LAUNCH="${2:-livox_avia.launch}"
PLAY_DUR="${PLAY_DUR:-20}"   # bag 固定播放时长(秒), 可 export PLAY_DUR=xx 覆盖
WS=/home/mao/Init_ws
PKG=$WS/src/D-LI-Init
LOG_DIR=$PKG/Log

if [ -z "$BAG" ] || [ ! -f "$BAG" ]; then
    echo "用法: $0 <bag路径> [launch文件名(默认 livox_avia.launch)]"
    exit 1
fi

source /opt/ros/noetic/setup.bash
source $WS/devel/setup.bash
mkdir -p "$LOG_DIR"

LAUNCH_PID=""
cleanup() {
    echo ""
    echo "=== 清理后台进程 ==="
    [ -n "$LAUNCH_PID" ] && kill "$LAUNCH_PID" 2>/dev/null
    sleep 2
    rosparam set use_sim_time false 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== 1. 备份旧日志 ==="
for f in Li_D_init_fastlio.txt Li_D_init_iter_time.txt Li_D_init_lio_time.txt Li_D_init_frame_time.txt time_result.txt; do
    if [ -f "$LOG_DIR/$f" ]; then
        mv "$LOG_DIR/$f" "$LOG_DIR/$f.bak_$(date +%Y%m%d_%H%M%S)"
        echo "  已备份 $f"
    fi
done

echo "=== 2. 启动 D-LI-Init ($LAUNCH, use_sim_time) ==="
rosparam set use_sim_time true
roslaunch lidar_imu_init "$LAUNCH" use_sim_time:=true &
LAUNCH_PID=$!
sleep 6

echo "=== 3. 回放 bag: 固定播放 ${PLAY_DUR}s 后自动停止 (RTK 已由节点内订阅记录) ==="
# --duration: 从起始点 (-s 起始偏移) 起播放 PLAY_DUR 秒后自动结束
# 若初始化本身需要 > ${PLAY_DUR}s 数据, 请增大 PLAY_DUR (或 export PLAY_DUR=xx)
rosbag play -s 250 --duration "$PLAY_DUR" --clock "$BAG"

echo ""
echo "=== 4. 停止节点 ==="
kill "$LAUNCH_PID" 2>/dev/null
wait "$LAUNCH_PID" 2>/dev/null
LAUNCH_PID=""

echo "=== 5. 日志检查 ==="
for f in Li_D_init_fastlio.txt Li_D_init_iter_time.txt; do
    if [ -s "$LOG_DIR/$f" ]; then
        echo "  [OK] $f ($(wc -l < "$LOG_DIR/$f") 行)"
    else
        echo "  [缺失/空] $f"
    fi
done
if [ -s "$LOG_DIR/Li_D_init_lio_time.txt" ]; then
    echo "  [信息] Li_D_init_lio_time.txt 已生成（正式 LIO 耗时，不纳入本次仪表盘）"
fi

echo ""
echo "=== 6. 启动分析仪表盘 ==="
python3 "$PKG/scripts/analyze_dashboard.py"