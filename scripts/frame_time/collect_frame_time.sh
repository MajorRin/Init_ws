#!/bin/bash
# =============================================================
# 采集一次动态初始化回放数据, 产出 Log/Li_D_init_frame_time.txt
# 用法: ./collect_frame_time.sh [bag路径] [起始偏移秒=155] [播放时长秒=15]
#
# 与 scripts/run_analysis.sh 的差异(均为有意为之):
#   1) rviz:=false —— rviz 会订阅被发布的话题并抢 CPU, 会污染耗时测量
#      (实测开关 rviz 对轮次总耗时影响约 5%, 但白拿的干净度没理由不要)
#   2) 先起 roscore 再设 use_sim_time, 保证节点启动时就读到该参数
#      —— run_analysis.sh 在无 roscore 时 `rosparam set` 会失败, 节点退回墙钟
#   3) rosbag play 结束后多等 10s, 让最后一轮回放跑完再 kill
#   4) 不自动拉起交互式仪表盘(会阻塞脚本)
#
# 采集完用同目录的 analyze_init_frame_time.py 出图:
#   python3 analyze_init_frame_time.py              # 单窗口仪表盘
#   python3 analyze_init_frame_time.py --no-show    # 无界面导出 PNG/CSV
# =============================================================
set -u

BAG="${1:-/home/mao/rosbag/HZ_LiDAR_Dataset/AVIA/rosbag_all_19700101_080314.bag}"
START_OFFSET="${2:-25}"     # rosbag play -s, 跳过 bag 开头
DUR="${3:-20}"               # rosbag play --duration
WS=/home/mao/Init_ws
LOG=$WS/src/D-LI-Init/Log
RUN=/tmp/dli_run_$(date +%Y%m%d_%H%M%S)

if [ ! -f "$BAG" ]; then
    echo "用法: $0 [bag路径] [起始偏移秒=155] [播放时长秒=15]"
    echo "找不到 bag: $BAG"
    exit 1
fi

mkdir -p "$RUN"
source /opt/ros/noetic/setup.bash
source $WS/devel/setup.bash

echo "=== 1. 启动 roscore ==="
roscore > "$RUN/roscore.log" 2>&1 &
ROSCORE_PID=$!
sleep 4
rosparam set use_sim_time true
echo "    use_sim_time = $(rosparam get use_sim_time)"

echo "=== 2. 启动节点 ==="
roslaunch lidar_imu_init livox_avia.launch rviz:=false > "$RUN/launch.log" 2>&1 &
LAUNCH_PID=$!
sleep 8

echo "=== 3. 回放 bag: -s $START_OFFSET --duration $DUR ==="
rosbag play -s "$START_OFFSET" --duration "$DUR" --clock "$BAG" > "$RUN/play.log" 2>&1
echo "    rosbag play 退出码=$?"

echo "=== 4. 等待最后一轮回放跑完 (10s) ==="
sleep 10

echo "=== 5. 停止节点 ==="
kill "$LAUNCH_PID" 2>/dev/null
sleep 3
kill -9 "$LAUNCH_PID" 2>/dev/null
kill "$ROSCORE_PID" 2>/dev/null
sleep 2
kill -9 "$ROSCORE_PID" 2>/dev/null
pkill -f 'roslaunch lidar_imu_init' 2>/dev/null
pkill -f 'lidar_imu_init/lidar_imu_init' 2>/dev/null
rosparam set use_sim_time false 2>/dev/null || true

echo "=== 6. 产物检查 ==="
for f in Li_D_init_frame_time.txt Li_D_init_iter_time.txt Li_D_init_lio_time.txt Li_D_init_fastlio.txt Li_D_init_lag.txt; do
    if [ -s "$LOG/$f" ]; then
        echo "  [OK] $f ($(wc -l < "$LOG/$f") 行)"
    else
        echo "  [缺失/空] $f"
    fi
done
# 每轮帧数 = launch 里的 Data_accum_length(当前 22), 轮数取决于迭代是否收敛, 故不硬编码
ACC=$(grep -oP 'name="Data_accum_length"[^>]*value="\K[0-9]+' "$WS/src/D-LI-Init/launch/livox_avia.launch" 2>/dev/null | head -1)
echo "  预期 Li_D_init_frame_time.txt = 1 表头 + 轮数×${ACC:-?} 行 (Data_accum_length=${ACC:-未知})"
echo "  注意: 初始化未收敛时节点会一直加轮, 采集窗口内可能跑不完 -> 无 Li_D_init_lag.txt 数据行"
echo
echo "=== 7. 初始化完成时刻的系统滞后 ==="
# 列: t_frame t_sim_now lag_ms pass_id converged
if [ -s "$LOG/Li_D_init_lag.txt" ]; then
    cat "$LOG/Li_D_init_lag.txt"
else
    echo "  [缺失/空] Li_D_init_lag.txt"
fi
echo
echo "=== 运行日志目录: $RUN ==="
tail -5 "$RUN/launch.log"
