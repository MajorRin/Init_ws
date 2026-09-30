#!/usr/bin/env bash
# =============================================================================
# run_dual_launch.sh - 同时启动 D-LI-Init 与 SFAST_LIO 两个节点
#
# 用途:
#   一键启动 D-LI-Init 初始化节点和 SFAST_LIO 定位/建图节点,
#   分别保存日志到 /tmp, Ctrl-C 或 kill 时自动结束两个进程组。
#
# 用法:
#   ./run_dual_launch.sh                      # 两个 launch 均以默认参数启动
#   ./run_dual_launch.sh --init-args "rviz:=true"
#   ./run_dual_launch.sh --lio-args "rviz:=true use_sim:=false"
#   ./run_dual_launch.sh --init-args "..." --lio-args "..."
#
# 选项:
#   --init-args "..."   传递给 livox_avia.launch 的 roslaunch 参数
#   --lio-args  "..."   传递给 mapping_avia.launch 的 roslaunch 参数
#   --lio-quiet         终端只实时显示 D-LI-Init 输出 (SFAST_LIO 逐帧输出较刷屏时使用)
#   -h, --help          显示帮助
# =============================================================================
set -u

# ----------------------------- 路径配置 --------------------------------------
WS_INIT="/home/mao/Init_ws"
WS_LIO="/home/mao/S_FAST_LIO"

LAUNCH_INIT="lidar_imu_init livox_avia.launch"          # D-LI-Init (pkg + launch)
LAUNCH_LIO="sfast_lio mapping_avia.launch"              # SFAST_LIO  (pkg + launch)

LOG_DIR="/tmp/dual_launch_logs"
mkdir -p "$LOG_DIR"

# ----------------------------- 参数解析 --------------------------------------
INIT_ARGS=""
LIO_ARGS=""
LIO_QUIET=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --init-args) INIT_ARGS="$2"; shift 2 ;;
        --lio-args)  LIO_ARGS="$2";  shift 2 ;;
        --lio-quiet) LIO_QUIET=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,2\}//'
            exit 0 ;;
        *)
            echo "[ERROR] 未知参数: $1 (使用 -h 查看帮助)"
            exit 1 ;;
    esac
done

# ----------------------------- 环境加载 --------------------------------------
# 注意: 两个工作区均为独立编译 (parent 链只含 /opt/ros/noetic), 
# 依次 source 会互相覆盖 ROS_PACKAGE_PATH/CMAKE_PREFIX_PATH。
# 因此每个 roslaunch 在独立子 shell 中启动, 各自只加载自己工作区的环境,
# 两者仍通过默认 ROS master (localhost:11311) 相互通信。
check_ws() {  # $1=工作区根目录 $2=ROS包名
    [ -f "$1/devel/setup.bash" ] || { echo "[ERROR] 缺少 $1/devel/setup.bash, 请先编译该工作区"; exit 1; }
}
check_ws "$WS_INIT" lidar_imu_init
check_ws "$WS_LIO"  sfast_lio

# ----------------------------- 进程管理 --------------------------------------
declare -a LAUNCH_PIDS=()
declare -a TAIL_PIDS=()

cleanup() {
    trap '' INT TERM   # 防止清理过程中被重复触发
    echo ""
    echo "[INFO] 正在停止所有 launch 进程..."
    # 注: 每个后台子 shell 用 exec 启动 roslaunch, 故 PID 即 roslaunch PID,
    # roslaunch 收到 SIGINT 会自行关闭它派生的所有 node
    for pid in "${LAUNCH_PIDS[@]}"; do
        kill -INT "$pid" 2>/dev/null || true
    done
    # 结束终端实时输出
    for pid in "${TAIL_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in "${LAUNCH_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    echo "[INFO] 已全部停止"
    exit 0
}
trap cleanup INT TERM

# ----------------------------- 启动 launch -----------------------------------
echo "[INFO] 日志目录: $LOG_DIR"
echo "[INFO] 启动 [1/2] D-LI-Init:  $LAUNCH_INIT $INIT_ARGS"
bash -c "source /opt/ros/noetic/setup.bash && source $WS_INIT/devel/setup.bash && exec roslaunch $LAUNCH_INIT $INIT_ARGS" \
    > "$LOG_DIR/lidar_imu_init.log" 2>&1 &
LAUNCH_PIDS+=($!)

echo "[INFO] 启动 [2/2] SFAST_LIO:  $LAUNCH_LIO $LIO_ARGS"
bash -c "source /opt/ros/noetic/setup.bash && source $WS_LIO/devel/setup.bash && exec roslaunch $LAUNCH_LIO $LIO_ARGS" \
    > "$LOG_DIR/sfast_lio.log" 2>&1 &
LAUNCH_PIDS+=($!)

echo "[INFO] 两个节点均已启动, 日志保存至:"
echo "       - D-LI-Init: $LOG_DIR/lidar_imu_init.log"
echo "       - SFAST_LIO: $LOG_DIR/sfast_lio.log"
echo "[INFO] 终端实时输出中: [INIT]=D-LI-Init (含动态初始化逐迭代求解结果), [LIO]=SFAST_LIO"
echo "[INFO] 按 Ctrl-C 停止全部"

# ----------------------------- 前台实时输出 -----------------------------------
# tail -F 跟踪两个日志文件, 带前缀显示到终端, 可看到动态初始化每次迭代求解输出
tail -n +1 -F "$LOG_DIR/lidar_imu_init.log" 2>/dev/null | sed -u 's/^/[INIT] /' &
TAIL_PIDS+=($!)
if [[ "$LIO_QUIET" -eq 0 ]]; then
    tail -n +1 -F "$LOG_DIR/sfast_lio.log" 2>/dev/null | sed -u 's/^/[LIO]  /' &
    TAIL_PIDS+=($!)
else
    echo "[INFO] --lio-quiet: SFAST_LIO 输出仅写入日志, 不在终端显示"
fi

# ----------------------------- 前台等待 ---------------------------------------
# roslaunch 任一退出则全部清理
wait -n "${LAUNCH_PIDS[@]}"
echo "[WARN] 检测到某个 launch 已退出, 清理另一个..."
cleanup