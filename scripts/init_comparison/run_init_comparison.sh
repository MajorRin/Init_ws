#!/bin/bash
# =============================================================================
# run_init_comparison.sh — D-LI-Init(动态) 与 FAST-LIO2(静态) 初始化状态对比实验编排
#
# 同一 bag、同一起点(运动段偏移), 顺序回放两侧系统, 采集初始化观测日志,
# 离线提取 RTK 伪真值并运行分析器。
#
# 设计依据: .joycode/specs/dli-fastlio2-init-state-comparison/design.md 2.5 节
#
# 可修改接口(优先级: 命令行参数 > 环境变量 > 默认值):
#   --bag <path>        bag 路径      [环境变量 INIT_CMP_BAG]
#   --offset <sec>      运动段时间偏移 [环境变量 INIT_CMP_OFFSET, 默认 250]
#   --duration <sec>    回放时长(0=播到结尾) [环境变量 INIT_CMP_DURATION, 默认 20]
#   --skip-a / --skip-b 跳过 A/B 侧回放(调试分析器时使用)
#
# 用法:
#   ./run_init_comparison.sh
#   ./run_init_comparison.sh --offset 30
#   INIT_CMP_OFFSET=40 ./run_init_comparison.sh
# =============================================================================
# 注: 不使用 set -u — ROS 的 profile.d 脚本引用未定义变量(如 ROS_DISTRO)会直接报错退出

# 管道内 Python/rosbag 失败必须向上传递，不能被 tee 的退出码掩盖。
set -o pipefail

# ------------------------- 参数与默认值 -------------------------
BAG="${INIT_CMP_BAG:-/home/mao/rosbag/HZ_LiDAR_Dataset/AVIA/rosbag_all_19700101_080314.bag}"
OFFSET="${INIT_CMP_OFFSET:-20}"
DURATION="${INIT_CMP_DURATION:-20}"
SKIP_A=0
SKIP_B=0
MODE=init
SIDE=A
BASELINE_RUN=""
CANDIDATE_RUN=""
OUTPUT_DIR=""
RTK=""
WINDOW=10
TOL=0.05
PLOT_ARGS=()
usage() {
    echo "用法: bash $0 [--mode init|tuning] [选项]"
    echo "  init (默认): 顺序运行动态 A 和静态 B，与 RTK 比较"
    echo "  tuning: --baseline-run 历史目录 [--candidate-run 已有调参后目录]"
    echo "          未指定 candidate 时只回放 --side 指定算法，使用当前配置"
    echo "  --side A|B          调参算法：A 动态(默认)，B 静态"
    echo "  --bag 路径 --offset 秒 --duration 秒 (默认偏移250、时长20)"
    echo "  --rtk CSV路径       优先显式指定，否则查找 baseline 副本/历史报告来源"
    echo "  --output-dir 新目录 输出目录必须不存在；默认自动创建唯一目录"
    echo "  --window 秒 --tol 秒 (默认10、0.05) --no-plot"
    echo "  --skip-a --skip-b   仅用于 init 模式"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bag|--offset|--duration|--mode|--side|--baseline-run|--candidate-run|--output-dir|--rtk|--window|--tol)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "[run] 参数 $1 缺少值" >&2; exit 2
            fi ;;
    esac
    case "$1" in
        --bag) BAG="$2"; shift 2 ;;
        --offset) OFFSET="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --side) SIDE="$2"; shift 2 ;;
        --baseline-run) BASELINE_RUN="$2"; shift 2 ;;
        --candidate-run) CANDIDATE_RUN="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --rtk) RTK="$2"; shift 2 ;;
        --window) WINDOW="$2"; shift 2 ;;
        --tol) TOL="$2"; shift 2 ;;
        --no-plot) PLOT_ARGS=(--no-plot); shift ;;
        --skip-a) SKIP_A=1; shift ;;
        --skip-b) SKIP_B=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[run] 未知参数: $1" >&2; exit 2 ;;
    esac
done
[[ "$MODE" == init || "$MODE" == tuning ]] || { echo "mode 必须为 init 或 tuning" >&2; exit 2; }
[[ "$SIDE" == A || "$SIDE" == B ]] || { echo "side 必须为 A 或 B" >&2; exit 2; }
if [[ "$MODE" == tuning ]]; then
    [[ -n "$BASELINE_RUN" && $SKIP_A -eq 0 && $SKIP_B -eq 0 ]] || {
        echo "tuning 必须指定 --baseline-run，且不能使用 --skip-a/--skip-b" >&2; exit 2;
    }
    for input in "$BASELINE_RUN" "${CANDIDATE_RUN:-$BASELINE_RUN}"; do
        for suffix in events states; do
            [[ -s "$input/${SIDE}_${suffix}.log" ]] || {
                echo "缺少日志: $input/${SIDE}_${suffix}.log" >&2; exit 2;
            }
        done
    done
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    RTK="$(python3 - "$SCRIPT_DIR" "$BASELINE_RUN" "$RTK" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from analyze_tuning_comparison import resolve_rtk
try:
    print(resolve_rtk(sys.argv[2], sys.argv[3] or None))
except (OSError, ValueError) as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(2)
PY
    )" || exit 2
elif [[ -n "$BASELINE_RUN$CANDIDATE_RUN$RTK" || "$SIDE" != A ]]; then
    echo "baseline-run/candidate-run/rtk/side 仅用于 tuning 模式" >&2; exit 2
fi

# 在启动任何 ROS 进程前验证数值，避免非法参数触发无效实验。
python3 -c 'import math,sys; v=list(map(float,sys.argv[1:])); sys.exit(0 if all(math.isfinite(x) for x in v) and v[0]>=0 and v[1]>=0 and v[2]>0 and v[3]>0 else 2)' \
    "$OFFSET" "$DURATION" "$WINDOW" "$TOL" || { echo "偏移/时长须非负，窗口/容差须为有限正数" >&2; exit 2; }
A_WS=/home/mao/Init_ws
B_WS=/home/mao/FAST_LIO_ws
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S_%N)"
RUN_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/runs/$RUN_ID}"
# 禁止覆盖历史实验；所有输入路径在进入其他工作目录前规范化。
if [[ -e "$RUN_DIR" ]]; then echo "输出目录已存在: $RUN_DIR" >&2; exit 2; fi
mkdir -p "$(dirname "$RUN_DIR")" || exit 1
mkdir "$RUN_DIR" || exit 1
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
if [[ "$MODE" == tuning ]]; then
    BASELINE_RUN="$(cd "$BASELINE_RUN" && pwd)"
    [[ -z "$CANDIDATE_RUN" ]] || CANDIDATE_RUN="$(cd "$CANDIDATE_RUN" && pwd)"
    RTK="$(realpath "$RTK")"
fi
STATUS_LOG="$RUN_DIR/status.log"
FAIL=0

log() { echo "[run $RUN_ID] $*" | tee -a "$STATUS_LOG"; }
fail() { log "失败: $*"; FAIL=1; }

# ------------------------- 前置检查 -------------------------
if [[ -z "$CANDIDATE_RUN" && ! -f "$BAG" ]]; then
    log "错误: bag 不存在: $BAG (可用 --bag 或 INIT_CMP_BAG 修改)"
    exit 1
fi
if [[ -n "$CANDIDATE_RUN" ]]; then
    log "纯离线对比，不使用本次 bag/offset/duration 参数"
else
    BAG="$(realpath "$BAG")"
   log "bag       = $BAG"
    log "运动段偏移 = ${OFFSET}s (可通过 --offset / INIT_CMP_OFFSET 修改)"
    log "回放时长   = ${DURATION}s (0=播到结尾)"
fi
log "运行目录   = $RUN_DIR"

# ------------------------- 旧日志备份(tar) -------------------------
# 两侧旧日志均以 ios::out 截断方式打开, 运行前必须备份
backup_logs() {
    local ws="$1" tag="$2"
    local logdir="$ws/Log"
    if [[ -d "$logdir" ]]; then
        tar czf "$RUN_DIR/old_logs_${tag}.tar.gz" -C "$ws" Log 2>>"$STATUS_LOG" \
            && log "已备份 $tag 侧旧日志 -> old_logs_${tag}.tar.gz" \
            || fail "备份 $tag 侧旧日志失败"
    else
        log "$tag 侧无旧日志目录 $logdir, 跳过备份"
    fi
}
# 仅在确实回放时备份，离线模式不访问 ROS 工作区日志。

# ------------------------- 回放单侧 -------------------------
# 参数: 1=工作空间 2=launch 包名 3=launch 文件 4=roscore 端口
run_side() {
    local ws="$1" pkg="$2" launch_file="$3" tag="$4"
    log "---- 开始 $tag 侧回放 ($pkg $launch_file) ----"

    if ! (cd "$ws" && source /opt/ros/noetic/setup.bash && source devel/setup.bash \
          && roslaunch "$pkg" "$launch_file" rviz:=false > "$RUN_DIR/${tag}_ros.log" 2>&1 & \
          echo $! > "$RUN_DIR/${tag}_node.pid"); then
        fail "启动 $tag 侧节点失败, 见 $RUN_DIR/${tag}_ros.log"
        return 1
    fi
    local node_pid
    node_pid=$(cat "$RUN_DIR/${tag}_node.pid")
    log "$tag 侧节点已启动 (pid=$node_pid), 等待 8s 初始化话题..."
    sleep 8

    local play_args=(-s "$OFFSET" "$BAG")
    if [[ "$DURATION" != "0" ]]; then
        play_args=(-s "$OFFSET" -u "$DURATION" "$BAG")
    fi
    log "$tag 侧回放: rosbag play ${play_args[*]}"
    if (cd "$ws" && source /opt/ros/noetic/setup.bash \
          && rosbag play "${play_args[@]}" > "$RUN_DIR/${tag}_play.log" 2>&1); then
        log "$tag 侧回放命令正常结束"
    else
        fail "$tag 侧 rosbag play 异常退出, 见 $RUN_DIR/${tag}_play.log"
    fi
    # 等待节点消化完缓存帧
    sleep 5
    kill "$node_pid" 2>/dev/null
    pkill -f "roslaunch $pkg" 2>/dev/null
    sleep 2

    # 检查本侧日志是否生成
    local ok=1
    for f in "${tag}_events.log" "${tag}_states.log"; do
        if [[ ! -s "$RUN_DIR/$f" ]]; then
            fail "$tag 侧未生成观测日志 $RUN_DIR/$f"
            ok=0
        fi
    done
    if [[ $ok -eq 1 ]]; then
        log "$tag 侧观测日志已生成: $(wc -l < "$RUN_DIR/${tag}_states.log") 行状态 / $(wc -l < "$RUN_DIR/${tag}_events.log") 行事件"
    fi
    return 0
}

# ------------------------- RTK 离线提取 -------------------------
extract_rtk() {
    log "---- 离线提取 RTK 速度序列 ----"
    if (source /opt/ros/noetic/setup.bash && python3 "$SCRIPT_DIR/extract_rtk.py" \
        "$BAG" -o "$RUN_DIR/rtk_velocity.csv" 2>&1 | tee -a "$STATUS_LOG"); then
        if [[ ! -s "$RUN_DIR/rtk_velocity.csv" ]]; then
            fail "RTK 提取结果为空"
        else
            log "RTK 序列: $(($(wc -l < "$RUN_DIR/rtk_velocity.csv") - 1)) 条样本"
        fi
    else
        fail "RTK 提取失败"
    fi
}

# ------------------------- 分析 -------------------------
run_analysis() {
    log "---- 运行分析器 ----"
    if python3 "$SCRIPT_DIR/analyze_init_comparison.py" \
        --run-dir "$RUN_DIR" --rtk "$RUN_DIR/rtk_velocity.csv" \
        --window "$WINDOW" --tol "$TOL" "${PLOT_ARGS[@]}" 2>&1 | tee -a "$STATUS_LOG"; then
        log "分析完成, 结果见 $RUN_DIR/analysis 与 $RUN_DIR/comparison_report.md"
    else
        fail "分析器运行失败"
    fi
}

# ------------------------- 主流程 -------------------------
export INIT_CMP_DIR="$RUN_DIR"   # 两侧节点读取该环境变量写观测日志

if [[ "$MODE" == tuning ]]; then
    log "调参对比: 算法=$SIDE, 调参前=$BASELINE_RUN"
    log "RTK 来源 = $RTK"
    # 新实验独立保存 RTK，后续可直接作为下一轮 baseline。
    cp "$RTK" "$RUN_DIR/rtk_velocity.csv" || exit 1
    RTK="$RUN_DIR/rtk_velocity.csv"
    if [[ -z "$CANDIDATE_RUN" ]]; then
        # 保存本次回放配置供追溯，不把当前配置冒充为历史配置。
        if [[ "$SIDE" == A ]]; then
            backup_logs "$A_WS" A
            [[ $FAIL -eq 0 ]] || exit 1
            cp "$A_WS/src/D-LI-Init/config/avia.yaml" "$RUN_DIR/candidate_config.yaml" || exit 1
            cp "$A_WS/src/D-LI-Init/launch/livox_avia.launch" "$RUN_DIR/candidate.launch" || exit 1
            run_side "$A_WS" lidar_imu_init livox_avia.launch A
        else
            backup_logs "$B_WS" B
            [[ $FAIL -eq 0 ]] || exit 1
            cp "$B_WS/src/FAST_LIO/config/avia.yaml" "$RUN_DIR/candidate_config.yaml" || exit 1
            cp "$B_WS/src/FAST_LIO/launch/mapping_avia.launch" "$RUN_DIR/candidate.launch" || exit 1
            run_side "$B_WS" fast_lio mapping_avia.launch B
        fi
        [[ $FAIL -eq 0 ]] || exit 1
        CANDIDATE_RUN="$RUN_DIR"
    fi
    if ! python3 "$SCRIPT_DIR/analyze_tuning_comparison.py" \
        --baseline-run "$BASELINE_RUN" --candidate-run "$CANDIDATE_RUN" \
        --side "$SIDE" --rtk "$RTK" --output-dir "$RUN_DIR/analysis" \
        --window "$WINDOW" --tol "$TOL" "${PLOT_ARGS[@]}" 2>&1 | tee -a "$STATUS_LOG"; then
        fail "调参分析失败"
    fi
else
    if [[ $SKIP_A -eq 0 ]]; then
        backup_logs "$A_WS" A
        [[ $FAIL -eq 0 ]] || exit 1
        run_side "$A_WS" lidar_imu_init livox_avia.launch A
    fi
    if [[ $SKIP_B -eq 0 ]]; then
        backup_logs "$B_WS" B
        [[ $FAIL -eq 0 ]] || exit 1
        run_side "$B_WS" fast_lio mapping_avia.launch B
    fi
    extract_rtk
    run_analysis
fi

# ------------------------- 总结 -------------------------
log "================ 实验结束 ================"
if [[ $FAIL -eq 0 ]]; then
    log "全部步骤成功。结果目录: $RUN_DIR"
else
    log "存在失败步骤(详见 status.log), 如实报告, 未伪造数据。结果目录: $RUN_DIR"
fi
exit $FAIL