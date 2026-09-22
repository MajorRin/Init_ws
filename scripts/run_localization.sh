#!/usr/bin/env bash
# 回放参数请编辑同目录 run_localization.py 顶部的 CONFIG 配置区。
# 本入口只负责加载 ROS 环境；命令行参数原样传给 Python。
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
if [[ ! -f "$WORKSPACE/devel/setup.bash" ]]; then
    echo "缺少工作空间 devel/setup.bash，请先编译。" >&2
    exit 2
fi
# catkin setup interprets positional arguments; do not pass replay CLI flags to it.
load_workspace() { source "$WORKSPACE/devel/setup.bash"; }
load_workspace
exec python3 "$SCRIPT_DIR/run_localization.py" "$@"