#!/usr/bin/env python3
"""D-LI-Init glog diagnostics: schema1 legacy plus schema2 unified channels.

Schema2 adds DLI_QUEUE heartbeats, DLI_BATCH batch solves, per-frame nested
state snapshots, RTK samples and clock samples. No ground truth or rotation."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np


def _symlog_axis(ax, linthresh=1e-10):
    """symlog y 轴并强制可见刻度: Matplotlib 默认的 LogFormatterSciNotation 在
    数据跨度不足一个数量级、或范围横跨正负/线性区时会返回空标签, 造成协方差
    面板 y 轴无刻度。这里显式用 decade 定位器 + %g 文本格式化, 子刻度密度按
    可见数量级跨度自适应; linthresh 按数据收紧(不深于最小量级一个数量级), 避免
    autoscale 的 5% margin 把下界推到 0 以下后, 零附近的对数阶梯在视区内展开出
    几十个刻度。跨度不足 0.6 个数量级的全正数据退化为线性轴: 视区放不下两个
    decade 刻度, 线性刻度更可读。"""
    from matplotlib.ticker import FuncFormatter, SymmetricalLogLocator
    values = np.concatenate([np.asarray(line.get_ydata(), dtype=float)
                             for line in ax.lines]) if ax.lines else np.array([])
    finite = values[np.isfinite(values)]
    if finite.size:
        lo = float(np.abs(finite).min())
        if lo > 0:
            linthresh = max(linthresh, lo / 10.0)
    if finite.size and float(finite.min()) > 0 and \
            float(np.log10(finite.max() / finite.min())) < 0.6:
        # 窄跨度全正数据: symlog 视区内放不下两个 decade 刻度。
        ax.set_yscale("linear")
        return
    ax.set_yscale("symlog", linthresh=linthresh)
    if finite.size and float(finite.min()) >= 0:
        # 全非负数据: autoscale 的 5% margin 会把下界推到 0 以下, 使 symlog
        # 负侧对数阶梯涌入视区。钳到数据最小值, 展示与纯 log 轴一致。
        ax.set_ylim(bottom=float(finite.min()))
    # 刻度密度随可见数量级跨度自适应: 跨度大时仍用 6 档子刻度会让
    # 标签在面板内(尤其原点附近的小值区)叠成一团。
    decades = 0.0
    if finite.size:
        top, bottom = float(finite.max()), float(finite.min())
        if bottom >= 0:
            decades = (np.log10(top / bottom) if bottom > 0
                       else np.log10(top / linthresh))
        else:
            decades = ((np.log10(top / linthresh) if top > linthresh else 0.0)
                       + (np.log10(-bottom / linthresh)
                          if bottom < -linthresh else 0.0))
    subs = ((1.0, 1.5, 2.0, 3.0, 5.0, 7.0) if decades <= 1.2 else
            (1.0, 2.0, 5.0) if decades <= 3.0 else (1.0,))
    ax.yaxis.set_major_locator(SymmetricalLogLocator(
        base=10, linthresh=linthresh, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _pos: "0" if v == 0 else f"{v:g}"))


CONFIG = {
    # 日志路径：要分析的 glog 目录（或具体 INFO 文件）。
    # 换一次运行就改这里；命令行 --log 优先于此处配置。
    "log": "drift_runs/20260922_142647_360333/glog",
    "output": "drift_analysis",
    "lio_window": 10.0,
    "show": True,
    "export_data": False,
    # 仪表盘全局字体缩放：1.0 为原始大小，调大如图例/刻度一起放大。
    "font_scale": 1.3,
}

# 全部界面字号统一由 font_scale 派生；改字体只改上面的 CONFIG。
FONT = {name: round(base * CONFIG["font_scale"], 1) for name, base in
        (("title", 10), ("label", 8), ("text", 8), ("note", 9),
         ("legend", 6), ("tick", 7), ("button", 10))}


VECTORS = {"position": 3, "velocity": 3, "gravity": 3, "bias_g": 3,
           "bias_a": 3, "quaternion_wxyz": 4, "covariance_diagonal": 24,
           "position_correction": 3, "pose_information_eigenvalues": 6,
           "pose_information_diagonal": 6}
SCALARS = ("time ros_time lag_ms residual condition cov_lidar frame_ms imu_ms "
           "down_ms icp_ms match_ms solve_ms map_ms diag_scope_ms "
           "rotation_correction_deg").split()
COUNTS = "frame points effective iterations searches".split()
FLAGS = "applied committed propagated".split()
TEXT = "stage status reason".split()
COV_GROUPS = ("R", "P", "RLI", "TLI", "V", "BG", "BA", "G")
MARKER = re.compile(r"\[(DLI_FRAME|DLI_EVENT|DLI_QUEUE|DLI_BATCH)\]\s*")

# ---- schema2 (unified protocol, see include/drift_diagnostics.h) ----
SCHEMA2_PREFIX = "schema run_id sequence steady_ms time data_time pass_id batch_id".split()
SCHEMA2_CLOCK = "ros_time lag_ms lag_s clock_mode lag_reason playback_rate playback_state".split()
SCHEMA2_CLOCK_MODES = ("sim", "system")
SCHEMA2_LAG_REASONS = ("non_sim_time_base_unverified", "clock_not_ready", "clock_reversed",
                       "invalid_data_time", "negative_lag", "valid")
SCHEMA2_SNAPSHOT = {"position": 3, "velocity": 3, "gravity": 3, "bias_g": 3, "bias_a": 3,
                    "quaternion_wxyz": 4, "covariance_diagonal": 24}
SCHEMA2_SNAPSHOT_KEYS = {*SCHEMA2_SNAPSHOT, "covariance_finite", "covariance_nonnegative"}
SCHEMA2_STAGES = ("direct_lio", "post_dynamic", "cache_reprocess", "initial_lgo")
SCHEMA2_STATUSES = ("post", "skipped", "partial_applied", "predict")
SCHEMA2_FRAME = frozenset(
    SCHEMA2_PREFIX + SCHEMA2_CLOCK + [
        "frame", "cache_index", "stage", "status", "reason", "applied", "committed", "propagated",
        "points", "effective", "iterations", "searches",
        "relative_time", "effective_ratio", "raw_points", "undistorted_points", "map_points",
        "publish_ms", "build_ms", "before_propagation", "after_propagation", "after_update",
        "covariance_source", "state_frame", "residual_source", "information_source",
        "residual", "condition", "cov_lidar", "frame_ms", "imu_ms", "down_ms", "icp_ms",
        "match_ms", "solve_ms", "map_ms", "diagnostic_timing_scope", "rotation_correction_deg",
        "position_correction", "position", "velocity", "gravity", "bias_g", "bias_a",
        "quaternion_wxyz", "covariance_diagonal", "pose_information_eigenvalues",
        "pose_information_diagonal", "speed", "gravity_norm", "vertical_velocity",
        "rtk_time", "rtk_time_difference", "rtk_found", "rtk_valid", "rtk_reason", "rtk_frame",
        "rtk_velocity", "rtk_speed", "rtk_axis_error", "rtk_alignment_reason", "diag_scope_ms"])
SCHEMA2_FRAME_NULLABLE = SCHEMA2_CLOCK[:3] + [
    "points", "effective", "iterations", "searches", "relative_time", "effective_ratio",
    "raw_points", "undistorted_points", "map_points", "publish_ms", "build_ms",
    "residual", "condition", "cov_lidar", "frame_ms", "imu_ms", "down_ms", "icp_ms",
    "match_ms", "solve_ms", "map_ms", "rotation_correction_deg", "speed", "gravity_norm",
    "vertical_velocity", "rtk_time", "rtk_time_difference", "rtk_speed", "rtk_axis_error",
    "diag_scope_ms", "steady_ms", "time", "data_time"]
SCHEMA2_EVENT = frozenset(SCHEMA2_PREFIX + ["event", "reason"])
SCHEMA2_QUEUE = frozenset(SCHEMA2_PREFIX + SCHEMA2_CLOCK + [
    "initialized", "lidar_queue", "imu_queue", "latest_lidar", "processed", "latest_imu"])
SCHEMA2_BATCH_WINDOW = ["window_start", "window_end", "window_size", "cache_size",
                        "configured_length", "estimate_acc_bias"]
SCHEMA2_BATCH_SOLVED = frozenset(
    SCHEMA2_PREFIX + ["stage", "refinement"] + SCHEMA2_BATCH_WINDOW + [
        "decomposition_status", "solve_status", "solution_finite", "system_finite",
        "normal_equation_relative_residual", "residual_reason", "solution", "gravity",
        "gravity_reason"])
SCHEMA2_BATCH_SOLVED_NON_GYRO = SCHEMA2_BATCH_SOLVED | {
    "velocity_frame", "velocity_transform_reason", "velocities", "gravity_tangent_increment"}
SCHEMA2_BATCH_END = frozenset(
    SCHEMA2_PREFIX + ["stage", "refinement"] + SCHEMA2_BATCH_WINDOW + [
        "initial_velocity", "initial_velocity_time", "initial_velocity_frame", "initial_speed",
        "gravity", "gravity_norm", "gravity_frame", "bias_g", "bias_a", "bias_a_source",
        "delta_v", "delta_v_reason", "velocity_threshold", "finite", "observations",
        "quality_reason", "velocity_frame", "velocity_transform_reason", "velocities",
        "batch_wall_ms", "batch_algorithm_ms", "observer_ms", "pass_wall_ms", "timing_scope",
        "serialization_ms_excludes_log_write"])
NOTES = [
    "No ground truth: motion, residuals and covariance are proxies, not absolute drift/error.",
    "World/IMU poses; quaternion w,x,y,z; no installation rotation or pose alignment.",
    "Comparison uses each run's init_done-relative time, not coordinate-overlap accuracy.",
    "Batch gravity is normalized to |G|=9.8 m/s^2 every pass; only components carry information.",
    "Covariance order: R,P,RLI,TLI,V,BG,BA,G (24 dimensions); no mixed-unit trace.",
    "match_ms includes search+matching and is inside icp_ms; never add these together.",
    "frame_ms excludes subsequent spectrum/logging; diag_scope_ms is not pure logging cost.",
    "Path length only sums adjacent valid posteriors; invalid/missing frames break lines.",
]


@dataclass
class Run:
    label: str
    files: list
    frames: list
    events: list
    counts: Counter
    messages: list
    init_time: object = None
    origin: object = None
    origin_kind: str = "unavailable"
    lio_window: float = 10.0
    schema: int = 1
    run_id: object = None
    queues: list = None
    batches: list = None

    def __post_init__(self):
        if self.queues is None:
            self.queues = []
        if self.batches is None:
            self.batches = []

    def warn(self, message):
        self.messages.append(message)
        warnings.warn(f"{self.label}: {message}", RuntimeWarning, stacklevel=2)


def collect_logs(source):
    """Use INFO real files once, including rotated logs (no symlink traversal)."""
    path = Path(source).expanduser()
    if not path.exists():
        raise ValueError(f"日志路径不存在: {path}")
    candidates = [path] if path.is_file() else [
        p for p in path.rglob("*")
        if p.is_file() and re.search(r"(^|[._-])INFO([._-]|$)", p.name)]
    unique = {}
    for p in sorted(candidates):
        real = p.resolve()
        stat = real.stat()
        unique.setdefault((stat.st_dev, stat.st_ino), real)

    def rotation_key(p):
        # glog timestamp names sort chronologically; .2 precedes .1 precedes base.
        glog = re.search(r"\.INFO\.(\d{8}-\d{6})", p.name)
        if glog:
            return glog[1], p.name, 0
        match = re.match(r"^(.*INFO(?:\..*?)?)\.(\d+)$", p.name)
        if match and int(match[2]) < 10000:
            return match[1], "", -int(match[2])
        return p.name, "", 0

    files = sorted(unique.values(), key=rotation_key)
    if not files:
        raise ValueError(f"未找到 INFO 日志真实文件: {path}")
    return files


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def positive_time(value):
    return numeric(value) and value > 0


def finite_positive_window(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("lio_window 必须为有限正值") from None
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError("lio_window 必须为有限正值")
    return number


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"重复 JSON 键: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"非标准 JSON 常数: {value}")


def validate(data, kind):
    if not isinstance(data, dict) or type(data.get("schema")) is not int or data["schema"] != 1:
        raise ValueError("仅支持 schema=1 对象")
    if kind not in ("DLI_FRAME", "DLI_EVENT"):
        raise ValueError("schema1 不包含该通道")
    if kind == "DLI_EVENT":
        if set(data) != {"schema", "event", "time", "reason"}:
            raise ValueError("事件字段不符合 schema1")
        if data["event"] not in ("startup", "init_done", "shutdown") or not isinstance(data["reason"], str):
            raise ValueError("事件名称或 reason 无效")
        if not numeric(data["time"]) or data["time"] < 0:
            raise ValueError("事件时间无效")
        if data["event"] == "init_done" and not positive_time(data["time"]):
            raise ValueError("init_done 时间必须有效且大于零")
        return
    expected = {"schema", *SCALARS, *COUNTS, *FLAGS, *TEXT, *VECTORS}
    if set(data) != expected:
        raise ValueError(f"帧字段不符合 schema1: {sorted(set(data) ^ expected)}")
    for name in SCALARS:
        if data[name] is not None and not numeric(data[name]):
            raise ValueError(f"{name} 应为有限数值或 null")
    for name in COUNTS:
        if type(data[name]) is not int or data[name] < 0:
            raise ValueError(f"{name} 应为非负整数")
    for name in FLAGS:
        if type(data[name]) not in (bool, int) or data[name] not in (0, 1):
            raise ValueError(f"{name} 应为 0/1")
    if not all(isinstance(data[name], str) for name in TEXT):
        raise ValueError("stage/status/reason 应为字符串")
    if data["status"] not in ("post", "skipped", "partial_applied", "predict"):
        raise ValueError("未知 status")
    for name, size in VECTORS.items():
        value = data[name]
        if not isinstance(value, list) or len(value) != size:
            raise ValueError(f"{name} 长度应为 {size}")
        if any(v is not None and not numeric(v) for v in value):
            raise ValueError(f"{name} 应含有限数值或 null")


def validate2_vector(value, name, size):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} 长度应为 {size}")
    if any(v is not None and not numeric(v) for v in value):
        raise ValueError(f"{name} 应含有限数值或 null")


def validate2_snapshot(value, name):
    if not isinstance(value, dict) or set(value) != SCHEMA2_SNAPSHOT_KEYS:
        raise ValueError(f"{name} 快照字段不符合 schema2")
    for key, size in SCHEMA2_SNAPSHOT.items():
        validate2_vector(value[key], f"{name}.{key}", size)
    for key in ("covariance_finite", "covariance_nonnegative"):
        if type(value[key]) not in (bool, int) or value[key] not in (0, 1):
            raise ValueError(f"{name}.{key} 应为 0/1")


def validate2_velocities(value):
    if not isinstance(value, list):
        raise ValueError("velocities 应为数组")
    for item in value:
        if not isinstance(item, dict) or set(item) != {"cache_index", "time", "velocity",
                                                       "R_local_to_batch_column_major"}:
            raise ValueError("velocities 元素字段不符合 schema2")
        if type(item["cache_index"]) is not int or item["cache_index"] < 0:
            raise ValueError("velocities.cache_index 应为非负整数")
        if item["time"] is not None and not numeric(item["time"]):
            raise ValueError("velocities.time 应为有限数值或 null")
        if item["velocity"] is not None:
            validate2_vector(item["velocity"], "velocities.velocity", 3)
        validate2_vector(item["R_local_to_batch_column_major"],
                         "velocities.R_local_to_batch_column_major", 9)


def validate2(data, kind):
    """Strict schema2 field-set validation; missing values stay JSON null (never zero)."""
    if not isinstance(data, dict) or data.get("schema") != 2:
        raise ValueError("仅支持 schema=2 对象")
    if not isinstance(data["run_id"], str) or not data["run_id"]:
        raise ValueError("run_id 应为非空字符串")
    if type(data["sequence"]) is not int or data["sequence"] < 1:
        raise ValueError("sequence 应为正整数")
    if not numeric(data["steady_ms"]) or data["steady_ms"] < 0:
        raise ValueError("steady_ms 应为非负有限数值")
    for key in ("time", "data_time"):
        if data[key] is not None and not numeric(data[key]):
            raise ValueError(f"{key} 应为有限数值或 null")
    for key in ("pass_id", "batch_id"):
        if type(data[key]) is not int or data[key] < 0:
            raise ValueError(f"{key} 应为非负整数")
    if kind == "DLI_EVENT":
        if set(data) != SCHEMA2_EVENT:
            raise ValueError(f"事件字段不符合 schema2: {sorted(set(data) ^ SCHEMA2_EVENT)}")
        if data["event"] not in ("startup", "init_done", "shutdown") or not isinstance(data["reason"], str):
            raise ValueError("事件名称或 reason 无效")
        if data["time"] is None or data["time"] < 0:
            raise ValueError("schema2 事件时间必须有效且不小于零")
        if data["event"] == "init_done" and not positive_time(data["data_time"]):
            raise ValueError("init_done data_time 必须有效且大于零")
        return
    if kind == "DLI_QUEUE":
        if set(data) != SCHEMA2_QUEUE:
            raise ValueError(f"队列字段不符合 schema2: {sorted(set(data) ^ SCHEMA2_QUEUE)}")
        if type(data["initialized"]) not in (bool, int) or data["initialized"] not in (0, 1):
            raise ValueError("initialized 应为 0/1")
        for key in ("lidar_queue", "imu_queue"):
            if type(data[key]) is not int or data[key] < 0:
                raise ValueError(f"{key} 应为非负整数")
        for key in ("latest_lidar", "processed", "latest_imu"):
            if data[key] is not None and not numeric(data[key]):
                raise ValueError(f"{key} 应为有限数值或 null")
    elif kind == "DLI_BATCH":
        if not isinstance(data["stage"], str):
            raise ValueError("stage 应为字符串")
        if data["stage"] == "batch_end":
            expected = SCHEMA2_BATCH_END
        elif data["stage"] == "gyro_bias":
            expected = SCHEMA2_BATCH_SOLVED
        else:
            expected = SCHEMA2_BATCH_SOLVED_NON_GYRO
        if set(data) != expected:
            raise ValueError(f"批解字段不符合 schema2: {sorted(set(data) ^ expected)}")
        for key in ("window_start", "window_end"):
            if data[key] is not None and not numeric(data[key]):
                raise ValueError(f"{key} 应为有限数值或 null")
        for key in ("window_size", "cache_size", "configured_length"):
            if type(data[key]) is not int or data[key] < 0:
                raise ValueError(f"{key} 应为非负整数")
        if type(data["estimate_acc_bias"]) not in (bool, int) or data["estimate_acc_bias"] not in (0, 1):
            raise ValueError("estimate_acc_bias 应为 0/1")
        if data["stage"] == "batch_end":
            if data["refinement"] is not None:
                raise ValueError("batch_end 的 refinement 应为 null")
            for key in ("initial_velocity", "gravity", "bias_g", "bias_a"):
                validate2_vector(data[key], key, 3)
            for key in ("initial_velocity_time", "initial_speed", "gravity_norm", "delta_v",
                        "velocity_threshold", "batch_wall_ms", "batch_algorithm_ms",
                        "observer_ms", "pass_wall_ms", "serialization_ms_excludes_log_write"):
                if data[key] is not None and not numeric(data[key]):
                    raise ValueError(f"{key} 应为有限数值或 null")
            if type(data["finite"]) not in (bool, int) or data["finite"] not in (0, 1):
                raise ValueError("finite 应为 0/1")
            if type(data["observations"]) is not int or data["observations"] < 0:
                raise ValueError("observations 应为非负整数")
            for key in ("initial_velocity_frame", "gravity_frame", "bias_a_source",
                        "delta_v_reason", "quality_reason", "velocity_frame",
                        "velocity_transform_reason", "timing_scope"):
                if not isinstance(data[key], str) or not data[key]:
                    raise ValueError(f"{key} 应为非空字符串")
        else:
            if type(data["refinement"]) is not int or data["refinement"] < -1:
                raise ValueError("refinement 应为不小于 -1 的整数")
            for key in ("decomposition_status", "solve_status"):
                if type(data[key]) is not int or data[key] < 0:
                    raise ValueError(f"{key} 应为非负整数")
            for key in ("solution_finite", "system_finite"):
                if type(data[key]) not in (bool, int) or data[key] not in (0, 1):
                    raise ValueError(f"{key} 应为 0/1")
            if not isinstance(data["normal_equation_relative_residual"], (int, float, type(None))):
                raise ValueError("normal_equation_relative_residual 应为数值或 null")
            if not isinstance(data["solution"], list) or any(
                    v is not None and not numeric(v) for v in data["solution"]):
                raise ValueError("solution 应为有限数值或 null 数组")
            if data["gravity"] is not None:
                validate2_vector(data["gravity"], "gravity", 3)
            for key in ("residual_reason", "gravity_reason"):
                if not isinstance(data[key], str) or not data[key]:
                    raise ValueError(f"{key} 应为非空字符串")
            if data["stage"] != "gyro_bias":
                for key in ("velocity_frame", "velocity_transform_reason"):
                    if not isinstance(data[key], str) or not data[key]:
                        raise ValueError(f"{key} 应为非空字符串")
                if data["gravity_tangent_increment"] is not None:
                    validate2_vector(data["gravity_tangent_increment"],
                                     "gravity_tangent_increment", 2)
        if "velocities" in data:
            validate2_velocities(data["velocities"])
        return
    elif kind == "DLI_FRAME":
        if set(data) != SCHEMA2_FRAME:
            raise ValueError(f"帧字段不符合 schema2: {sorted(set(data) ^ SCHEMA2_FRAME)}")
        if data["clock_mode"] not in SCHEMA2_CLOCK_MODES:
            raise ValueError("clock_mode 无效")
        if data["lag_reason"] not in SCHEMA2_LAG_REASONS:
            raise ValueError("lag_reason 无效")
        if data["playback_rate"] is not None and not numeric(data["playback_rate"]):
            raise ValueError("playback_rate 应为有限数值或 null")
        if not isinstance(data["playback_state"], str):
            raise ValueError("playback_state 应为字符串")
        if type(data["frame"]) is not int or data["frame"] < 0:
            raise ValueError("frame 应为非负整数")
        if type(data["cache_index"]) is not int or data["cache_index"] < -1:
            raise ValueError("cache_index 应为不小于 -1 的整数")
        if data["stage"] not in SCHEMA2_STAGES:
            raise ValueError(f"未知 stage: {data['stage']}")
        if data["status"] not in SCHEMA2_STATUSES:
            raise ValueError("未知 status")
        if not isinstance(data["reason"], str):
            raise ValueError("reason 应为字符串")
        for key in ("applied", "committed", "propagated", "rtk_found", "rtk_valid"):
            if type(data[key]) not in (bool, int) or data[key] not in (0, 1):
                raise ValueError(f"{key} 应为 0/1")
        for key in SCHEMA2_FRAME_NULLABLE:
            if data[key] is not None and not numeric(data[key]):
                raise ValueError(f"{key} 应为有限数值或 null")
        for name, size in VECTORS.items():
            validate2_vector(data[name], name, size)
        validate2_snapshot(data["before_propagation"], "before_propagation")
        if data["after_propagation"] is not None:
            validate2_snapshot(data["after_propagation"], "after_propagation")
        validate2_snapshot(data["after_update"], "after_update")
        for key in ("covariance_source", "state_frame", "residual_source", "information_source",
                    "diagnostic_timing_scope", "rtk_reason", "rtk_frame", "rtk_alignment_reason"):
            if not isinstance(data[key], str) or not data[key]:
                raise ValueError(f"{key} 应为非空字符串")
        if data["rtk_velocity"] is not None:
            validate2_vector(data["rtk_velocity"], "rtk_velocity", 3)
        return
    else:
        raise ValueError("未知 schema2 通道")
    if data["clock_mode"] not in SCHEMA2_CLOCK_MODES:
        raise ValueError("clock_mode 无效")
    if data["lag_reason"] not in SCHEMA2_LAG_REASONS:
        raise ValueError("lag_reason 无效")
    if data["playback_rate"] is not None and not numeric(data["playback_rate"]):
        raise ValueError("playback_rate 应为有限数值或 null")
    if not isinstance(data["playback_state"], str):
        raise ValueError("playback_state 应为字符串")


def load_run(source, label="log", include_direct=False, lio_window=10.0):
    lio_window = finite_positive_window(lio_window)
    files = collect_logs(source)
    run = Run(label, [str(p) for p in files], [], [], Counter(), [], lio_window=lio_window)
    seen = {}
    active_init = None
    gap = False
    schema = None
    run_id = None
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for lineno, line in enumerate(stream, 1):
                marker = MARKER.search(line)
                if not marker:
                    continue
                kind = marker[1]
                run.counts["tagged_lines"] += 1
                run.counts[kind] += 1
                try:
                    data = json.loads(line[marker.end():], object_pairs_hook=unique_object,
                                      parse_constant=reject_constant)
                    data_schema = data.get("schema") if isinstance(data, dict) else None
                    if data_schema == 1:
                        validate(data, kind)
                    elif data_schema == 2:
                        validate2(data, kind)
                    else:
                        raise ValueError("仅支持 schema=1 或 schema=2 对象")
                except (ValueError, TypeError, OverflowError, KeyError) as exc:
                    run.counts["bad_lines"] += 1
                    if len(run.messages) < 10:
                        run.warn(f"坏行 {path.name}:{lineno}: {exc}")
                    gap = True
                    continue
                if schema is None:
                    schema = data["schema"]
                elif schema != data["schema"]:
                    raise ValueError(f"{label}: 混入不同 schema 对象（{schema} 与 {data['schema']}），拒绝解析多版本日志")
                run.schema = schema
                if schema == 2:
                    if run_id is None:
                        run_id = data["run_id"]
                        run.run_id = run_id
                    elif run_id != data["run_id"]:
                        raise ValueError(f"{label}: 检测到多个 run_id，拒绝混入多运行；请分别指定日志")
                if kind == "DLI_EVENT":
                    run.events.append(dict(data, source=str(path), line=lineno))
                    run.counts[data["event"]] += 1
                    if data["event"] == "startup" and run.counts["startup"] > 1:
                        raise ValueError(f"{label}: 检测到多次 startup，拒绝混入多运行；请分别指定日志")
                    if data["event"] == "startup" and (run.frames or active_init is not None):
                        raise ValueError(f"{label}: startup 出现在帧或初始化事件之后，日志顺序或运行边界不可靠")
                    if data["event"] == "shutdown":
                        gap = True
                        active_init = None
                    if data["event"] == "init_done":
                        if run.counts["init_done"] > 1:
                            raise ValueError(f"{label}: 多次 init_done，无法可靠确定单次运行")
                        active_init = data["time"]
                        run.init_time = active_init
                    continue
                if kind in ("DLI_QUEUE", "DLI_BATCH"):
                    # Schema2-only channels; stored verbatim in input order.
                    target = run.queues if kind == "DLI_QUEUE" else run.batches
                    target.append(dict(data, source=str(path), line=lineno))
                    continue
                key = data["frame"]
                if key in seen:
                    old, row = seen[key]
                    if old == data:
                        run.counts["duplicates"] += 1
                    else:
                        run.counts["conflicts"] += 1
                        row["conflict"] = True
                        gap = True
                    continue
                row = dict(data, source=str(path), line=lineno, init_seen=active_init,
                           gap_before=gap, conflict=False)
                if schema == 2:
                    # Preserve the emitted value; prepare_run overwrites relative_time.
                    row["emitted_relative_time"] = data["relative_time"]
                seen[key] = (data, row)
                run.frames.append(row)
                gap = False
    if not run.frames:
        raise ValueError(f"{label}: 没有可解析的 schema{schema if schema else '?'} 帧")
    for key in ("bad_lines", "duplicates", "conflicts"):
        if run.counts[key]:
            run.warn(f"{key}={run.counts[key]}；重复帧保留首次，冲突帧不用于运动量")
    if not run.counts["startup"]:
        run.warn("缺少 startup，无法完全确认运行边界")
    if run.init_time is None and any(r["stage"] == "post_dynamic" for r in run.frames):
        run.warn("缺少有效 init_done，动态帧仅保留诊断，不计为有效运动")
    prepare_run(run, include_direct)
    return run


def vector(row, name):
    return np.array(row[name], dtype=float)


def rpy_degrees(q):
    if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
        return np.full(3, np.nan)
    w, x, y, z = q / np.linalg.norm(q)
    return np.degrees([math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                       math.asin(np.clip(2*(w*y-z*x), -1, 1)),
                       math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))])


# ---- legacy IMU excitation (legacy_logs/imu.txt beside the glog run directory) ----
# IMU_Processing.hpp 写出: [t, 陀螺 xyz (rad/s), 比力 xyz] 共 7 列, 原始值(未去偏置)。
# Livox 驱动比力以 g 为单位, 故 avia.yaml 的 mean_acc_norm=1.0; 单位仍自动检测以兼容 m/s^2 日志。
G_M_S2 = 9.805
IMU_GRAVITY_WINDOW_S = 2.0   # 开头静止窗口: 估计重力方向 + 检测比力单位
IMU_ROLLING_WINDOW_S = 1.0   # 滚动均值窗口: 指示"持续同向加速度"(可辨识性风险)
IMU_MIN_SAMPLES = 50


def find_imu_log(files):
    """Locate legacy_logs/imu.txt beside the glog run directory of any log file."""
    for name in files:
        for parent in list(Path(name).resolve().parents)[:5]:
            candidate = parent / "legacy_logs" / "imu.txt"
            if candidate.is_file():
                return candidate
    return None


def load_imu_excitation(run):
    """Parse legacy imu.txt and derive IMU excitation proxies for initialization.

    Returns a dict cached on the run (imu_excite). Gravity direction is the mean
    specific force over the initial window; de-gravity acceleration is the proxy
    for linear excitation, gyroscope magnitude for rotational excitation. The
    rolling mean keeps only sustained (near-constant direction) acceleration:
    magnitude alone does not make v0/gravity identifiable.
    """
    if getattr(run, "imu_excite", None) is not None:
        return run.imu_excite
    result = {"error": None, "path": None, "unit": None, "stationary": None}
    run.imu_excite = result
    path = find_imu_log(run.files)
    result["path"] = str(path) if path else None
    if path is None:
        result["error"] = "missing legacy_logs/imu.txt beside the glog run directory"
        return result
    times, gyro, acc = [], [], []
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                parts = line.split()
                if len(parts) != 7:
                    continue
                try:
                    row = [float(p) for p in parts]
                except ValueError:
                    continue
                if not all(math.isfinite(v) for v in row):
                    continue
                times.append(row[0])
                gyro.append(row[1:4])
                acc.append(row[4:7])
    except OSError as exc:
        result["error"] = f"cannot read {path}: {exc}"
        return result
    if len(times) < IMU_MIN_SAMPLES:
        result["error"] = f"only {len(times)} usable samples in {path}"
        return result
    t = np.asarray(times)
    gyro = np.asarray(gyro)
    acc = np.asarray(acc)
    # imu.txt 写入方式: 动态初始化的 22-pass 循环会把求解窗口(约 1.2s)逐遍重新写日志,
    # 形成 N 份完全相同的重放段, 之后才是 LIO 阶段连续段, 时间轴因此非单调
    # (同一条曲线在同一 x 上出现多个 y 的根源; searchsorted 滚动均值也随之失真)。
    # 按时间戳去重(保留首次出现), 必要时稳定排序, 恢复单调时间轴。
    replay_jumps = int((np.diff(t) < 0).sum())
    _, first_idx = np.unique(t, return_index=True)
    keep = np.sort(first_idx)
    if len(keep) != len(t):
        t, gyro, acc = t[keep], gyro[keep], acc[keep]
    if (np.diff(t) < 0).any():
        order = np.argsort(t, kind="stable")
        t, gyro, acc = t[order], gyro[order], acc[order]
    window = t < t[0] + IMU_GRAVITY_WINDOW_S
    if not window.any():
        window = np.ones_like(t, dtype=bool)
    norms = np.linalg.norm(acc[window], axis=1)
    norm0 = float(norms.mean())
    # 单位自检: Avia legacy 日志为 g; 兼容 m/s^2; 其他视为未知(按 m/s^2 原样并标记)。
    if 0.3 <= norm0 <= 3.0:
        unit, scale = "g", G_M_S2
    elif 7.0 <= norm0 <= 12.0:
        unit, scale = "m/s^2", 1.0
    else:
        unit, scale = "unknown", 1.0
    gravity = acc[window].mean(axis=0)          # 比力静止均值(原始单位)
    dyn = (acc - gravity) * scale               # 去重力线加速度 (m/s^2)
    stationary = bool(np.median(np.linalg.norm(gyro[window], axis=1)) < 0.05
                      and float(norms.std()) < 0.05)
    origin = run.origin if run.origin is not None else t[0]
    dt_roll = max(IMU_ROLLING_WINDOW_S, 1e-9)
    left = np.searchsorted(t, t - dt_roll, side="left")
    counts = (np.arange(len(t)) - left + 1).astype(float)

    def _rolling(values):
        sums = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
        sums = sums[np.arange(len(t)) + 1] - sums[left]
        return sums / counts[:, None]

    roll_xyz = _rolling(dyn)
    dyn_mag = np.linalg.norm(dyn, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    force_norm_g = np.linalg.norm(acc, axis=1) * scale / G_M_S2
    # 求解器实际使用的 IMU 窗口(DLI_BATCH 的 window_start/end, 绝对 data_time):
    # 换算到与仪表盘一致的相对时间并统计窗口内激励——判读"初始化激励是否足够"
    # 应以该窗口内数据为准, 而非 imu.txt 全跨度(其大部分常为 LIO 阶段)。
    window = None
    for batch in getattr(run, "batches", None) or []:
        if (batch.get("stage") not in (None, "batch_end")
                and batch.get("window_start") is not None
                and batch.get("window_end") is not None):
            window = (float(batch["window_start"]), float(batch["window_end"]))
            break
    result.update(error=None, unit=unit, stationary=stationary, count=int(len(t)),
                  span_s=float(t[-1] - t[0]), t=t - origin, origin_used=float(origin),
                  replayed_logs=replay_jumps,
                  gyro_mag=gyro_mag, dyn_mag=dyn_mag, dyn_xyz=dyn,
                  roll_mag=np.linalg.norm(roll_xyz, axis=1), roll_xyz=roll_xyz,
                  force_norm_g=force_norm_g, gravity_g=gravity.tolist())
    if window is not None:
        mask = (t >= window[0]) & (t <= window[1])
        result.update(win_t0=window[0] - origin, win_t1=window[1] - origin,
                      win_count=int(mask.sum()))
        if int(mask.sum()) >= 2:
            result.update(win_acc_mean=float(dyn_mag[mask].mean()),
                          win_acc_p95=float(np.percentile(dyn_mag[mask], 95)),
                          win_acc_max=float(dyn_mag[mask].max()),
                          win_gyro_mean=float(gyro_mag[mask].mean()),
                          win_gyro_max=float(gyro_mag[mask].max()),
                          win_force_std_g=float(force_norm_g[mask].std()))
    return result



# ---- 逐环节初始化剖析 (ported from scripts/batch_stage_analysis.py) ----
# 环节1: linear_initial 逐帧 R_local_to_batch (ICP 里程计旋转) vs legacy imu.txt
#        陀螺积分 (bg=0) 的逐帧相对旋转误差;
# 环节2: gyro_bias 阶段实际 delta_bg 解 vs 误差向量折算的应有修正量 (欠拟合判据);
# 环节3: 线性解 |v0| / |g| 与法方程相对残差 (病态解判据)。
STAGE_IMU_STEP_S = 0.002


def _rotvec(R):
    """SO(3) 旋转矩阵 -> 旋转向量 (rad)。"""
    angle = math.acos(min(1.0, max(-1.0, (float(np.trace(R)) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(angle))
    return axis * angle


def _stage_imu_log(run):
    """读 legacy imu.txt 陀螺 (rad/s), 返回 (t, gyro, error); 复用 pass 重放去重逻辑。"""
    cached = getattr(run, "stage_imu", None)
    if cached is not None:
        return cached
    result = (None, None, "missing legacy_logs/imu.txt beside the glog run directory")
    path = find_imu_log(run.files)
    if path is not None:
        times, gyro = [], []
        try:
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    parts = line.split()
                    if len(parts) != 7:
                        continue
                    try:
                        row = [float(p) for p in parts]
                    except ValueError:
                        continue
                    if not all(math.isfinite(v) for v in row):
                        continue
                    times.append(row[0])
                    gyro.append(row[1:4])   # imu.txt 列序: t, 陀螺 xyz, 比力 xyz
        except OSError as exc:
            result = (None, None, f"cannot read {path}: {exc}")
        else:
            if len(times) < IMU_MIN_SAMPLES:
                result = (None, None, f"only {len(times)} usable samples in {path}")
            else:
                t, w = np.asarray(times), np.asarray(gyro)
                _, first_idx = np.unique(t, return_index=True)   # pass 重放去重
                keep = np.sort(first_idx)
                if len(keep) != len(t):
                    t, w = t[keep], w[keep]
                order = np.argsort(t, kind="stable")
                result = (t[order], w[order], None)
    run.stage_imu = result
    return result
def _imu_integrate_rot(t, gyro, t0, t1):
    """角速度数值积分 (bg=0) 得相对旋转, Rodrigues 递推, 步长 STAGE_IMU_STEP_S。"""
    Rg = np.eye(3)
    for tt in np.arange(t0, t1, STAGE_IMU_STEP_S):
        w = np.array([np.interp(tt, t, gyro[:, k]) for k in range(3)])
        theta = float(np.linalg.norm(w)) * STAGE_IMU_STEP_S
        if theta > 1e-12:
            k_ = w / np.linalg.norm(w)
            K = np.array([[0, -k_[2], k_[1]], [k_[2], 0, -k_[0]], [-k_[1], k_[0], 0]])
            Rg = Rg @ (np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K)
    return Rg


def load_stage_analysis(run):
    """逐环节剖析, 缓存在 run.stage_diag; 单位: 角度 deg, 角速度 rad/s。"""
    if getattr(run, "stage_diag", None) is not None:
        return run.stage_diag
    result = {"error": None, "passes": []}
    run.stage_diag = result
    if run.schema != 2 or not run.batches:
        result["error"] = "schema1 log; stage details require schema2 DLI_BATCH"
        return result
    groups = {}
    for b in run.batches:
        groups.setdefault((b["pass_id"], b["batch_id"]), {})[b["stage"]] = b
    if not groups:
        result["error"] = "no DLI_BATCH records in this log"
        return result
    t_imu, gyro, err = _stage_imu_log(run)
    if err is not None:
        result["error"] = err
        return result
    for (_pid, _bid), stages in sorted(groups.items()):
        gb, li, be = stages.get("gyro_bias"), stages.get("linear_initial"), stages.get("batch_end")
        entry = {"pass_id": _pid, "err_deg": None, "err_mean": None, "err_max": None,
                 "err_rate_mean": None, "dbg": None, "dbg_implied": None, "v0_norm": None,
                 "g_norm": None, "v0_ref": None, "g_ref": None, "delta_v": None,
                 "resid": None, "system_finite": None}
        if li is not None:
            solution = li.get("solution")
            if (isinstance(solution, list) and len(solution) >= 6
                    and all(numeric(v) for v in solution[:3] + solution[-3:])):
                x = np.asarray(solution, float)
                entry["v0_norm"] = float(np.linalg.norm(x[0:3]))
                entry["g_norm"] = float(np.linalg.norm(x[-3:]))
            resid = li.get("normal_equation_relative_residual")
            entry["resid"] = float(resid) if numeric(resid) else None
            entry["system_finite"] = li.get("system_finite")
            frames = [v for v in (li.get("velocities") or [])
                      if isinstance(v, dict) and numeric(v.get("time"))
                      and isinstance(v.get("R_local_to_batch_column_major"), list)
                      and len(v["R_local_to_batch_column_major"]) == 9
                      and all(numeric(e) for e in v["R_local_to_batch_column_major"])]
            if len(frames) >= 2:
                ts = np.array([v["time"] for v in frames], float)
                Rs = [np.asarray(v["R_local_to_batch_column_major"], float).reshape(3, 3)
                      for v in frames]
                dt = np.diff(ts)
                err_deg, err_vec = [], []
                for i in range(len(ts) - 1):
                    Rrel = Rs[i].T @ Rs[i + 1]
                    Rimu = _imu_integrate_rot(t_imu, gyro, ts[i], ts[i + 1])
                    e = _rotvec(Rimu.T @ Rrel)
                    err_deg.append(math.degrees(float(np.linalg.norm(e))))
                    err_vec.append(e)
                err_deg, err_vec = np.asarray(err_deg), np.asarray(err_vec)
                valid = (dt > 1e-6) & np.isfinite(err_deg) & np.isfinite(err_vec).all(axis=1)
                if valid.any():
                    entry["err_deg"] = err_deg[valid]
                    entry["err_mean"] = float(err_deg[valid].mean())
                    entry["err_max"] = float(err_deg[valid].max())
                    rates = err_vec[valid] / dt[valid, None]
                    entry["err_rate_mean"] = rates.mean(0)
                    # 环节2 折算: solve_Rot_bias_gyro 的 LS 等价 dbg ≈ -2 * mean(err_vec/dt)
                    entry["dbg_implied"] = -2.0 * entry["err_rate_mean"]
        if gb is not None and isinstance(gb.get("solution"), list) and len(gb["solution"]) == 3 \
                and all(numeric(v) for v in gb["solution"]):
            entry["dbg"] = np.asarray(gb["solution"], float)
        if be is not None:
            if numeric(be.get("initial_speed")):
                entry["v0_ref"] = float(be["initial_speed"])
            if numeric(be.get("gravity_norm")):
                entry["g_ref"] = float(be["gravity_norm"])
            if numeric(be.get("delta_v")):
                entry["delta_v"] = float(be["delta_v"])
        result["passes"].append(entry)
    return result


def prepare_run(run, include_direct):
    finite_times = [r["time"] for r in run.frames if positive_time(r["time"])]
    run.origin = run.init_time
    if run.origin is not None:
        run.origin_kind = "init_done"
    elif finite_times:
        run.origin = finite_times[0]
        run.origin_kind = "first_frame_diagnostic_only"
        if include_direct and any(r["stage"] == "direct_lio" for r in run.frames):
            run.origin_kind = "direct_lio_first_frame_no_init_done"
            run.warn("direct_lio 无 init_done：首帧为时间原点，不是初始化对齐")
    high_time = -math.inf
    high_frame = -1
    segment_time = -math.inf
    segment_key = None
    previous = None
    for row in run.frames:
        stamp = row["time"]
        if run.schema == 2:
            # 计划第 4 节: 相同雷达时间跨轮重处理合法, 时间单调性按 (stage, pass_id)
            # 分段判定, 跨段回访不计入非单调; 帧序号全局递增, 仍全局判定。
            key = (row["stage"], row["pass_id"])
            if key != segment_key:
                segment_key = key
                segment_time = -math.inf
            time_ok = positive_time(stamp) and stamp > segment_time
            if positive_time(stamp):
                segment_time = max(segment_time, stamp)
        else:
            time_ok = positive_time(stamp) and stamp > high_time
            if positive_time(stamp):
                high_time = max(high_time, stamp)
        id_ok = row["frame"] > high_frame
        if not time_ok or not id_ok:
            run.counts["invalid_or_nonmonotonic"] += 1
        high_frame = max(high_frame, row["frame"])
        initialized = (row["init_seen"] is not None and positive_time(stamp)
                       and stamp >= row["init_seen"])
        row["eligible_lio"] = ((row["stage"] == "post_dynamic" and initialized)
                               or (include_direct and row["stage"] == "direct_lio"))
        row["relative_time"] = (stamp - run.origin if positive_time(stamp)
                                and run.origin is not None else math.nan)
        # Keep every row in input order: excluded rows must still break adjacency.
        row["in_lio_window"] = bool(0 <= row["relative_time"] <= run.lio_window)
        row["selected"] = bool(row["eligible_lio"] and row["in_lio_window"])
        row["valid_motion"] = bool(row["selected"] and time_ok and id_ok
                                   and not row["conflict"] and row["status"] == "post"
                                   and row["applied"] and row["committed"]
                                   and row["effective"] is not None and row["effective"] > 0)
        row["rpy_deg"] = rpy_degrees(vector(row, "quaternion_wxyz")).tolist()
        row["dt"] = math.nan
        row["delta_position"] = [math.nan] * 3
        row["delta_rpy_deg"] = [math.nan] * 3
        row["step_m"] = row["position_rate_mps"] = math.nan
        row["rotation_step_deg"] = math.nan
        row["connected"] = bool(previous is not None and row["valid_motion"]
                                and previous["valid_motion"] and not row["gap_before"]
                                and row["frame"] == previous["frame"] + 1
                                and row["stage"] == previous["stage"])
        if row["connected"]:
            dt = stamp - previous["time"]
            row["dt"] = dt
            delta = vector(row, "position") - vector(previous, "position")
            row["delta_position"] = delta.tolist()
            row["step_m"] = float(np.linalg.norm(delta))
            row["position_rate_mps"] = row["step_m"] / dt
            angles = np.array(row["rpy_deg"]) - previous["rpy_deg"]
            row["delta_rpy_deg"] = ((angles + 180) % 360 - 180).tolist()
            qa, qb = vector(row, "quaternion_wxyz"), vector(previous, "quaternion_wxyz")
            norms = np.linalg.norm(qa) * np.linalg.norm(qb)
            if np.isfinite(norms) and norms > 1e-12:
                row["rotation_step_deg"] = float(np.degrees(2*np.arccos(
                    np.clip(abs(np.dot(qa, qb)/norms), 0, 1))))
        previous = row
    if run.counts["invalid_or_nonmonotonic"]:
        note = ("（schema2 按 stage/pass 分段判定，跨轮重处理的时间回访合法）"
                if run.schema == 2 else "")
        run.warn(f"无效/重复/非单调时间或帧序号: {run.counts['invalid_or_nonmonotonic']}；"
                 f"不计算对应运动量{note}")
    excluded = sum(r["stage"] == "post_dynamic" and not r["eligible_lio"] for r in run.frames)
    if excluded:
        run.warn(f"{excluded} 个动态帧不满足事件先于帧且时间不早于 init_done，仅保留诊断")
    if not any(r["valid_motion"] for r in run.frames):
        run.warn("观察窗口内无有效后验帧；仍保留诊断，不生成虚假运动统计")


def finite_stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {"count": int(a.size), "mean": float(a.mean()) if a.size else None,
            "max": float(a.max()) if a.size else None,
            "p95": float(np.percentile(a, 95)) if a.size else None}


def window_summary(run):
    selected = [r["relative_time"] for r in run.frames if r["selected"]]
    valid = [r["relative_time"] for r in run.frames if r["valid_motion"]]

    def coverage(times):
        return {"first_s": min(times) if times else None,
                "last_s": max(times) if times else None,
                "span_s": max(times) - min(times) if times else 0.0}

    return {"requested_s": run.lio_window, "bounds_inclusive_s": [0.0, run.lio_window],
            "origin_kind": run.origin_kind, "selected_coverage": coverage(selected),
            "valid_coverage": coverage(valid),
            "outside_window_frames": sum(r["eligible_lio"] and not r["in_lio_window"]
                                         for r in run.frames)}


def schema2_metrics(run):
    """Initialization/batch/queue/RTK aggregates; reuses logged init_done, no recompute."""
    ends = [b for b in run.batches if b["stage"] == "batch_end"]
    solved = [b for b in run.batches if b["stage"] != "batch_end"]
    passes = [{
        "pass_id": b["pass_id"], "batch_id": b["batch_id"],
        "window_start": b["window_start"], "window_end": b["window_end"],
        "window_size": b["window_size"], "initial_speed": b["initial_speed"],
        "gravity_norm": b["gravity_norm"], "delta_v": b["delta_v"],
        "velocity_threshold": b["velocity_threshold"], "delta_v_reason": b["delta_v_reason"],
        "bias_a_source": b["bias_a_source"], "observations": b["observations"],
        "quality_reason": b["quality_reason"], "finite": b["finite"],
        "batch_wall_ms": b["batch_wall_ms"], "batch_algorithm_ms": b["batch_algorithm_ms"],
        "observer_ms": b["observer_ms"], "pass_wall_ms": b["pass_wall_ms"],
    } for b in ends]
    init_event = next((e for e in run.events if e["event"] == "init_done"), None)
    final = ends[-1] if ends else None
    solved_failures = sum(1 for b in solved
                          if not b["system_finite"] or b["decomposition_status"] != 0
                          or b["solve_status"] != 0)
    # 失败细分: 解/系统非有限是数据数值问题; 状态码非零(Eigen NumericalIssue 等)单独计数。
    solved_nonfinite = sum(1 for b in solved
                           if not b["system_finite"] or not b["solution_finite"])
    solved_status_nonzero = sum(1 for b in solved
                                if b["decomposition_status"] != 0 or b["solve_status"] != 0)
    finite_end = next((b for b in reversed(ends) if b["finite"]), None)

    def pass_result(b):
        return None if b is None else {
            "pass_id": b["pass_id"], "window_start": b["window_start"],
            "window_end": b["window_end"], "initial_velocity": b["initial_velocity"],
            "initial_speed": b["initial_speed"], "gravity": b["gravity"],
            "gravity_norm": b["gravity_norm"], "bias_g": b["bias_g"],
            "bias_a": b["bias_a"], "bias_a_source": b["bias_a_source"],
            "delta_v": b["delta_v"], "velocity_threshold": b["velocity_threshold"],
            "delta_v_reason": b["delta_v_reason"]}

    valid_lag = [r["lag_ms"] for r in run.frames
                 if r.get("lag_reason") == "valid" and numeric(r.get("lag_ms"))]
    rtk_rows = [r for r in run.frames if r["rtk_valid"]]
    rtk_speed_error = [abs(r["speed"] - r["rtk_speed"]) for r in rtk_rows
                       if numeric(r.get("speed")) and numeric(r.get("rtk_speed"))]
    # 交接质量: 批解有限轮数/首个非有限轮 + LIO 帧有效性, 定位"init_done 但状态为 NaN"的退化。
    finite_pass_ids = [b["pass_id"] for b in ends if b["finite"]]
    nonfinite_pass_ids = [b["pass_id"] for b in ends if not b["finite"]]
    post_rows = [r for r in run.frames if r["stage"] == "post_dynamic"]
    post_effective_zero = sum(1 for r in post_rows
                              if r["effective"] is not None and r["effective"] == 0)
    post_state_null = sum(1 for r in post_rows if not isinstance(r.get("velocity"), list)
                          or all(v is None for v in r["velocity"]))
    post_reasons = Counter(r["reason"] for r in post_rows)
    return {
        "run_id": run.run_id,
        "init_done_reason": init_event["reason"] if init_event else None,
        "pass_count": len(passes), "passes": passes,
        "final_pass": pass_result(final),
        "last_finite_pass": pass_result(finite_end),
        "batch_finite_pass_count": len(finite_pass_ids),
        "first_nonfinite_pass": min(nonfinite_pass_ids) if nonfinite_pass_ids else None,
        "post_dynamic_frames": len(post_rows),
        "post_effective_zero_frames": post_effective_zero,
        "post_state_null_frames": post_state_null,
        "post_reason_counts": dict(post_reasons),
        "batch_solved_observations": len(solved),
        "batch_solve_failures": solved_failures,
        "batch_solve_nonfinite": solved_nonfinite,
        "batch_solve_status_nonzero": solved_status_nonzero,
        "queue_heartbeats": len(run.queues),
        "queue_depth_max": {"lidar": max((q["lidar_queue"] for q in run.queues), default=None),
                            "imu": max((q["imu_queue"] for q in run.queues), default=None)},
        "lag_ms_stats": finite_stats(valid_lag),
        "rtk_valid_frames": len(rtk_rows),
        "rtk_speed_error_mps": finite_stats(rtk_speed_error),
    }


def run_summary(run):
    valid = [r for r in run.frames if r["valid_motion"]]
    steps = [r["step_m"] for r in run.frames if np.isfinite(r["step_m"])]
    summary = {"files": run.files, "counts": dict(run.counts), "warnings": run.messages,
               "events": run.events, "init_done": run.init_time,
               "time_origin": run.origin, "time_origin_kind": run.origin_kind,
               "lio_window": window_summary(run),
               "frames": len(run.frames), "selected_frames": sum(r["selected"] for r in run.frames),
               "valid_posteriors": len(valid), "status_counts": dict(Counter(r["status"] for r in run.frames)),
               "connected_path_length_m": sum(steps) if steps else None,
               "connected_steps": len(steps),
               "speed_mps": finite_stats([np.linalg.norm(vector(r, "velocity")) for r in valid]),
               "position_rate_mps": finite_stats([r["position_rate_mps"] for r in run.frames]),
               "diagnostics_all_frames": {key: finite_stats([r[key] for r in run.frames])
                                          for key in ("effective", "residual", "condition", "lag_ms",
                                                      "frame_ms", "diag_scope_ms", "icp_ms", "match_ms")}}
    if run.schema == 2:
        summary["schema"] = 2
        summary["initialization"] = schema2_metrics(run)
    return summary


def export_data(runs, output):
    summary = {"schema": 1, "notes": NOTES, "runs": {r.label: run_summary(r) for r in runs}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                   allow_nan=False), encoding="utf-8")
    rows = []
    for run in runs:
        for frame in run.frames:
            flat = {"run": run.label}
            for key, value in frame.items():
                if isinstance(value, list):
                    for i, number in enumerate(value):
                        flat[f"{key}_{i}"] = math.nan if number is None else number
                elif isinstance(value, dict):
                    # Schema2 nested state snapshots stay verbatim JSON in one cell.
                    flat[key] = json.dumps(value, ensure_ascii=False, allow_nan=False)
                else:
                    flat[key] = math.nan if value is None else value
            rows.append(flat)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    # 同一字段可为 null 标量或定长数组(如 rtk_velocity), 列集合取全行并集,
    # 缺失单元格留空, 不再以首行形状假设表头。
    with (output / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def series(run, key, motion=False, order=False, frames=None, breaks=None):
    x, y = [], []
    previous = None
    rows = run.frames if frames is None else frames
    for index, row in enumerate(rows):
        value = np.asarray(row[key], dtype=float)
        split = motion and not row["connected"]
        if breaks is not None and previous is not None and breaks(previous, row):
            split = True
        if split:
            x.append(math.nan)
            y.append(np.full_like(value, np.nan))
        x.append(index if order else row["relative_time"])
        y.append(value if not motion or row["valid_motion"] else np.full_like(value, np.nan))
        previous = row
    return np.asarray(x), np.asarray(y)


def init_series_break(previous, row):
    """Split lines at pass resets, stage changes and input gaps (no interpolation)."""
    return (previous["pass_id"] != row["pass_id"] or previous["stage"] != row["stage"]
            or bool(row.get("gap_before")))


SECTION_NAMES = ("overview", "initialization", "lio")

SECTION_PAGES = {
    "overview": ("summary",),
    "initialization": ("lgo_quality", "batch_convergence", "reprocess_quality",
                        "imu_excitation", "stage_analysis"),
    "lio": ("motion_rtk", "registration", "gravity_biases", "covariance", "timing"),
}

INIT_STAGES = ("initial_lgo", "cache_reprocess")
LIO_STAGES = ("post_dynamic", "direct_lio")
STAGE_ORDER = {name: index for index, name in enumerate(INIT_STAGES + LIO_STAGES)}
STATUS_ORDER = {name: index for index, name in enumerate(
    ("post", "skipped", "partial_applied", "predict"))}


class PageContext:
    """Plot-time state shared by pages: axis mode, pass filter, axes registry."""

    def __init__(self, fig, runs, order_mode=False, pass_filter=None):
        self.fig, self.runs = fig, runs
        self.order_mode = order_mode    # False: data time; True: processing order.
        self.pass_filter = pass_filter  # None: all passes; int: single init pass.
        self.axes = []
        self.foot = None

    @property
    def xlabel(self):
        return ("Processing order (frames, log input order)" if self.order_mode
                else "Relative data time (s); origin in summary")

    def stage_frames(self, run, stage):
        rows = [r for r in run.frames if r["stage"] == stage]
        if self.pass_filter is not None and stage in INIT_STAGES:
            rows = [r for r in rows if r["pass_id"] == self.pass_filter]
        return rows

    def init_frames(self, run):
        if run.schema != 2:
            return []
        return [r for stage in INIT_STAGES for r in self.stage_frames(run, stage)]

    def lio_frames(self, run):
        return [r for r in run.frames if r["stage"] in LIO_STAGES]

    def passes(self, run):
        ends = [b for b in run.batches if b["stage"] == "batch_end"]
        if self.pass_filter is not None:
            ends = [b for b in ends if b["pass_id"] == self.pass_filter]
        return ends

    def solved(self, run):
        rows = [b for b in run.batches if b["stage"] != "batch_end"]
        if self.pass_filter is not None:
            rows = [b for b in rows if b["pass_id"] == self.pass_filter]
        return rows

    def pairs(self, frames_of):
        return [(run, frames_of(run)) for run in self.runs]


def per_pass_aggregate(frames, field, component=None):
    """Collapse reprocessing frames to one (pass_id, mean) point per pass.

    Reprocessing revisits the same cached window once per pass, so frame-level
    series against data time stack 20+ lines. Mean per pass keeps the per-pass
    trend as a single line; use the pass filter for frame-level detail.
    """
    groups = {}
    for row in frames:
        value = row.get(field)
        if isinstance(value, list):
            if component is None or not (0 <= component < len(value)):
                continue
            value = value[component]
        if not numeric(value):
            continue
        groups.setdefault(row["pass_id"], []).append(value)
    return [(pid, sum(values) / len(values)) for pid, values in sorted(groups.items())]


def _page_axis(ctx, index, title, grid=(3, 2), projection=None, xlabel=None):
    ax = ctx.fig.add_subplot(*grid, index, projection=projection)
    ctx.axes.append(ax)
    ax._drift_time = projection is None
    ax.set_title(title, fontsize=FONT["title"])
    ax.set_xlabel(xlabel if xlabel is not None else ctx.xlabel, fontsize=FONT["text"])
    ax.grid(True, alpha=.25)
    return ax


def _empty_note(ax):
    ax.text(.5, .5, "No data available (missing or unsupported in this log)",
            ha="center", va="center", transform=ax.transAxes, fontsize=FONT["note"])
    ax.set_xticks([])
    ax.set_yticks([])


def _null_note(ax):
    """Field exists in the log but every sample is null: state never populated."""
    ax.text(.5, .5, "Field logged as null in every row: estimator state is\n"
            "non-finite or was not serialized; see overview for batch\n"
            "finiteness and LIO frame validity",
            ha="center", va="center", transform=ax.transAxes, fontsize=FONT["note"])
    ax.set_xticks([])
    ax.set_yticks([])


def _panel(ctx, index, title, keys, pairs, motion=False, symlog=False, labels=None,
           grid=(3, 2), breaks=None):
    """Time-series panel; rows missing a key are skipped, never zero-filled."""
    ax = _page_axis(ctx, index, title, grid)
    plotted = 0
    for run, frames in pairs:
        for key in keys:
            usable = [r for r in frames if key in r]
            if not usable:
                continue
            x, y = series(run, key, motion, ctx.order_mode, usable, breaks)
            if y.ndim == 1:
                ax.plot(x, y, '.-', ms=2, lw=.8, label=f"{run.label}:{key}")
                plotted += 1
            else:
                for j in range(y.shape[1]):
                    name = labels[j] if labels and j < len(labels) else str(j)
                    ax.plot(x, y[:, j], '.-', ms=2, lw=.8, label=f"{run.label}:{name}")
                    plotted += 1
    if plotted:
        ax.legend(fontsize=FONT["legend"], loc="best", ncol=2)
        if symlog:
            _symlog_axis(ax)
        if not any(np.isfinite(np.asarray(line.get_ydata(), dtype=float)).any()
                   for line in ax.lines):
            _null_note(ax)  # series exist but every sample is null/NaN
    else:
        _empty_note(ax)
    return ax


def _pass_panel(ctx, index, title, fields, grid=(3, 2)):
    """One marker per completed pass; x = pass id (processing-order view)."""
    ax = _page_axis(ctx, index, title, grid, xlabel="Initialization pass id")
    plotted = 0
    for run in ctx.runs:
        for field in fields:
            pts = [(b["pass_id"], b[field]) for b in ctx.passes(run) if numeric(b.get(field))]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, '.-', ms=6, lw=.8, label=f"{run.label}:{field}")
            plotted += 1
    if plotted:
        ax.legend(fontsize=FONT["legend"], loc="best", ncol=2)
    else:
        _empty_note(ax)
    return ax


def _pass_trend_panel(ctx, index, title, field, component=None, label=None, grid=(3, 2)):
    """Per-pass aggregated frame statistics; one line per run, x = pass id."""
    ax = _page_axis(ctx, index, title, grid, xlabel="Initialization pass id")
    plotted = 0
    for run in ctx.runs:
        pts = per_pass_aggregate(ctx.init_frames(run), field, component)
        if not pts:
            continue
        xs, ys = zip(*pts)
        name = label or (f"{field}[{component}]" if component is not None else field)
        ax.plot(xs, ys, '.-', ms=6, lw=1, label=f"{run.label}:{name}")
        plotted += 1
    if plotted:
        ax.legend(fontsize=FONT["legend"], loc="best", ncol=2)
    else:
        _empty_note(ax)
    return ax


def _fmt(value, digits=3):
    return f"{value:.{digits}f}" if numeric(value) else str(value)


def _vector_pass_panel(ctx, index, title, field, labels=("x", "y", "z"), grid=(3, 2)):
    """Per-pass vector components from batch_end rows; x = pass id."""
    ax = _page_axis(ctx, index, title, grid, xlabel="Initialization pass id")
    plotted = 0
    for run in ctx.runs:
        for j, label in enumerate(labels):
            pts = []
            for b in ctx.passes(run):
                value = b.get(field)
                if (isinstance(value, list) and len(value) == len(labels)
                        and numeric(value[j])):
                    pts.append((b["pass_id"], value[j]))
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, '.-', ms=6, lw=.8, label=f"{run.label}:{label}")
            plotted += 1
    if plotted:
        ax.legend(fontsize=FONT["legend"], loc="best", ncol=3)
    else:
        _empty_note(ax)
    return ax


def _text_panel(ctx, index, title, lines, grid=(3, 2)):
    ax = ctx.fig.add_subplot(*grid, index)
    ctx.axes.append(ax)
    ax.set_title(title, fontsize=FONT["title"])
    ax.axis("off")
    ax.text(.02, .98, "\n".join(lines), va="top", fontsize=FONT["text"])
    return ax

def draw_overview(ctx):
    grid = (3, 2)
    lines = []
    for run in ctx.runs:
        summary = run_summary(run)
        head = f"{run.label}: schema {run.schema}"
        if run.run_id:
            head += f", run_id {run.run_id}"
        lines.append(head)
        lines.append(f"frames {summary['frames']} | selected {summary['selected_frames']} | "
                     f"valid posteriors {summary['valid_posteriors']}")
        lines.append("init_done: " + (f"observed at data time {run.init_time:g} s"
                                       if run.init_time is not None
                                       else "NOT observed in this log"))
        if run.schema == 2:
            metrics = summary["initialization"]
            lines.append(f"init_done reason {metrics['init_done_reason']} | passes "
                         f"{metrics['pass_count']} | batch finite passes "
                         f"{metrics['batch_finite_pass_count']}/{metrics['pass_count']}"
                         + (f" (NaN from pass {metrics['first_nonfinite_pass']})"
                            if metrics["first_nonfinite_pass"] is not None else ""))
            lines.append(f"post_dynamic frames {metrics['post_dynamic_frames']} | "
                         f"effective=0 on {metrics['post_effective_zero_frames']} | "
                         f"state null on {metrics['post_state_null_frames']}")
        lines.append(f"warnings {len(run.messages)}")
        lines.append("")
    _text_panel(ctx, 1, "Run summary (terminal mirror; no ground truth)", lines, grid)

    ax = _page_axis(ctx, 2, "Stage timeline (frame stage; dashed line = init_done)", grid)
    for run in ctx.runs:
        xs = [i if ctx.order_mode else r["relative_time"]
              for i, r in enumerate(run.frames)]
        ys = [STAGE_ORDER.get(r["stage"], math.nan) for r in run.frames]
        ax.plot(xs, ys, '.', ms=3, lw=.5, label=run.label)
        if run.init_time is not None and run.origin is not None and not ctx.order_mode:
            ax.axvline(run.init_time - run.origin, color="k", ls="--", lw=.8)
    ax.set_yticks(range(len(STAGE_ORDER)))
    ax.set_yticklabels(list(STAGE_ORDER), fontsize=FONT["tick"])
    ax.legend(fontsize=FONT["legend"], loc="best")

    lines = []
    for run in ctx.runs:
        lines.append(f"{run.label}:")
        if run.schema == 2:
            metrics = schema2_metrics(run)
            final = metrics["final_pass"]
            if final:
                lines.append(f"final pass {final['pass_id']}: initial_speed "
                             f"{_fmt(final['initial_speed'])} m/s at window start, "
                             f"delta_v {_fmt(final['delta_v'])} (threshold "
                             f"{_fmt(final['velocity_threshold'])})")
            else:
                lines.append("no completed batch pass (initialization unfinished)")
            finite = metrics["last_finite_pass"]
            if finite and (not final or finite["pass_id"] != final["pass_id"]):
                lines.append(f"last finite pass {finite['pass_id']}: initial_speed "
                             f"{_fmt(finite['initial_speed'])} m/s")
        else:
            lines.append("schema1 log: batch/handoff details require schema2")
        first_lio = next((r for r in run.frames if r["stage"] == "post_dynamic"), None)
        if first_lio is not None:
            speed = float(np.linalg.norm(vector(first_lio, "velocity")))
            lines.append(f"first LIO frame {first_lio['frame']} at rel "
                         f"{first_lio['relative_time']:.3f} s: |velocity| "
                         f"{_fmt(speed)} m/s (filter state, not batch initial speed), "
                         f"effective {first_lio['effective']}")
        lines.append("")
    _text_panel(ctx, 3, "Handoff boundary (batch result vs first LIO filter state)", lines, grid)

    ax = _page_axis(ctx, 4, "Data integrity counters (per run)", grid)
    integrity = ["bad_lines", "duplicates", "conflicts", "invalid_or_nonmonotonic",
                 "skipped", "partial_applied", "predict"]
    width = .8 / max(len(ctx.runs), 1)
    for k, run in enumerate(ctx.runs):
        values = [run.counts.get(key, 0) for key in integrity[:4]]
        values += [sum(1 for r in run.frames if r["status"] == key)
                   for key in integrity[4:]]
        ax.bar([j + k * width for j in range(len(integrity))], values,
               width=width, label=run.label)
    ax.set_xticks([j + width * (len(ctx.runs) - 1) / 2 for j in range(len(integrity))])
    ax.set_xticklabels(integrity, rotation=30, fontsize=FONT["tick"], ha="right")
    ax.legend(fontsize=FONT["legend"], loc="best")

    _panel(ctx, 5, "LIO effective points after init_done (0 = no map match)", ["effective"],
           [(run, [r for r in run.frames
                   if r["stage"] in LIO_STAGES and r["selected"]]) for run in ctx.runs],
           grid=grid)

    lines = []
    for run in ctx.runs:
        if run.schema == 2:
            metrics = schema2_metrics(run)
            reasons = metrics.get("post_reason_counts", {})
            lines.append(f"{run.label}: post_dynamic reasons " +
                         (", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
                          if reasons else "none"))
            lines.append(f"batch solve failures {metrics['batch_solve_failures']} "
                         f"(nonfinite {metrics['batch_solve_nonfinite']}, status "
                         f"{metrics['batch_solve_status_nonzero']})")
            lines.append(f"RTK valid frames {metrics['rtk_valid_frames']}")
        lines.append("")
    _text_panel(ctx, 6, "Degeneration pointers", lines, grid)
    ctx.foot = ("Initialization pass boundaries break lines; the same cached data may be "
                "re-estimated across passes. RTK comparison is magnitude-only before "
                "ENU alignment.")


def draw_init_lgo_quality(ctx):
    grid = (3, 2)
    pairs = ctx.pairs(lambda run: ctx.stage_frames(run, "initial_lgo"))
    _panel(ctx, 1, "Point counts (raw input / undistorted / downsampled)",
           ["raw_points", "undistorted_points", "points"], pairs, grid=grid)
    _panel(ctx, 2, "Downsampled / effective points", ["points", "effective"],
           pairs, grid=grid)
    _panel(ctx, 3, "Effective ratio (effective / points; blank when denominator is zero)",
           ["effective_ratio"], pairs, grid=grid)
    _panel(ctx, 4, "Registration iterations / nearest-neighbor searches",
           ["iterations", "searches"], pairs, grid=grid)
    _panel(ctx, 5, "Mean point-to-plane residual (m) / lidar noise", ["residual", "cov_lidar"],
           pairs, grid=grid)
    _text_panel(ctx, 6, "Stage notes", [
        "First-pass LGO: frame-wise laser odometry over the accumulation window,",
        "before the first inertial batch solve. Point counts show how much data the",
        "window actually accumulated; effective points show how much of it matched",
        "the seeded map. Use batch_convergence for what the solver did with it.",
    ], grid)
    ctx.foot = ("First LGO is frame-wise laser odometry and mapping before the first "
                "inertial batch solve; it is a separate stage, not one batch solve.")


def draw_init_batch_convergence(ctx):
    grid = (3, 2)
    _pass_panel(ctx, 1, "Batch initial speed per pass (m/s; window start)",
                ["initial_speed"], grid)
    _pass_panel(ctx, 2, "Adjacent-pass speed change vs threshold (convergence criterion)",
                ["delta_v", "velocity_threshold"], grid)
    _vector_pass_panel(ctx, 3, "Gravity components per pass (m/s^2; normalized |G|=9.8, "
                       "direction is the estimate)", "gravity", grid=grid)
    _vector_pass_panel(ctx, 4, "Gyroscope bias per pass (rad/s)", "bias_g", grid=grid)
    _vector_pass_panel(ctx, 5, "Accelerometer bias per pass (m/s^2)",
                       "bias_a", grid=grid)
    lines = []
    for run in ctx.runs:
        if run.schema != 2:
            lines.append(f"{run.label}: schema1 log; batch details require schema2")
            continue
        metrics = schema2_metrics(run)
        lines.append(f"{run.label}:")
        if metrics["first_nonfinite_pass"] is not None:
            lines.append(f"batch solutions non-finite from pass "
                         f"{metrics['first_nonfinite_pass']} "
                         f"({metrics['batch_finite_pass_count']}/{metrics['pass_count']} "
                         f"passes finite)")
        else:
            lines.append(f"all {metrics['pass_count']} passes finite")
        for b in ctx.passes(run):
            lines.append(f"pass {b['pass_id']}: quality {b.get('quality_reason')}, "
                         f"finite {b.get('finite')}, bias_a_source "
                         f"{b.get('bias_a_source')}, delta_v_reason "
                         f"{b.get('delta_v_reason')}")
        lines.append("")
    _text_panel(ctx, 6, "Per-pass quality and finiteness", lines, grid)
    ctx.foot = ("Internal gravity refinements are batch sub-steps, not extra passes. "
                "Gravity is re-normalized to |G| each pass; only its components move.")


def draw_init_reprocess_quality(ctx):
    grid = (3, 2)
    if ctx.pass_filter is not None:
        pairs = ctx.pairs(lambda run: [r for r in run.frames if r["stage"] in INIT_STAGES])
        _panel(ctx, 1, f"Pass {ctx.pass_filter}: points / effective (frame level)",
               ["points", "effective"], pairs, grid=grid, breaks=init_series_break)
        _panel(ctx, 2, f"Pass {ctx.pass_filter}: effective ratio (frame level)",
               ["effective_ratio"], pairs, grid=grid, breaks=init_series_break)
        _panel(ctx, 3, f"Pass {ctx.pass_filter}: residual (m) / lidar noise (frame level)",
               ["residual", "cov_lidar"], pairs, grid=grid, breaks=init_series_break)
        _panel(ctx, 4, f"Pass {ctx.pass_filter}: iterations / searches (frame level)",
               ["iterations", "searches"], pairs, grid=grid, breaks=init_series_break)
        _panel(ctx, 5, f"Pass {ctx.pass_filter}: frame phase timing (ms; frame/imu/down)",
               ["frame_ms", "imu_ms", "down_ms"], pairs, grid=grid, breaks=init_series_break)
        _text_panel(ctx, 6, "Frame-level detail", [
            f"Showing pass {ctx.pass_filter} only (top Pass filter).",
            "Each pass re-processes the same cached window; these are the",
            "per-frame values inside that single pass.",
        ], grid)
        ctx.foot = ("Frame-level curves for the selected pass; the pass filter button "
                    "at the top cycles through passes.")
        return
    _pass_trend_panel(ctx, 1, "Mean effective points per pass (all reprocessing passes)",
                      "effective", label="effective")
    _pass_trend_panel(ctx, 2, "Mean effective ratio per pass", "effective_ratio",
                      label="effective_ratio")
    _pass_trend_panel(ctx, 3, "Mean residual per pass (m)", "residual", label="residual")
    _pass_trend_panel(ctx, 4, "Mean iterations per pass", "iterations", label="iterations")
    _pass_panel(ctx, 5, "Pass cost (ms; wall covers frames+batch+diagnostics)",
                ["pass_wall_ms", "batch_wall_ms", "batch_algorithm_ms"], grid)
    _text_panel(ctx, 6, "How to read this page", [
        "Reprocessing revisits the same cached window once per pass, so this page",
        "shows per-pass means as a single line instead of 20 overlapping",
        "frame-level lines. Use the top Pass filter to inspect one pass at",
        "frame level (points, residual, timing).",
    ], grid)
    ctx.foot = ("Per-pass means aggregate initial_lgo and cache_reprocess frames; "
                "the same data window is re-estimated each pass.")


def draw_init_imu_excitation(ctx):
    """IMU excitation proxies from legacy_logs/imu.txt; data time axis, no pass filter."""
    grid = (3, 2)
    data = [(run, load_imu_excitation(run)) for run in ctx.runs]
    ok = [(run, d) for run, d in data if d["error"] is None]

    def _shade_window(ax):
        for run, d in ok:
            if d.get("win_t0") is not None:
                ax.axvspan(d["win_t0"], d["win_t1"], color="0.85", alpha=0.55,
                           lw=0, zorder=0)

    def _excite_axis(index, title, value, name, ref=None):
        ax = _page_axis(ctx, index, title, grid)
        plotted = 0
        for run, d in ok:
            ax.plot(d["t"], d[value], '.-', ms=2, lw=.8, label=f"{run.label}:{name}")
            plotted += 1
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=.8)
        _shade_window(ax)
        if plotted:
            ax.legend(fontsize=FONT["legend"], loc="best")
        else:
            _empty_note(ax)
        return ax

    _excite_axis(1, "De-gravity linear acceleration magnitude (m/s^2)", "dyn_mag", "acc")
    _excite_axis(2, "Sustained acceleration: 1s rolling mean magnitude (m/s^2)", "roll_mag",
                 "sustained")
    ax = _page_axis(ctx, 3, "Sustained acceleration components (1s rolling mean, m/s^2)",
                    grid)
    plotted = 0
    for run, d in ok:
        for j, axis in enumerate("xyz"):
            ax.plot(d["t"], d["roll_xyz"][:, j], '.-', ms=2, lw=.8,
                    label=f"{run.label}:{axis}")
            plotted += 1
    _shade_window(ax)
    if plotted:
        ax.legend(fontsize=FONT["legend"], loc="best", ncol=3)
    else:
        _empty_note(ax)
    _excite_axis(4, "Gyroscope magnitude (rad/s; rotational excitation)", "gyro_mag", "gyro")
    _excite_axis(5, "Specific force norm (g; 1.000 = gravity only, no linear motion)",
                 "force_norm_g", "|f|", ref=1.0)
    lines = []
    for run, d in data:
        lines.append(f"{run.label}:")
        if d["error"] is not None:
            lines.append(f"  {d['error']}")
            lines.append("")
            continue
        dyn, gyro = d["dyn_mag"], d["gyro_mag"]
        lines.append(f"  {d['count']} samples, span {d['span_s']:.1f}s, "
                     f"accel unit {d['unit']}, "
                     f"gravity window stationary: {d['stationary']}")
        if d.get("replayed_logs"):
            lines.append(f"  {d['replayed_logs']} re-logged window replays removed "
                         "(dynamic-init pass loop re-logs the solver window; "
                         "stats use de-duplicated samples)")
        lines.append(f"  de-gravity acc mean {dyn.mean():.2f} / p95 "
                     f"{np.percentile(dyn, 95):.2f} / max {dyn.max():.2f} m/s^2")
        lines.append(f"  gyro mean {gyro.mean():.3f} / p95 "
                     f"{np.percentile(gyro, 95):.3f} / max {gyro.max():.3f} rad/s")
        if d.get("win_t0") is not None:
            lines.append(f"  solver window [{d['win_t0']:.2f}, {d['win_t1']:.2f}] s "
                         f"(shaded), {d['win_count']} samples in window")
            if d.get("win_acc_mean") is not None:
                lines.append(f"  window acc mean {d['win_acc_mean']:.2f} / p95 "
                             f"{d['win_acc_p95']:.2f} / max {d['win_acc_max']:.2f} m/s^2, "
                             f"force std {d['win_force_std_g']:.3f} g")
                lines.append(f"  window gyro mean {d['win_gyro_mean']:.3f} / max "
                             f"{d['win_gyro_max']:.3f} rad/s")
        lines.append("")
    lines += ["How to read: the shaded band is the solver's actual IMU window",
              "(DLI_BATCH window_start/end); judge initialization excitation",
              "inside the band, not over the full imu.txt span. Magnitude alone",
              "does not guarantee identifiability, and violent rotation also",
              "degrades the lidar odometry constraints feeding the solve",
              "(cross-check the first-frame effective ratio). See batch_convergence",
              "for the solver verdict (residual_reason, gravity drift)."]
    _text_panel(ctx, 6, "Excitation summary", lines, grid)
    ctx.foot = ("IMU excitation proxies from legacy_logs/imu.txt (raw IMU, logged units "
                "auto-detected). Time axis shares the summary origin (init_done when "
                "present, so initialization data sits at negative times); the pass filter "
                "and order-axis mode do not apply to this page.")





def draw_init_stage_analysis(ctx):
    """逐环节剖析: ICP-IMU 旋转一致性 / gyro_bias 欠拟合 / 线性解演化。"""
    grid = (3, 2)
    data = [(run, load_stage_analysis(run)) for run in ctx.runs]
    ok = [(run, d) for run, d in data if d["error"] is None]

    def _pass_points(d, key, norm=False):
        pts = []
        for p in d["passes"]:
            value = p.get(key)
            if value is None:
                continue
            pts.append((p["pass_id"], float(np.linalg.norm(value)) if norm else float(value)))
        return pts

    # 1) 环节1: 每 pass 平均/最大逐帧 ICP-IMU 旋转误差 (跨 pass 系统性判据)
    ax = _page_axis(ctx, 1, "Stage 1: per-frame ICP-IMU rotation error, mean/max per pass (deg)",
                    grid, xlabel="Initialization pass id")
    plotted = 0
    for run, d in ok:
        for key, style, name in (("err_mean", ".-", "mean"), ("err_max", ".", "max")):
            pts = _pass_points(d, key)
            if pts:
                ax.plot(*zip(*pts), style, ms=6 if key == "err_mean" else 3,
                        lw=1 if key == "err_mean" else 0, alpha=.5 if key == "err_max" else 1,
                        label=f"{run.label}:{name}")
                plotted += 1
    if plotted:
        ax.legend(fontsize=FONT["legend"], loc="best", ncol=2)
    else:
        _empty_note(ax)

    # 2) 环节1: 逐帧误差曲线 (pass filter 选定 pass; 默认最早有数据的 pass)
    candidates = sorted({p["pass_id"] for _run, d in ok for p in d["passes"]
                         if p["err_deg"] is not None})
    sel = (ctx.pass_filter if ctx.pass_filter in candidates
           else (candidates[0] if candidates else None))
    ax = _page_axis(ctx, 2, f"Stage 1: per-frame error curve, pass {sel} (deg)"
                    if sel is not None else "Stage 1: per-frame error curve (deg)",
                    grid, xlabel="Frame index in batch window")
    plotted = 0
    if sel is not None:
        for run, d in ok:
            for p in d["passes"]:
                if p["pass_id"] == sel and p["err_deg"] is not None:
                    ax.plot(np.arange(len(p["err_deg"])), p["err_deg"], '.-', ms=4, lw=.8,
                            label=f"{run.label}:pass {sel}")
                    plotted += 1
                    break
    if plotted:
        ax.legend(fontsize=FONT["legend"], loc="best")
    else:
        _empty_note(ax)
    # 3) 环节2: 实际 delta_bg 解 vs 误差折算的应有修正量 (欠拟合判据)
    ax = _page_axis(ctx, 3, "Stage 2: gyro-bias update |delta_bg|, solved vs implied (rad/s)",
                    grid, xlabel="Initialization pass id")
    plotted = 0
    for run, d in ok:
        pts = _pass_points(d, "dbg", norm=True)
        if pts:
            ax.plot(*zip(*pts), '.-', ms=6, lw=1, label=f"{run.label}:solved")
            plotted += 1
        pts = _pass_points(d, "dbg_implied", norm=True)
        if pts:
            ax.plot(*zip(*pts), '.--', ms=4, lw=1, alpha=.7, label=f"{run.label}:implied")
            plotted += 1
    if plotted:
        ax.legend(fontsize=FONT["legend"], loc="best", ncol=2)
        _symlog_axis(ax)
    else:
        _empty_note(ax)

    # 4) 环节3: 线性解 |v0| / |g| 与精化后对照的逐 pass 演化 (发散判据)
    ax = _page_axis(ctx, 4, "Stage 3: linear-solve |v0| / |g| per pass, refined |v0| for reference",
                    grid, xlabel="Initialization pass id")
    plotted = 0
    for run, d in ok:
        for key, style, name in (("v0_norm", ".-", "|v0| linear"),
                                 ("g_norm", ".-", "|g| linear"),
                                 ("v0_ref", ".", "|v0| refined")):
            pts = _pass_points(d, key)
            if pts:
                ax.plot(*zip(*pts), style, ms=6 if style == ".-" else 3, lw=1,
                        alpha=1 if style == ".-" else .6, label=f"{run.label}:{name}")
                plotted += 1
    if plotted:
        ax.legend(fontsize=FONT["legend"], loc="best", ncol=2)
        _symlog_axis(ax)
        ax.axhline(9.805, color="k", ls=":", lw=.8)
    else:
        _empty_note(ax)

    # 5) 环节1: 误差向量/dt 的均值分量 (y 主导 = 系统性旋转不一致)
    ax = _page_axis(ctx, 5, "Stage 1: mean error-vector components, err/dt (rad/s)",
                    grid, xlabel="Initialization pass id")
    plotted = 0
    for run, d in ok:
        for j, axis_name in enumerate("xyz"):
            pts = [(p["pass_id"], float(p["err_rate_mean"][j])) for p in d["passes"]
                   if p["err_rate_mean"] is not None]
            if pts:
                ax.plot(*zip(*pts), '.-', ms=6, lw=.8, label=f"{run.label}:{axis_name}")
                plotted += 1
    if plotted:
        ax.legend(fontsize=FONT["legend"], loc="best", ncol=3)
    else:
        _empty_note(ax)

    # 6) 环节汇总表
    lines = []
    for run, d in data:
        lines.append(f"{run.label}:")
        if d["error"] is not None:
            lines.append(f"  {d['error']}")
            lines.append("")
            continue
        rot = [p for p in d["passes"] if p["err_mean"] is not None]
        lines.append(f"  {len(d['passes'])} passes analyzed, rotation data on {len(rot)}")
        if rot:
            lines.append(f"  stage 1 error: pass {rot[0]['pass_id']} mean "
                         f"{rot[0]['err_mean']:.2f} deg/frame; all-pass mean "
                         f"{sum(p['err_mean'] for p in rot) / len(rot):.2f} deg/frame")
        p0 = d["passes"][0] if d["passes"] else None
        pl = d["passes"][-1] if d["passes"] else None
        if p0 is not None and p0.get("dbg") is not None and p0.get("dbg_implied") is not None:
            lines.append(f"  stage 2 pass {p0['pass_id']}: |delta_bg| solved "
                         f"{np.linalg.norm(p0['dbg']):.4f} vs implied "
                         f"{np.linalg.norm(p0['dbg_implied']):.4f} rad/s "
                         "(1-2 orders gap = under-fit)")
        if p0 is not None and pl is not None:
            def _fmt(v):
                return f"{v:.3f}" if v is not None else "-"
            lines.append(f"  stage 3 |v0| linear: pass {p0['pass_id']} {_fmt(p0['v0_norm'])} -> "
                         f"pass {pl['pass_id']} {_fmt(pl['v0_norm'])} m/s; |g| linear "
                         f"{_fmt(p0['g_norm'])} -> {_fmt(pl['g_norm'])} (9.805 expected)")
            tail = f"refined |v0|: {_fmt(p0['v0_ref'])} -> {_fmt(pl['v0_ref'])}"
            if pl["delta_v"] is not None:
                tail += f"; delta_v last {pl['delta_v']:.3f}"
            if pl["resid"] is not None:
                tail += f"; residual last {pl['resid']:.1e}"
            lines.append(f"  {tail}")
        lines.append("")
    lines += ["How to read: stage 1 both runs at a similar level means the ICP-IMU",
              "rotation mismatch is systematic, not the pass-fork source; a solved",
              "delta_bg far below the implied one is stage-2 under-fit; |v0| growing",
              "pass by pass with near-zero residuals is an ill-conditioned stage-3",
              "solve (wrong equations solved exactly)."]
    _text_panel(ctx, 6, "Stage summary", lines, grid)
    ctx.foot = ("Stage analysis ports batch_stage_analysis.py: ICP rotations from DLI_BATCH "
                "linear_initial velocities, IMU rotation integrates the raw legacy gyro with "
                "bg=0. Pass-level panels always show every pass; the pass filter only selects "
                "the frame-level panel.")


def draw_lio_motion_rtk(ctx):
    grid = (3, 2)
    ax = ctx.fig.add_subplot(*grid, 1, projection="3d")
    ctx.axes.append(ax)
    ax._drift_time = False
    for run in ctx.runs:
        _, pos = series(run, "position", True, ctx.order_mode, ctx.lio_frames(run))
        ax.plot(*pos.T, '.-', ms=2, lw=.6, label=run.label)
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
           title="Native world/IMU trajectory (NOT accuracy)")
    _panel(ctx, 2, "Filter velocity (m/s; posterior, not batch initial speed)",
           ["velocity"], ctx.pairs(ctx.lio_frames), motion=True, labels=list("XYZ"),
           grid=grid)
    _panel(ctx, 3, "Filter speed vs RTK speed (m/s; magnitude only before ENU alignment)",
           ["speed", "rtk_speed"],
           [(run, [r for r in ctx.lio_frames(run)
                   if r.get("rtk_valid") and r.get("rtk_speed") is not None])
            for run in ctx.runs], grid=grid)
    _panel(ctx, 4, "Adjacent position rate (m/s; not truth error)", ["position_rate_mps"],
           ctx.pairs(ctx.lio_frames), motion=True, grid=grid)
    _panel(ctx, 5, "Quaternion-derived RPY (deg; Euler singularities)", ["rpy_deg"],
           ctx.pairs(ctx.lio_frames), motion=True, labels=list("RPY"), grid=grid)
    ax = _page_axis(ctx, 6, "Speed error vs RTK (m/s; magnitude only)", grid)
    plotted = 0
    for run in ctx.runs:
        pts = [(r["relative_time"], abs(r["speed"] - r["rtk_speed"]))
               for r in ctx.lio_frames(run)
               if numeric(r.get("speed")) and numeric(r.get("rtk_speed"))
               and r.get("rtk_valid")]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, '.-', ms=2, lw=.8, label=run.label)
            plotted += 1
    if plotted:
        ax.legend(fontsize=FONT["legend"], loc="best")
    else:
        _empty_note(ax)
    ctx.foot = ("Default observation window is the configured LIO seconds after "
                "init_done; broken lines mark resets, gaps and invalid posteriors.")


def draw_lio_gravity_biases(ctx):
    grid = (2, 2)
    pairs = ctx.pairs(ctx.lio_frames)
    _panel(ctx, 1, "Gravity components (m/s^2)", ["gravity"], pairs, labels=list("XYZ"),
           grid=grid)
    ax = _page_axis(ctx, 2, "Gravity direction change vs initialization result (deg)", grid)
    plotted = 0
    for run in ctx.runs:
        reference = None
        for b in reversed(ctx.passes(run)):
            if isinstance(b.get("gravity"), list):
                candidate = np.asarray(b["gravity"], float)
                if np.isfinite(candidate).all() and np.linalg.norm(candidate) > 1e-9:
                    reference = candidate
                    break
        rows = ctx.lio_frames(run)
        if reference is None and rows and isinstance(rows[0].get("gravity"), list):
            candidate = np.asarray(rows[0]["gravity"], float)
            if np.isfinite(candidate).all() and np.linalg.norm(candidate) > 1e-9:
                reference = candidate
        if reference is None:
            continue
        pts = []
        for i, r in enumerate(rows):
            if not isinstance(r.get("gravity"), list):
                continue
            g = np.asarray(r["gravity"], float)
            norm = np.linalg.norm(g)
            if not np.isfinite(g).all() or norm < 1e-9:
                continue
            angle = math.degrees(math.acos(np.clip(
                float(np.dot(g, reference)) / (norm * np.linalg.norm(reference)), -1, 1)))
            pts.append((i if ctx.order_mode else r["relative_time"], angle))
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, '.-', ms=2, lw=.8, label=run.label)
            plotted += 1
    if plotted:
        ax.legend(fontsize=FONT["legend"], loc="best")
    else:
        _empty_note(ax)
    _panel(ctx, 3, "Gyroscope bias (rad/s)", ["bias_g"], pairs, labels=list("XYZ"), grid=grid)
    _panel(ctx, 4, "Accelerometer bias (m/s^2)", ["bias_a"], pairs, labels=list("XYZ"), grid=grid)
    ctx.foot = ("Bias entries may be fixed initialization priors before they are "
                "estimated; check the initialization pages for the estimation source. "
                "Blank panels mean the log recorded null state values (estimator "
                "state non-finite or not serialized); timing pages stay populated.")


def draw_lio_registration(ctx):
    grid = (3, 2)
    pairs = ctx.pairs(ctx.lio_frames)
    _panel(ctx, 1, "Points (downsampled) / effective", ["points", "effective"], pairs, grid=grid)
    _panel(ctx, 2, "Effective ratio (effective / points; blank when denominator is zero)",
           ["effective_ratio"], pairs, grid=grid)
    _panel(ctx, 3, "Mean point-to-plane residual (m) / lidar noise", ["residual", "cov_lidar"],
           pairs, grid=grid)
    _panel(ctx, 4, "Position correction (posterior - prediction, m)", ["position_correction"],
           pairs, motion=True, labels=list("XYZ"), grid=grid)
    _panel(ctx, 5, "Rotation correction (deg)", ["rotation_correction_deg"], pairs,
           motion=True, grid=grid)
    _panel(ctx, 6, "Iterations / searches", ["iterations", "searches"], pairs, grid=grid)
    ctx.foot = ("Residual is the last linearization point-to-plane mean absolute residual; "
                "it is not a re-registered posterior error or true drift. effective=0 "
                "rows mean no map match (see overview).")


def draw_lio_covariance(ctx):
    grid = (3, 3)
    units = ("rad^2", "m^2", "rad^2", "m^2", "(m/s)^2", "(rad/s)^2", "(m/s^2)^2", "(m/s^2)^2")
    pairs = ctx.pairs(ctx.lio_frames)
    for i, (group, unit) in enumerate(zip(COV_GROUPS, units)):
        _panel(ctx, i + 1, f"Covariance {group}, indices {3*i}:{3*i+3} ({unit})",
               [f"cov_{group}"], pairs, symlog=True, labels=list("XYZ"), grid=grid)
    ctx.foot = ("Covariance: 24-dim filter diagonal groups, mixed units, never summed. "
                "Post-init covariance is the converged ESKF state, not the manual "
                "initialization values.")


def draw_lio_timing(ctx):
    grid = (2, 2)
    pairs = ctx.pairs(ctx.lio_frames)
    _panel(ctx, 1, "Frame processing time (ms; excludes logging)",
           ["frame_ms"], pairs, grid=grid)
    _panel(ctx, 2, "Frame phase timing (ms; NOT stacked/summed)",
           ["imu_ms", "down_ms", "icp_ms", "map_ms"], pairs, grid=grid)
    _panel(ctx, 3, "ICP sub-items (ms): match includes search", ["match_ms", "solve_ms"],
           pairs, grid=grid)
    _panel(ctx, 4, "Match / solve / map time (ms)", ["match_ms", "solve_ms", "map_ms"],
           pairs, grid=grid)
    ctx.foot = ("match is within ICP. frame excludes later spectrum/logging. Never add "
                "phase timings together; they overlap the enclosing scopes.")


_INIT_DRAWERS = {
    "lgo_quality": draw_init_lgo_quality,
    "batch_convergence": draw_init_batch_convergence,
    "reprocess_quality": draw_init_reprocess_quality,
    "imu_excitation": draw_init_imu_excitation,
    "stage_analysis": draw_init_stage_analysis,
}

_LIO_DRAWERS = {
    "motion_rtk": draw_lio_motion_rtk,
    "registration": draw_lio_registration,
    "gravity_biases": draw_lio_gravity_biases,
    "covariance": draw_lio_covariance,
    "timing": draw_lio_timing,
}


def plot_page(fig, section, page, runs, order_mode=False, pass_filter=None):
    """Dispatch one dashboard page; section in SECTION_NAMES, page index within it."""
    from mpl_toolkits.mplot3d import Axes3D  # Register projection on older Matplotlib.
    fig.clear()
    ctx = PageContext(fig, runs, order_mode, pass_filter)
    if section not in SECTION_PAGES:
        section = SECTION_NAMES[0]
    pages = SECTION_PAGES[section]
    name = pages[page] if isinstance(page, int) and 0 <= page < len(pages) else pages[0]
    if section == "initialization":
        _INIT_DRAWERS[name](ctx)
    elif section == "lio":
        _LIO_DRAWERS[name](ctx)
    else:
        draw_overview(ctx)
    fig.suptitle(f"D-LI-Init | {section} > {name} | proxies only, no ground truth", y=.895)
    fig.text(.5, .065, ctx.foot or
             ("All rows retained for diagnostics; motion only selected valid posteriors. "
              "Separate run origins; no pose alignment. Broken lines mark resets, gaps "
              "and invalid posteriors; no interpolation."), ha="center", fontsize=FONT["text"])
    fig.subplots_adjust(top=.84, bottom=.15, hspace=.75, wspace=.3)
    return ctx.axes


def add_plot_fields(run):
    for row in run.frames:
        for i, axis in enumerate("xyz"):
            row[f"position_{axis}"] = row["position"][i]
        for i, group in enumerate(COV_GROUPS):
            row[f"cov_{group}"] = row["covariance_diagonal"][3*i:3*i+3]
        row["gravity_norm"] = float(np.linalg.norm(vector(row, "gravity")))
        # Schema2 rows carry null (not zero) for missing point counts.
        row["effective_ratio"] = (row["effective"] / row["points"]
                                  if row["points"] is not None and row["points"] > 0
                                  and row["effective"] is not None else math.nan)


def configure_backend(no_show):
    import matplotlib
    # Disable both classic/toolmanager toolbars and the save keyboard shortcut.
    matplotlib.rcParams["toolbar"] = "None"
    matplotlib.rcParams["keymap.save"] = []
    # Panel titles/labels use explicit FONT sizes; ticks and default text follow here.
    matplotlib.rcParams["xtick.labelsize"] = FONT["tick"]
    matplotlib.rcParams["ytick.labelsize"] = FONT["tick"]
    matplotlib.rcParams["axes.labelsize"] = FONT["label"]
    matplotlib.rcParams["axes.titlesize"] = FONT["title"]
    matplotlib.rcParams["legend.fontsize"] = FONT["legend"]
    headless = no_show or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if headless:
        matplotlib.use("Agg", force=True)
    else:
        from matplotlib import rcsetup
        headless = matplotlib.get_backend().lower() not in {
            name.lower() for name in rcsetup.interactive_bk}
    return headless


class Viewer:
    """One persistent window; section/page buttons, controls and info survive redraws."""
    def __init__(self, runs, plt):
        import matplotlib
        self.runs, self.plt = runs, plt
        with matplotlib.rc_context({"toolbar": "None", "keymap.save": []}):
            self.fig = plt.figure(figsize=(15, 11))
        manager = self.fig.canvas.manager
        key_handler = getattr(manager, "key_press_handler_id", None)
        if key_handler is not None:
            self.fig.canvas.mpl_disconnect(key_handler)
        self.section = SECTION_NAMES[0]
        self.page = 0
        self.order_mode = False
        self.pass_filter = None
        self.section_buttons, self.page_buttons, self.controls = [], [], []
        self.info = None
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.show(SECTION_NAMES[0], 0)

    def pass_options(self):
        options = {r["pass_id"] for run in self.runs for r in run.frames
                   if r["stage"] in INIT_STAGES and isinstance(r.get("pass_id"), int)}
        options |= {b["pass_id"] for run in self.runs for b in run.batches
                    if isinstance(b.get("pass_id"), int)}
        return sorted(options)

    def show(self, section=None, page=None):
        section = section if section in SECTION_PAGES else self.section
        pages = SECTION_PAGES[section]
        page = page if page is not None and 0 <= page < len(pages) else 0
        self.section, self.page = section, page
        plot_page(self.fig, section, page, self.runs, self.order_mode, self.pass_filter)
        self._draw_buttons()
        self.info = self.fig.text(.02, .012, "Click a time-series point for "
                                   "frame/time/status details; cursor is synced across "
                                   "panels.", fontsize=FONT["text"])
        self.fig.canvas.draw_idle()

    def _draw_buttons(self):
        from matplotlib.widgets import Button
        self.section_buttons, self.page_buttons, self.controls = [], [], []
        for index, name in enumerate(SECTION_NAMES):
            ax = self.fig.add_axes([.035 + index*.15, .958, .14, .032])
            button = Button(ax, name)
            button.label.set_fontsize(FONT["button"])
            button.on_clicked(lambda event, s=name: self.show(s, 0))
            self.section_buttons.append(button)
        pages = SECTION_PAGES[self.section]
        width = .93 / len(pages)
        for index, name in enumerate(pages):
            ax = self.fig.add_axes([.035 + index*width, .922, width*.94, .032])
            button = Button(ax, name)
            button.label.set_fontsize(FONT["button"])
            button.on_clicked(lambda event, p=index: self.show(self.section, p))
            self.page_buttons.append(button)
        ax = self.fig.add_axes([.50, .958, .21, .032])
        order = Button(ax, "X axis: processing order" if self.order_mode
                       else "X axis: data time")
        order.label.set_fontsize(FONT["button"])
        order.on_clicked(self.toggle_order)
        self.controls.append(order)
        options = self.pass_options()
        if options:
            label = (f"Pass filter: {self.pass_filter}" if self.pass_filter is not None
                     else "Pass filter: all passes")
            ax = self.fig.add_axes([.73, .958, .235, .032])
            pass_button = Button(ax, label)
            pass_button.label.set_fontsize(FONT["button"])
            pass_button.on_clicked(self.toggle_pass)
            self.controls.append(pass_button)

    def toggle_order(self, event):
        self.order_mode = not self.order_mode
        self.show()

    def toggle_pass(self, event):
        options = self.pass_options()
        if not options:
            return
        if self.pass_filter is None:
            self.pass_filter = options[0]
        else:
            pos = options.index(self.pass_filter) if self.pass_filter in options else -1
            self.pass_filter = None if pos >= len(options) - 1 else options[pos + 1]
        self.show()

    def on_motion(self, event):
        axes = [a for a in self.fig.axes if getattr(a, "_drift_time", False)]
        if event.inaxes in axes and event.xdata is not None:
            for ax in axes:
                line = getattr(ax, "_drift_cursor", None)
                if line is None:
                    ax._drift_cursor = ax.axvline(event.xdata, color="k", lw=.5, alpha=.4)
                else:
                    line.set_xdata([event.xdata, event.xdata])
                    line.set_visible(True)
        else:
            for ax in axes:
                line = getattr(ax, "_drift_cursor", None)
                if line is not None:
                    line.set_visible(False)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if not getattr(event.inaxes, "_drift_time", False) or event.xdata is None:
            return
        details = []
        for run in self.runs:
            rows = [r for r in run.frames if np.isfinite(r["relative_time"])]
            if not rows:
                continue
            row = min(rows, key=lambda r: abs(r["relative_time"] - event.xdata))
            details.append(f"{run.label}: frame={row['frame']} time={row['time']:.6f} "
                           f"rel={row['relative_time']:.3f}s {row['stage']}/{row['status']} "
                           f"reason={row['reason']} effective={row['effective']} "
                           f"applied/committed={row['applied']}/{row['committed']} "
                           f"valid={row['valid_motion']}")
        self.info.set_text("\n".join(details))
        self.fig.canvas.draw_idle()


def main(argv=None):
    parser = argparse.ArgumentParser(description="D-LI-Init schema1/2 glog 诊断（无真值代理量；schema2 含批解/队列/RTK）")
    parser.add_argument("--log", default=CONFIG["log"], help="单次运行的日志文件或递归 INFO 目录")
    parser.add_argument("--opt", help="可选独立对比运行，按各自 init_done 相对时间")
    parser.add_argument("--output", default=CONFIG["output"], help="显式导出时的 CSV/JSON 目录（不保存图片）")
    parser.add_argument("--lio-window", default=CONFIG["lio_window"], type=float,
                        help="初始化后 LIO 观察秒数，有限正值，两端均包含（默认 10）")
    show = parser.add_mutually_exclusive_group()
    show.add_argument("--show", dest="show", action="store_true")
    show.add_argument("--no-show", dest="show", action="store_false", help="只显示摘要，不打开窗口")
    export = parser.add_mutually_exclusive_group()
    export.add_argument("--export-data", dest="export_data", action="store_true", help="明确启用 JSON/CSV 导出")
    export.add_argument("--no-export-data", dest="export_data", action="store_false", help="禁止数据导出")
    parser.set_defaults(show=CONFIG["show"], export_data=CONFIG["export_data"])
    parser.add_argument("--include-direct-lio", action="store_true", help="显式纳入 direct_lio 运动量")
    args = parser.parse_args(argv)
    if not args.log:
        parser.error("请在 CONFIG 中填写日志路径，或用 --log 指定")
    if not Path(args.log).exists():
        # CONFIG 中相对路径默认相对本脚本目录，便于在任意工作目录裸跑。
        fallback = Path(__file__).resolve().parent / args.log
        if fallback.exists():
            args.log = str(fallback)
    try:
        args.lio_window = finite_positive_window(args.lio_window)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        runs = [load_run(args.log, "log", args.include_direct_lio, args.lio_window)]
        if args.opt:
            runs.append(load_run(args.opt, "opt", args.include_direct_lio, args.lio_window))
            if any(r.init_time is None for r in runs):
                raise ValueError("对比要求两侧都有有效 init_done；不能伪造初始化相对时间")
        for run in runs:
            add_plot_fields(run)
            summary = run_summary(run)
            window = summary["lio_window"]
            print(f"{run.label}: 全部诊断 {summary['frames']} 帧，正式选择 "
                  f"{summary['selected_frames']} 帧，有效后验 {summary['valid_posteriors']} 帧")
            print(f"  LIO 请求 [0, {run.lio_window:g}] s（含端点），原点 "
                  f"{run.origin_kind}={run.origin}；窗外 {window['outside_window_frames']} 帧仅保留诊断")
            for label, key in (("选择", "selected_coverage"), ("有效", "valid_coverage")):
                coverage = window[key]
                print(f"  {label}实际覆盖: {coverage['first_s']} 至 {coverage['last_s']} s，"
                      f"跨度 {coverage['span_s']:g} s（不是连续覆盖保证）")
            if run.schema == 2:
                metrics = summary["initialization"]
                final = metrics["final_pass"]
                print(f"  schema2 运行 {metrics['run_id']}: 批解完成 {metrics['pass_count']} 轮，"
                      f"其中有限解 {metrics['batch_finite_pass_count']} 轮"
                      + (f"（pass {metrics['first_nonfinite_pass']} 起非有限）"
                         if metrics["first_nonfinite_pass"] is not None else "；"
                         " solved 阶段观察 "
                         f"{metrics['batch_solved_observations']} 次"
                         f"（失败 {metrics['batch_solve_failures']}）"))
                print(f"  LIO 交接质量: post_dynamic {metrics['post_dynamic_frames']} 帧，"
                      f"effective=0 共 {metrics['post_effective_zero_frames']} 帧，"
                      f"状态向量 null 共 {metrics['post_state_null_frames']} 帧；"
                      f"init_done reason={metrics['init_done_reason']}")
                if final:
                    print(f"  末轮批解: initial_speed={final['initial_speed']} m/s, "
                          f"delta_v={final['delta_v']}"
                          f"（阈值 {final['velocity_threshold']}，{final['delta_v_reason']}），"
                          f"bias_a_source={final['bias_a_source']}")
                finite_pass = metrics["last_finite_pass"]
                if finite_pass and (final is None or finite_pass["pass_id"] != final["pass_id"]):
                    print(f"  末个有限解轮 pass={finite_pass['pass_id']}: "
                          f"initial_speed={finite_pass['initial_speed']} m/s, "
                          f"delta_v={finite_pass['delta_v']}"
                          f"（阈值 {finite_pass['velocity_threshold']}）")
        print("全部初始化/原始诊断行保留；运动量仅使用窗口内相邻有效后验；无真值，不代表绝对漂移。")
        if args.export_data:
            output = Path(args.output).expanduser()
            output.mkdir(parents=True, exist_ok=True)
            export_data(runs, output)
            print(f"已导出 {output}: summary.json、frames.csv；不保存图片")
        else:
            print("未启用数据导出，不创建输出目录、不写 JSON/CSV 或图片。")
        headless = configure_backend(not args.show)
        if not headless:
            try:
                import matplotlib.pyplot as plt
                viewer = Viewer(runs, plt)
                print("打开统一仪表盘：Overview / Initialization / LIO 三级导航；顶部按钮切页与筛选轮次、点击曲线查看详情；原生工具栏与保存快捷键禁用。", flush=True)
                plt.show()
                del viewer
            except Exception as exc:
                print(f"交互窗口未成功打开或显示失败，仅提供摘要：{exc}", file=sys.stderr)
        else:
            print("窗口未打开：已禁用显示、无桌面环境或非交互后端；仅提供摘要（显式导出单独执行）。")
        return 0
    except (ValueError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())



