#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D-LI-Init 动态初始化阶段「每帧点云处理耗时」分析器

设计依据: FRAME_TIME_ANALYSIS_PLAN.md (工作空间根目录)
数据源 (由 laserMapping.cpp 写出, 目录: src/D-LI-Init/Log/):
  Li_D_init_frame_time.txt  (必需, 26 列, 表头以 # 开头)
      0  t                缓存帧的 lidar_end_time; 跨轮逐字重复, 不是唯一键
      1  t_wall_ms        本帧处理完成时刻相对节点启动的单调墙钟(ms); 相邻两行之差即帧间间隔
      2  frame_id         全局帧序号(含被跳过的帧), 唯一键
      3  pass_id          动态初始化第几轮优化
      4  measures_num     本轮内第几帧 (0..data_accum_length-1)
      5  row_type         0=完整帧; 1=每轮首帧(建树后跳过); 2=无去畸变点云跳过帧
      6  imu_en           0=Point3D 路径(pass 0), 1=PCL 路径(pass>=1)
      7  frame_total_ms   本帧净耗时 (>= 六段之和; 差额见 uncovered_ms)
      8  imu_ms           前向传播+去畸变
      9  downfilter_ms    地图分块 + 下采样
      10 icp_ms           ESKF 迭代(父指标)
      11 knn_ms           近邻搜索累计(icp 的子集)
      12 solve_ms         状态求解累计(icp 的子集)
      13 publish_odom_ms  icp 结束 -> 建图起点 (含 pass0 的 point3DtoPCL + 里程计发布)
      14 map_inc_ms       增量建图
      15 publish_tail_ms  建图结束 -> 帧结束 (effect/path 发布)
      16 iter_count       本帧 ESKF 迭代次数
      17 knn_count        本帧近邻搜索次数
      18 rematch_num      重匹配次数
      19 effect_feat_num  有效特征点数
      20 feats_down_size  下采样后点数
      21 feats_full_size  去畸变后点数(未下采样)
      22 kdtree_size_st   本帧建图前地图点数
      23 kdtree_size_end  本帧建图后地图点数
      24 res_mean_last    上轮平均残差 —— **实测恒为 0, 无信息量, 见下**
      25 build_ms         仅 row_type=1 有效: 建树 + 判停前耗时
  非 0 行中本行不适用的列写 -1 (哨兵); 真实值恒 >= 0。

  关于 res_mean_last: laserMapping.cpp 中 total_residual 在 1681 行清零后**从未被累加**,
  1775 行的 res_mean_last = total_residual / effect_feat_num 结构上恒得 0 (实测 87 帧
  唯一值 {0.0})。属上游既有半成品埋点, 非 nan/inf。本分析器仍解析该列以保持数据契约
  完整, 但不拿它做任何展示或结论。
  计时口径(ms): icp_ms 为父指标, 内含 knn_ms 与 solve_ms, 三者禁止相加;
  净 ICP = icp_ms - knn_ms - solve_ms。
  六段首尾相接但未覆盖全部代码, 差额为 uncovered_ms（恒 >= 0, 实测均值 0.017 ms）。
  Li_D_init_iter_time.txt (可选) 5 列: pass_id solve_ms pass_total_ms delta_g converged
      (首轮 delta_g=-1 表示预处理+首次求解, 无收敛判定)
  Li_D_init_lio_time.txt (可选, 页面6 数据源) 8/9/12 列:
      0 time             帧数据时刻 lidar_end_time(bag 时间, 不是完成时刻!)
      1 frame_total_ms   本帧净耗时 (t_imu_process_begin -> frame_end)
      2 imu_ms  3 downfilter_ms  4 icp_ms(父指标, 内含 knn+solve)  5 knn_ms  6 solve_ms
      7 map_increment_ms
      8 t_wall_ms        本帧处理完成时刻相对节点启动的单调墙钟(ms);
                         相邻两行之差 = 相邻两次状态更新完成的墙钟间隔
      9 effect_feat_num  本帧参与 ESKF 更新的有效特征点数
      10 feats_down_size 降采样后点数 (ICP/KNN 的直接输入)
      11 feats_full_size 降采样前原始点云数量 (imu_en 路径=去畸变后 feats_undistort)
      仅记初始化完成(dynamic_init_fished)后的帧。旧 8 列日志无墙钟列, 完成间隔
      将退化为数据时间差(恒等于输入周期, 看不出积压), 需重新采集才有真实完成间隔;
      8/9 列旧日志无点数列, 页面6 中图退化为提示。

页面 (单窗口 RadioButtons 按钮切换):
  1 每帧耗时分解   | 2 帧间间隔     | 3 ICP 内部构成
  4 归一化指标     | 5 轮次汇总     | 6 LIO后窗口(初始化完成后: 间隔/点数/耗时构成)

注意: 旧 time_log.txt 的 "odometry time" 列把 icp 与 knn/solve 直接相加, 系统性高估,
      本分析器不读取该文件。
"""
import argparse
import os
import sys

import numpy as np

# ---------- 后端选择: 无显示环境 / --no-show 时用 Agg, 否则 TkAgg ----------
import matplotlib
_BACKEND = 'Agg' if ('--no-show' in sys.argv or not os.environ.get('DISPLAY')) else 'TkAgg'
matplotlib.use(_BACKEND)
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

# ---------- 中文字体与统一可读性主题 (沿用 analyze_dashboard.py 的约定) ----------
TITLE_SIZE = 20
AXIS_TITLE_SIZE = 17
TICK_SIZE = 15
LEGEND_SIZE = 14
ANNOTATION_SIZE = 14
EMPTY_SIZE = 22

try:
    from matplotlib.font_manager import findfont, FontProperties
    for _f in ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans CJK',
               'SimHei', 'AR PL UMing CN', 'Source Han Sans CN']:
        try:
            _p = findfont(FontProperties(family=_f), fallback_to_default=False)
            if _p and 'DejaVu' not in _p:
                plt.rcParams['font.sans-serif'] = [_f]
                break
        except Exception:
            continue
except Exception:
    pass

plt.rcParams.update({
    'axes.unicode_minus': False,
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': AXIS_TITLE_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': LEGEND_SIZE,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'Log')

# 右边界留 0.07 给第 3 页的双 Y 轴标签, 否则会被图幅裁掉
PLOT_RECT = [0.19, 0.09, 0.74, 0.84]
MAIN_LOG = 'Li_D_init_frame_time.txt'
ITER_LOG = 'Li_D_init_iter_time.txt'
LAG_LOG = 'Li_D_init_lag.txt'
LIO_LOG = 'Li_D_init_lio_time.txt'
# 页面6: 进入 LIO 后的观察窗口长度(秒), 可用 --lio-window 覆盖
LIO_WINDOW_SECS = 10.0
# 实时预算参考线: 10 Hz 雷达 CustomMsg 经 cut_frame_num=3 分帧 -> 算法帧周期 ≈ 33.3 ms
INPUT_PERIOD_MS = 100

# ---------- 列定义 ----------
COLUMNS = ['t', 't_wall_ms', 'frame_id', 'pass_id', 'measures_num', 'row_type', 'imu_en',
           'frame_total_ms', 'imu_ms', 'downfilter_ms', 'icp_ms', 'knn_ms', 'solve_ms',
           'publish_odom_ms', 'map_inc_ms', 'publish_tail_ms',
           'iter_count', 'knn_count', 'rematch_num', 'effect_feat_num',
           'feats_down_size', 'feats_full_size', 'kdtree_size_st', 'kdtree_size_end',
           'res_mean_last', 'build_ms']
NCOL = len(COLUMNS)
IDX = {name: i for i, name in enumerate(COLUMNS)}
INT_COLS = ('frame_id', 'pass_id', 'measures_num', 'row_type', 'imu_en',
            'iter_count', 'knn_count', 'rematch_num', 'effect_feat_num',
            'feats_down_size', 'feats_full_size', 'kdtree_size_st', 'kdtree_size_end')
# 分项六段 + 帧总耗时 (行类型 0 时首尾相接, 其和恒等于 frame_total_ms)
SEGMENTS = ['imu_ms', 'downfilter_ms', 'icp_ms',
            'publish_odom_ms', 'map_inc_ms', 'publish_tail_ms']
SEG_COLORS = {'imu_ms': '#4C72B0', 'downfilter_ms': '#55A868', 'icp_ms': '#C44E52',
              'publish_odom_ms': '#8172B2', 'map_inc_ms': '#CCB974',
              'publish_tail_ms': '#64B5CD', 'build_ms': '#8C8C8C'}
SEG_LABELS = {'imu_ms': '前向传播+去畸变', 'downfilter_ms': '地图分块+下采样',
              'icp_ms': 'ESKF 迭代', 'publish_odom_ms': '里程计发布(+pass0转换)',
              'map_inc_ms': '增量建图', 'publish_tail_ms': 'effect/path 发布',
              'build_ms': '建树(每轮首帧)'}
ROW_TYPE_LABEL = {0: '完整帧', 1: '每轮首帧(建树)', 2: '无点云跳过帧'}


# ============================== 通用工具 ==============================
def show_empty(ax, message):
    """在页面中央显示一致的空数据提示。"""
    ax.text(0.5, 0.5, message, ha='center', va='center',
            transform=ax.transAxes, color='firebrick', fontsize=EMPTY_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])


def style_axes(ax, legend=False, **legend_kwargs):
    """统一坐标轴文字样式。"""
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.title.set_fontsize(TITLE_SIZE)
    ax.xaxis.label.set_size(AXIS_TITLE_SIZE)
    ax.yaxis.label.set_size(AXIS_TITLE_SIZE)
    if legend:
        ax.legend(fontsize=LEGEND_SIZE, **legend_kwargs)


def safe_loadtxt(path, min_cols):
    """安全读取数值日志；返回 (二维数组或 None, 中文错误说明)。"""
    if not os.path.exists(path):
        return None, '未找到日志：%s' % os.path.basename(path)
    if os.path.getsize(path) == 0:
        return None, '日志为空：%s' % os.path.basename(path)
    try:
        data = np.genfromtxt(path, comments='#', invalid_raise=False, ndmin=2)
    except (OSError, ValueError) as exc:
        return None, '日志读取失败：%s' % exc
    if data.size == 0 or data.shape[0] == 0:
        return None, '日志无有效数据：%s' % os.path.basename(path)
    data = data[~np.all(~np.isfinite(data), axis=1)]
    if data.shape[0] == 0 or data.shape[1] < min_cols:
        cols = data.shape[1] if data.ndim == 2 else 0
        return None, '日志字段不足：%s（需要至少%d列，实际%d列）' % (
            os.path.basename(path), min_cols, cols)
    return data, None


def add_bar_labels(ax, bars, fmt='%.1f'):
    """为柱状图增加数值标签。"""
    for bar in bars:
        value = bar.get_height()
        if not np.isfinite(value) or value == 0:
            continue
        ax.annotate(fmt % value, (bar.get_x() + bar.get_width() / 2.0, value),
                    xytext=(0, 3), textcoords='offset points', ha='center',
                    va='bottom', fontsize=ANNOTATION_SIZE)


def sub_rect(row, col, nrows=1, ncols=1, gap_x=0.07, gap_y=0.13, rect=None):
    """在 rect（默认 PLOT_RECT）内按网格计算第 (row, col) 个子图的 axes 位置。"""
    x0, y0, w, h = PLOT_RECT if rect is None else rect
    cw = (w - gap_x * (ncols - 1)) / ncols
    ch = (h - gap_y * (nrows - 1)) / nrows
    left = x0 + col * (cw + gap_x)
    # 行号自上而下
    bottom = y0 + (nrows - 1 - row) * (ch + gap_y)
    return [left, bottom, cw, ch]


def new_axes(rect, label, **kwargs):
    """创建 axes。

    必须带唯一 label：matplotlib 在「参数与已有 axes 相同」时会**复用同一个 axes 实例**，
    多个页面若都用 plt.axes(PLOT_RECT) 会画到同一坐标系上互相覆盖（导出图完全相同）。
    label 使每个页面各自独立。
    """
    return plt.axes(rect, label=label, **kwargs)


def stat_line(values):
    """返回 (n, mean, median, p95, max) 的中文摘要；无有效值时返回 '无有效数据'。"""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size == 0:
        return '无有效数据'
    return 'n=%d 均值=%.2f 中位=%.2f P95=%.2f 最大=%.2f' % (
        v.size, v.mean(), np.median(v), np.percentile(v, 95), v.max())


# ============================== 数据加载 ==============================
def load_frame_time(path):
    """读取每帧耗时日志；返回 (dict 或 None, 中文错误说明)。"""
    data, error = safe_loadtxt(path, NCOL)
    if data is None:
        return None, error
    d = {}
    for name in COLUMNS:
        col = data[:, IDX[name]].astype(float)
        if name in INT_COLS:
            col = np.rint(col)
        d[name] = col
    # -1 哨兵 -> NaN (真实值恒 >= 0); 只对 frame_total_ms 起的度量列处理, 不动标识列
    for name in COLUMNS[IDX['frame_total_ms']:]:
        d[name] = np.where(d[name] == -1.0, np.nan, d[name])
    d['_n'] = len(d['t'])
    return d, None


def load_iter_time(path):
    """读取轮次耗时日志 (可选)；返回 (dict 或 None, 中文错误说明)。"""
    data, error = safe_loadtxt(path, 5)
    if data is None:
        return None, error
    # 首轮 delta_g 是 C++ 侧哨兵 -1 (laserMapping.cpp:2178「首轮预处理+首次求解, 无收敛判定」),
    # 不是测出来的残差: delta_g = abs(Last_g-current_g) 恒 >= 0, 故 -1 无碰撞。
    # 一律归一成 NaN, 否则终端会打出 "Δg=-1.0000 收敛=否" 这种看着像真实残差的假值,
    # 第 5 页还会画出一根 -1 的假柱。converged 同步置 -1 -> 终端渲染为「未知」。
    delta_g = data[:, 3].copy()
    converged = np.rint(data[:, 4])
    nag = delta_g == -1.0
    delta_g[nag] = np.nan
    converged[nag] = -1
    return {'pass_id': np.rint(data[:, 0]), 'solve_ms': data[:, 1],
            'pass_total_ms': data[:, 2], 'delta_g': delta_g,
            'converged': converged}, None


def load_lio_time(path):
    """读取 LIO 阶段(初始化完成后)耗时日志 (可选)；返回 (dict 或 None, 中文错误说明)。

    12 列为最新版(t_wall_ms 墙钟 + 3 列点数), 9 列为中间版(仅墙钟), 8 列为旧版
    (无墙钟, 完成间隔退化为数据时间差)。knn_ms/solve_ms 是 icp_ms 的子集
    (同 frame_time 的父子口径)。
    """
    data, error = safe_loadtxt(path, 8)
    if data is None:
        return None, error
    d = {'time': data[:, 0].astype(float),
         'frame_total_ms': data[:, 1].astype(float),
         'imu_ms': data[:, 2].astype(float),
         'downfilter_ms': data[:, 3].astype(float),
         'icp_ms': data[:, 4].astype(float),
         'knn_ms': data[:, 5].astype(float),
         'solve_ms': data[:, 6].astype(float),
         'map_inc_ms': data[:, 7].astype(float),
         '_n': data.shape[0]}
    d['_has_wall'] = data.shape[1] >= 9
    if d['_has_wall']:
        d['t_wall_ms'] = data[:, 8].astype(float)
    else:
        d['t_wall_ms'] = np.full(d['_n'], np.nan)
    # 末三列点数(12 列最新版): 有效点 / 降采样后 / 降采样前原始; 旧版日志缺列置 NaN
    d['_has_pts'] = data.shape[1] >= 12
    if d['_has_pts']:
        d['effect_feat_num'] = data[:, 9].astype(float)
        d['feats_down_size'] = data[:, 10].astype(float)
        d['feats_full_size'] = data[:, 11].astype(float)
    else:
        d['effect_feat_num'] = np.full(d['_n'], np.nan)
        d['feats_down_size'] = np.full(d['_n'], np.nan)
        d['feats_full_size'] = np.full(d['_n'], np.nan)
    n = d['_n']
    # 完成间隔: 相邻两次「状态更新完成」的墙钟差 —— 只有新版日志才能算
    interval = np.full(n, np.nan)
    if n > 1 and d['_has_wall']:
        interval[1:] = np.diff(d['t_wall_ms'])
    d['interval_ms'] = interval
    # 数据时间差: 相邻两次状态更新的 lidar_end_time 之差 —— 恒等于输入周期,
    # 若远大于输入周期则说明中间有帧被丢弃/跳过
    dint = np.full(n, np.nan)
    if n > 1:
        dint[1:] = np.diff(d['time']) * 1000.0
    d['data_interval_ms'] = dint
    # 完成空档 = 完成间隔 - 帧内净耗时 (等待/调度/发布等帧外开销)
    d['gap_ms'] = d['interval_ms'] - d['frame_total_ms']
    d['icp_net_ms'] = d['icp_ms'] - d['knn_ms'] - d['solve_ms']
    # LIO 日志只埋了 imu/down/icp/map_inc 四段, 其余(发布等)归入未埋点残差
    d['uncovered_ms'] = d['frame_total_ms'] - (
        d['imu_ms'] + d['downfilter_ms'] + d['icp_ms'] + d['map_inc_ms'])
    return d, None


def lio_window(d, secs=LIO_WINDOW_SECS):
    """截取「进入 LIO 后 secs 秒」的窗口；返回 (子集 dict, 元信息 dict)。

    窗口起点 = 首个 LIO 行(初始化收敛后的第一次状态更新):
      - 新版日志按墙钟截取(进入 LIO 后真实 10 s 内完成的更新);
      - 旧版日志无墙钟, 按数据时间截取(窗口内帧数会偏多, 因积压时数据时间走得慢)。
    """
    meta = {'has_wall': d['_has_wall'], 'secs': secs, 'total_rows': d['_n']}
    if d['_has_wall']:
        anchor = d['t_wall_ms'][0]
        sel = (d['t_wall_ms'] - anchor) / 1000.0 <= secs
        meta['mode'] = 'wall'
    else:
        anchor = d['time'][0]
        sel = (d['time'] - anchor) <= secs
        meta['mode'] = 'data'
    meta['anchor'] = float(anchor)
    meta['rows'] = int(sel.sum())
    win = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    win['_n'] = meta['rows']
    return win, meta


def add_derived(d):
    """计算派生指标 (全部在 Python 侧算, 日志不存冗余列)。"""
    n = d['_n']
    # 帧间间隔: 相邻两次「该帧处理完成」的墙钟差 —— 以每次完成 ESKF 迭代的结果为分界线
    interval = np.full(n, np.nan)
    if n > 1:
        interval[1:] = np.diff(d['t_wall_ms'])
    d['interval_ms'] = interval
    # 帧间空档 = 间隔 - 本帧净耗时 (写日志/spinOnce/rate.sleep/轮末批量求解等)
    d['gap_ms'] = interval - d['frame_total_ms']
    # ICP 父子分解: 净 ICP 绝不能再与 knn/solve 相加
    d['icp_net_ms'] = d['icp_ms'] - d['knn_ms'] - d['solve_ms']
    with np.errstate(divide='ignore', invalid='ignore'):
        d['knn_share'] = d['knn_ms'] / d['icp_ms']
        d['ms_per_valid_pt'] = d['icp_ms'] / d['effect_feat_num']
        d['ms_per_down_pt'] = d['knn_ms'] / d['feats_down_size']
        d['valid_ratio'] = d['effect_feat_num'] / d['feats_down_size']
        d['knn_per_iter'] = d['knn_count'] / d['iter_count']
    d['publish_ms'] = d['publish_odom_ms'] + d['publish_tail_ms']
    d['map_added_pts'] = d['kdtree_size_end'] - d['kdtree_size_st']
    # 未埋点残差 = 帧总耗时 - Σ六段。六段覆盖了绝大部分, 但段与段之间仍有少量
    # 未被任何计时区间包住的代码(如下采样结束到 ICP 开始之间), 该残差恒 >= 0。
    # 实测均值 0.017 ms (0.11% 帧耗时), 最大 0.304 ms —— 故不能当"必须为 0"来判错。
    seg_sum = np.zeros(n)
    for name in SEGMENTS:
        seg_sum = seg_sum + np.nan_to_num(d[name], nan=0.0)
    d['uncovered_ms'] = d['frame_total_ms'] - seg_sum
    return d


def check_data(d):
    """自洽校验；返回中文告警列表 (只告警, 不中止, 不伪造)。
    直接传入 load_frame_time 的结果也可用。"""
    if 'uncovered_ms' not in d:
        add_derived(d)
    warns = []
    full = d['row_type'] == 0
    # 校验①: 六段不得重叠/重复累计(残差为负), 也不得有大块未埋点代码(占比过高)。
    #       残差恒 >= 0 是"六段互不重叠"的证据; 小正值属正常, 不是错误。
    res = d['uncovered_ms']
    neg = full & np.isfinite(res) & (res < -0.05)
    if np.any(neg):
        worst = np.min(res[neg])
        warns.append('分项之和超出帧总耗时：%d 帧（最大超出 %.3f ms），'
                     '计时区间存在重叠或重复累计' % (int(neg.sum()), -worst))
    share = np.where(np.isfinite(d['frame_total_ms']) & (d['frame_total_ms'] > 0),
                     res / d['frame_total_ms'], np.nan)
    big = full & np.isfinite(share) & (share > 0.05)
    if np.any(big):
        warns.append('未埋点残差占帧总耗时超过 5%%：%d 帧（最大 %.1f%%），'
                     '存在未被任何分项覆盖的耗时' % (int(big.sum()),
                                                100 * np.nanmax(share[big])))
    over = full & np.isfinite(d['icp_ms']) & \
        (np.nan_to_num(d['knn_ms'], nan=0.0) + np.nan_to_num(d['solve_ms'], nan=0.0)
         > d['icp_ms'] * 1.02)
    if np.any(over):
        warns.append('knn+solve 超过 icp：%d 帧（父子口径异常，'
                     '净 ICP 可能为负）' % int(over.sum()))
    neg_total = full & np.isfinite(d['frame_total_ms']) & (d['frame_total_ms'] < 0)
    if np.any(neg_total):
        warns.append('帧总耗时出现负值：%d 帧' % int(neg_total.sum()))
    return warns


def pass_summary(d, iter_d=None):
    """按 pass 汇总；返回 dict 列表。直接传入 load_frame_time 的结果也可用。"""
    if 'interval_ms' not in d:
        add_derived(d)
    out = []
    for pid in sorted(set(int(p) for p in d['pass_id'])):
        sel = d['pass_id'] == pid
        item = {'pass_id': pid,
                'rows': int(sel.sum()),
                'full': int((sel & (d['row_type'] == 0)).sum()),
                'skip_build': int((sel & (d['row_type'] == 1)).sum()),
                'imu_en': int(np.nanmax(d['imu_en'][sel])) if np.any(sel) else 0,
                'wall_ms': float(np.nansum(d['interval_ms'][sel])),
                'total_ms': float(np.nansum(d['frame_total_ms'][sel])),
                'build_ms': float(np.nansum(d['build_ms'][sel]))}
        for name in SEGMENTS:
            item[name] = float(np.nansum(d[name][sel]))
        # 先落默认值: 采集若在末轮中途结束, 该轮跑不到轮末, iter_time 里没有它的行;
        # 若只在 iter_d 为 None 时兜底, 这种"缺行的轮"会缺键 -> console_summary KeyError
        item['pass_total_ms'] = np.nan
        item['delta_g'] = np.nan
        item['converged'] = -1
        if iter_d is not None:
            hit = np.where(iter_d['pass_id'] == pid)[0]
            if hit.size:
                item['pass_total_ms'] = float(iter_d['pass_total_ms'][hit[-1]])
                item['delta_g'] = float(iter_d['delta_g'][hit[-1]])
                item['converged'] = int(iter_d['converged'][hit[-1]])
        out.append(item)
    return out


def pass_index(d):
    """返回每个 (pass_id) 在全局帧序中的起始下标, 用于画轮次分隔线。"""
    bounds = []
    prev = None
    for i, pid in enumerate(d['pass_id']):
        if prev is None or pid != prev:
            bounds.append((i, int(pid)))
            prev = pid
    return bounds


# ============================== 页面绘制 ==============================
def build_page_1(d, title_suffix=''):
    """页面1: 每帧耗时分解 (堆叠柱, 六段首尾相接 + 每轮首帧的建树段)。"""
    ax = new_axes(PLOT_RECT, 'p1')
    n = d['_n']
    x = np.arange(n)
    stack = list(SEGMENTS) + ['build_ms']
    bottom = np.zeros(n)
    for name in stack:
        vals = np.nan_to_num(d[name], nan=0.0)
        if not np.any(vals > 0):
            continue
        ax.bar(x, vals, bottom=bottom, width=0.8,
               color=SEG_COLORS[name], label=SEG_LABELS[name])
        bottom = bottom + vals
    peak = bottom.max() if bottom.max() > 0 else 1.0
    # 顶部留 35% 余量供图例, 轮次标注贴柱顶(余量区之下), 两者不重叠
    ax.set_ylim(0, peak * 1.35)
    for i, pid in pass_index(d):
        if i > 0:
            ax.axvline(i - 0.5, color='0.35', linestyle='--', linewidth=1.2)
        ax.text(i + 0.4, peak * 1.02, '第%d轮' % pid,
                fontsize=ANNOTATION_SIZE, color='0.25', va='bottom')
    # 标记非完整帧
    marks = np.where(d['row_type'] != 0)[0]
    if marks.size:
        top = np.nan_to_num(d['frame_total_ms'], nan=0.0)[marks]
        ax.plot(marks, top, 'v', color='black', markersize=8,
                label='每轮首帧/跳过帧 (见下表)')
    style_axes(ax, legend=True, ncol=4, loc='upper center')
    ax.set_xlabel('帧序号（初始化回放阶段, 按处理先后）')
    ax.set_ylabel('耗时 (ms)')
    ax.set_title('每帧处理耗时分解%s' % title_suffix)
    return [ax]


def build_page_2(d, title_suffix=''):
    """页面2: 帧间间隔 —— 以每次完成 ESKF 迭代的结果为分界线。"""
    ax = new_axes(PLOT_RECT, 'p2')
    n = d['_n']
    if n < 2:
        show_empty(ax, '有效帧不足，无法计算帧间间隔')
        return [ax]
    x = np.arange(1, n)
    interval = d['interval_ms'][1:]
    gap = d['gap_ms'][1:]
    total = d['frame_total_ms'][1:]
    ax.plot(x, interval, color='#C44E52', linewidth=1.6, label='帧间间隔（相邻两次处理完成之差）')
    ax.fill_between(x, total, interval, where=np.isfinite(gap) & (gap > 0),
                    color='#8172B2', alpha=0.35, label='帧间空档（写日志/spinOnce/轮末求解等）')
    ax.plot(x, total, color='#4C72B0', linewidth=1.6, label='帧内净耗时')
    # 参考线取值依据(已核实, 非假设): bag 中 /livox/lidar 为 CustomMsg, 实测 24000 点/帧 @ 10 Hz;
    # config/avia.yaml 的 cut_frame_num=3, 即 avia 处理器把每个 100 ms 的 CustomMsg 切成 3 个子帧,
    # 故算法实际收到的帧周期是 100/3 ≈ 33.3 ms(约 30 Hz)。这与本日志中 pass0 缓存帧
    # lidar_end_time 的逐差一致(前几帧 100 ms —— cut_frame_init_num=5 期间不分帧; 之后 ≈33 ms)。
    ax.axhline(33.33, color='0.35', linestyle='--', linewidth=1.3,
               label='实时预算 33.3 ms（10 Hz 雷达经 cut_frame_num=3 分帧）')
    for i, pid in pass_index(d):
        if i > 0:
            ax.axvline(i - 0.5, color='0.35', linestyle='--', linewidth=1.2)
    style_axes(ax, legend=True, ncol=2, loc='upper left')
    ax.set_xlabel('帧序号（初始化回放阶段）')
    ax.set_ylabel('时间 (ms)')
    ax.set_title('帧间间隔 vs 帧内净耗时%s' % title_suffix)
    return [ax]


def build_page_3(d, title_suffix=''):
    """页面3: ICP 内部构成 (knn / solve / 净ICP) + 迭代次数。"""
    ax = new_axes(sub_rect(0, 0, 2, 1, gap_y=0.20), 'p3a')
    ax2 = new_axes(sub_rect(1, 0, 2, 1, gap_y=0.20), 'p3b', sharex=ax)
    n = d['_n']
    x = np.arange(n)
    parts = [('knn_ms', '近邻搜索 KNN', '#DD8452'),
             ('solve_ms', '状态求解', '#937860'),
             ('icp_net_ms', '净 ICP 其余部分', '#4C72B0')]
    bottom = np.zeros(n)
    for name, label, color in parts:
        vals = np.nan_to_num(d[name], nan=0.0)
        ax.bar(x, vals, bottom=bottom, width=0.8, color=color, label=label)
        bottom = bottom + vals
    for i, pid in pass_index(d):
        if i > 0:
            ax.axvline(i - 0.5, color='0.35', linestyle='--', linewidth=1.2)
    style_axes(ax, legend=True, ncol=3, loc='upper left')
    ax.set_ylabel('耗时 (ms)')
    ax.set_title('ICP 内部构成%s（knn+solve 是 icp 的子集，禁止与 icp 相加）' % title_suffix)

    share = d['knn_share'] * 100.0
    ax2.plot(x, share, color='#C44E52', linewidth=1.6, label='KNN 占 ICP 比例 (左轴, %)')
    ax2.set_ylim(0, 105)
    ax2.set_xlabel('帧序号（初始化回放阶段）')
    ax2.set_ylabel('KNN 占比 (%)')
    ax2b = ax2.twinx()
    ax2b.step(x, d['iter_count'], where='mid', color='#55A868', linewidth=1.2,
              label='ESKF 迭代次数 (右轴)')
    ax2b.step(x, d['rematch_num'], where='mid', color='#8172B2', linewidth=1.2,
              linestyle=':', label='重匹配次数 (右轴)')
    ax2b.set_ylabel('次数')
    # 迭代次数恒顶满上限时曲线会贴在上边框上, 留出余量才看得出"贴顶"
    ax2b.set_ylim(0, max(np.nanmax(d['iter_count']), np.nanmax(d['rematch_num']), 1) * 1.25)
    style_axes(ax2)
    ax2b.tick_params(axis='both', labelsize=TICK_SIZE)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=LEGEND_SIZE, ncol=3, loc='upper left')
    return [ax, ax2, ax2b]


def build_page_4(d, title_suffix=''):
    """页面4: 归一化指标 (跨 pass / 跨配置可比口径)。"""
    axes = []
    specs = [
        ('ms_per_down_pt', 'KNN 每下采样点耗时 (ms/点)', '#DD8452'),
        ('ms_per_valid_pt', 'ICP 每有效点耗时 (ms/点)', '#4C72B0'),
        ('valid_ratio', '有效点占比 effect_feat/下采样点', '#55A868'),
        # 原计划此处画 res_mean_last, 但实测该列恒为 0 —— laserMapping.cpp 中
        # total_residual 在 1681 行清零后从未被累加, 1775 行的
        # res_mean_last = total_residual / effect_feat_num 恒得 0, 是既有的半成品埋点,
        # 不携带信息。改画每帧地图新增点数(已实测有效)。
        ('map_added_pts', '每帧地图新增点数', '#8172B2'),
    ]
    # 本页上方要放整页标题, 故把 2×2 子图整体压低 (top 由 0.93 降到 0.87),
    # 否则子图标题与整页标题会画在同一高度上互相压字。
    rect = [PLOT_RECT[0], PLOT_RECT[1], PLOT_RECT[2], PLOT_RECT[3] - 0.06]
    x = np.arange(d['_n'])
    for k, (name, label, color) in enumerate(specs):
        ax = new_axes(sub_rect(k // 2, k % 2, 2, 2, rect=rect), 'p4_%d' % k)
        vals = np.asarray(d[name], dtype=float)
        vals = np.where(np.isfinite(vals), vals, np.nan)
        ax.plot(x, vals, color=color, linewidth=1.4, marker='.', markersize=3)
        for i, pid in pass_index(d):
            if i > 0:
                ax.axvline(i - 0.5, color='0.35', linestyle='--', linewidth=1.0)
        ax.set_title(label, fontsize=ANNOTATION_SIZE + 2)
        ax.set_xlabel('帧序号' if k >= 2 else '')
        style_axes(ax)
        ax.grid(alpha=0.25)
        axes.append(ax)
    # 页面标题放在独立的标题轴上, 不能用 fig.suptitle ——
    # figure 级艺术家不随页面隐藏, 会叠在所有其他页面的标题上。
    # 标题用 axes 内的 text 而非 set_title: set_title 画在该 axes 的**上方**,
    # 而本 axes 已顶到 y=0.97, 再往上会被图幅上边缘裁掉。
    tax = new_axes([PLOT_RECT[0], 0.90, PLOT_RECT[2], 0.06], 'p4_title')
    tax.axis('off')
    tax.text(0.5, 0.5, '归一化指标%s（用于跨轮次/跨配置比较）' % title_suffix,
             ha='center', va='center', fontsize=TITLE_SIZE,
             transform=tax.transAxes)
    axes.append(tax)
    return axes


def build_page_5(d, summaries, title_suffix=''):
    """页面5: 轮次汇总 (每 pass 各分项之和 vs 轮次总耗时)。"""
    ax = new_axes(PLOT_RECT, 'p5')
    if not summaries:
        show_empty(ax, '无有效轮次数据')
        return [ax]
    x = np.arange(len(summaries))
    bottom = np.zeros(len(summaries))
    stack = list(SEGMENTS) + ['build_ms']
    for name in stack:
        vals = np.array([s[name] for s in summaries])
        if not np.any(vals > 0):
            continue
        ax.bar(x, vals, bottom=bottom, width=0.55,
               color=SEG_COLORS[name], label=SEG_LABELS[name])
        bottom = bottom + vals
    if np.any(np.isfinite([s['pass_total_ms'] for s in summaries])):
        ax.plot(x, [s['pass_total_ms'] for s in summaries], 'o--', color='black',
                linewidth=1.6, markersize=9, label='轮次总耗时（Li_D_init_iter_time）')
    top = bottom.max() if bottom.max() > 0 else 1.0
    tops = np.maximum(bottom, np.array([
        s['pass_total_ms'] if np.isfinite(s['pass_total_ms']) else 0.0 for s in summaries]))
    # 留出顶部空白: 图例置于其内, Δg 标注紧贴柱顶, 两者不重叠
    ax.set_ylim(0, tops.max() * 1.45)
    for i, s in enumerate(summaries):
        txt = 'Δg=%s' % ('—' if not np.isfinite(s['delta_g']) else '%.3f' % s['delta_g'])
        if s['converged'] == 1:
            txt += '（已收敛）'   # 不用 ✓ 等符号: CJK 字体常缺该字形, 会渲染成方框
        ax.annotate(txt, (i, tops[i]), xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=ANNOTATION_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(['第%d轮\nimu_en=%d\n完整帧×%d' % (s['pass_id'], s['imu_en'], s['full'])
                        for s in summaries])
    style_axes(ax, legend=True, ncol=4, loc='upper center')
    ax.set_ylabel('耗时 (ms)')
    ax.set_title('各轮次耗时构成%s（柱=逐帧分项之和）' % title_suffix)
    return [ax]


def build_page_6(win, meta, secs=LIO_WINDOW_SECS):
    """页面6: 初始化完成进入 LIO 后 secs 秒内, 每次状态更新的间隔 / 点数 / 耗时构成。

    上图: 完成间隔 = 上一次状态更新完成 -> 本次状态更新完成的墙钟差;
          与帧内净耗时 frame_total_ms 的差即帧外空档(等待/调度)。
    中图: 点数漏斗 = 降采样前原始点数 -> 降采样后点数 -> ESKF 有效点数。
    下图: 耗时构成堆叠柱。
    旧版 8 列日志无墙钟列(上图退化为数据时间差), 8/9 列旧日志无点数列(中图提示重采)。
    """
    ax = new_axes(sub_rect(0, 0, 3, 1, gap_y=0.15), 'p6a')
    axb = new_axes(sub_rect(1, 0, 3, 1, gap_y=0.15), 'p6b')
    ax2 = new_axes(sub_rect(2, 0, 3, 1, gap_y=0.15), 'p6c')
    if win is None or win['_n'] == 0:
        show_empty(ax, '无 LIO 阶段日志（Li_D_init_lio_time.txt）')
        show_empty(axb, '初始化未完成或日志缺失')
        show_empty(ax2, '初始化未完成或日志缺失')
        style_axes(ax)
        style_axes(axb)
        style_axes(ax2)
        ax.set_xlabel('进入LIO后时间 (s)')
        axb.set_xlabel('进入LIO后时间 (s)')
        ax2.set_xlabel('进入LIO后时间 (s)')
        ax.set_ylabel('时间 (ms)')
        axb.set_ylabel('点数')
        ax2.set_ylabel('耗时 (ms)')
        return [ax, axb, ax2]
    n = win['_n']
    x = np.arange(n)
    has_wall = win['_has_wall']
    # 统一横轴: 新版用墙钟秒, 旧版退化为数据时间秒
    if has_wall:
        xs = (win['t_wall_ms'] - win['t_wall_ms'][0]) / 1000.0
        xlab = '进入LIO后墙钟时间 (s)'
    else:
        xs = win['time'] - win['time'][0]
        xlab = '进入LIO后数据时间 (s)'

    # ---- 上图: 完成间隔 vs 帧内净耗时 ----
    if n >= 2:
        if has_wall:
            ax.plot(xs[1:], win['interval_ms'][1:], color='#C44E52',
                    linewidth=1.6, label='状态更新完成间隔（上一次完成→本次完成, 墙钟）')
            gap = win['gap_ms'][1:]
            ax.fill_between(xs[1:], win['frame_total_ms'][1:], win['interval_ms'][1:],
                            where=np.isfinite(gap) & (gap > 0),
                            color='#8172B2', alpha=0.35, label='帧外空档（等待/调度等）')
        else:
            ax.plot(xs[1:], win['data_interval_ms'][1:], color='#DD8452',
                    linewidth=1.6, linestyle='--',
                    label='相邻帧数据时间差（旧版日志无墙钟列, 非完成间隔!）')
        ax.plot(xs, win['frame_total_ms'], color='#4C72B0',
                linewidth=1.6, label='帧内净耗时')
    ax.axhline(INPUT_PERIOD_MS, color='0.35', linestyle='--', linewidth=1.3,
               label='实时预算 100 ms')
    style_axes(ax, legend=True, ncol=2, loc='upper left')
    ax.set_xlabel(xlab)
    ax.set_ylabel('时间 (ms)')
    if has_wall and n >= 2:
        mean_iv = np.nanmean(win['interval_ms'][1:])
        ax.set_title('LIO 状态更新完成间隔 vs 帧内净耗时（进入LIO后%.0fs窗口）' % secs)
        # # 吞吐判定: 完成间隔均值 vs 输入周期
        # if mean_iv > INPUT_PERIOD_MS * 1.05:
        #     verdict = '完成间隔均值 %.1f ms > 输入周期 33.3 ms\n→ 处理跟不上输入, 积压持续增长' % mean_iv
        # elif np.nanmean(win['frame_total_ms']) > INPUT_PERIOD_MS:
        #     verdict = '完成间隔均值 %.1f ms ≈ 输入周期, 但帧内净耗时均值 %.1f ms 已超预算\n→ 处于跟上/积压临界' % (
        #         mean_iv, np.nanmean(win['frame_total_ms']))
        # else:
        #     verdict = '完成间隔均值 %.1f ms ≤ 输入周期 33.3 ms\n→ 保持实时' % mean_iv
        # ax.text(0.98, 0.03, verdict, transform=ax.transAxes, ha='right', va='bottom',
        #         fontsize=ANNOTATION_SIZE,
        #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        ax.set_title('LIO 相邻帧间隔 vs 帧内净耗时（进入LIO后%.0fs窗口, 旧版日志无墙钟列）' % secs)

    # ---- 中图: 每次状态更新的点数 (降采样前原始 -> 降采样后 -> ESKF 有效) ----
    if win.get('_has_pts'):
        axb.plot(xs, win['feats_full_size'], color='#4C72B0', linewidth=1.4,
                 label='降采样前原始点数')
        axb.plot(xs, win['feats_down_size'], color='#55A868', linewidth=1.4,
                 label='降采样后点数（ICP/KNN 输入）')
        axb.plot(xs, win['effect_feat_num'], color='#C44E52', linewidth=1.4,
                 label='ESKF 有效点数（参与状态更新）')
        m_full = np.nanmean(win['feats_full_size'])
        m_down = np.nanmean(win['feats_down_size'])
        m_eff = np.nanmean(win['effect_feat_num'])
        keep = m_down / m_full * 100 if m_full > 0 else float('nan')
        effr = m_eff / m_down * 100 if m_down > 0 else float('nan')
        axb.text(0.98, 0.05,
                 '窗口均值: 原始 %.0f → 降采样 %.0f (保留 %.1f%%)\n→ ESKF 有效 %.0f (占降采样 %.1f%%)'
                 % (m_full, m_down, keep, m_eff, effr),
                 transform=axb.transAxes, ha='right', va='bottom',
                 fontsize=ANNOTATION_SIZE,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        axb.set_title('每次状态更新的点数：降采样前原始 / 降采样后 / ESKF 有效')
        style_axes(axb, legend=True, ncol=3, loc='upper right')
    else:
        show_empty(axb, '旧版日志无点数列\n(effect_feat_num 等 3 列, 需重新采集)')
        axb.set_title('每次状态更新的点数（日志缺列）')
        style_axes(axb)
    axb.set_xlabel(xlab)
    axb.set_ylabel('点数')

    # ---- 下图: 每次状态更新的耗时构成 (堆叠到 frame_total) ----
    parts = [('imu_ms', '前向传播+去畸变', '#4C72B0'),
             ('downfilter_ms', '地图分块+下采样', '#55A868'),
             ('knn_ms', 'KNN', '#DD8452'),
             ('solve_ms', '状态求解', '#937860'),
             ('icp_net_ms', 'ICP其余', '#C44E52'),
             ('map_inc_ms', '增量建图', '#CCB974'),
             ('uncovered_ms', '未埋点(发布等)', '#8C8C8C')]
    bottom = np.zeros(n)
    for name, label, color in parts:
        # icp_net 可能因 ikd-Tree 后台重建线程抢核出现微负值, 堆叠时截到 0
        vals = np.clip(np.nan_to_num(win[name], nan=0.0), 0.0, None)
        if not np.any(vals > 0):
            continue
        ax2.bar(x, vals, bottom=bottom, width=0.8, color=color, label=label)
        bottom = bottom + vals
    peak = bottom.max() if bottom.max() > 0 else 1.0
    ax2.set_ylim(0, peak * 1.35)
    style_axes(ax2, legend=True, ncol=4, loc='upper center')
    ax2.set_xlabel(xlab)
    ax2.set_ylabel('耗时 (ms)')
    ax2.set_title('每次状态更新的耗时构成%s（knn+solve 是 icp 的子集, 柱顶≈帧内净耗时）'
                  % ('' if n >= 2 else ''))
    return [ax, axb, ax2]


# ============================== 输出 ==============================
def console_summary(d, summaries, warns, log_dir, iter_error=None):
    """终端分节摘要。"""
    print('[frame_time] 日志目录: %s' % log_dir)
    print('[frame_time] 总行数=%d  row_type: %s' % (
        d['_n'], ', '.join('%d=%d' % (k, int((d['row_type'] == k).sum())) for k in (0, 1, 2))))
    print('[frame_time] 轮次数=%d  每轮帧数: %s' % (
        len(summaries), ', '.join('第%d轮=%d' % (s['pass_id'], s['rows']) for s in summaries)))
    full = d['row_type'] == 0
    for name, label in [('frame_total_ms', '帧内净耗时'), ('interval_ms', '帧间间隔'),
                        ('icp_ms', 'ICP 总'), ('knn_ms', 'KNN'),
                        ('solve_ms', '求解'), ('icp_net_ms', '净ICP其余'),
                        ('map_inc_ms', '建图'), ('publish_ms', '发布合计'),
                        ('uncovered_ms', '未埋点残差')]:
        vals = d[name][full] if name != 'interval_ms' else d[name]
        print('[frame_time]   %s: %s' % (label, stat_line(vals)))
    unc = d['uncovered_ms'][full]
    tot = d['frame_total_ms'][full]
    ok = np.isfinite(unc) & np.isfinite(tot) & (tot > 0)
    if np.any(ok):
        print('[frame_time]   未埋点残差占比: 均值=%.2f%%  最大=%.2f%%'
              % (100 * np.mean(unc[ok] / tot[ok]), 100 * np.max(unc[ok] / tot[ok])))
    knn_share = d['knn_share'][full]
    knn_share = knn_share[np.isfinite(knn_share)]
    if knn_share.size:
        print('[frame_time]   KNN 占 ICP 比例: 中位=%.1f%%  P95=%.1f%%' % (
            np.median(knn_share) * 100, np.percentile(knn_share, 95) * 100))
    print('[frame_time] 每轮汇总:')
    for s in summaries:
        print('[frame_time]   第%d轮: 逐帧合计=%.1f ms  轮次总耗时=%s ms  Δg=%s  收敛=%s'
              % (s['pass_id'], s['total_ms'],
                 '—' if not np.isfinite(s['pass_total_ms']) else '%.1f' % s['pass_total_ms'],
                 '—' if not np.isfinite(s['delta_g']) else '%.4f' % s['delta_g'],
                 {0: '否', 1: '是'}.get(s['converged'], '未知')))
    if iter_error:
        print('[frame_time] 提示: %s（轮次总耗时列将留空）' % iter_error)
    if warns:
        for w in warns:
            print('[frame_time] 告警: %s' % w)
    else:
        print('[frame_time] 告警: 0 项')


def report_init_lag(log_dir):
    """终端打印: 动态初始化完成时刻, 系统处理的数据落后现实多少秒。

    数据来源 Li_D_init_lag.txt —— 节点在 dynamic_init_fished=true 处当场采集一行。
    该量无法由其余日志推导: 帧时间戳是 bag/仿真时间, 而 Li_D_init_frame_time.txt
    只记单调墙钟 t_wall_ms, 两者之间没有换算桥梁, 故必须在完成点当场采样。
    列: t_frame t_sim_now lag_ms pass_id converged
    """
    path = os.path.join(log_dir, LAG_LOG)
    if not os.path.isfile(path):
        print('[frame_time] 系统滞后: 未找到 %s（该埋点为后加, 旧日志无此文件；'
              '需重新编译节点并重跑一次采集）' % LAG_LOG)
        return
    try:
        with open(path, encoding='utf-8') as fh:
            rows = [ln.split() for ln in fh
                    if ln.strip() and not ln.lstrip().startswith('#')]
    except OSError as err:
        print('[frame_time] 系统滞后: 读取 %s 失败: %s' % (LAG_LOG, err))
        return
    if not rows:
        print('[frame_time] 系统滞后: %s 无数据行' % LAG_LOG)
        return

    print('[frame_time] 初始化完成时刻的系统滞后(处理滞后现实):')
    shown = False
    for r in rows:
        try:
            t_frame, t_sim, lag_ms = float(r[0]), float(r[1]), float(r[2])
        except (IndexError, ValueError):
            print('[frame_time]   数据行格式异常, 已跳过: %s' % ' '.join(r))
            continue
        pass_id = r[3] if len(r) > 3 else '—'
        try:
            reason = {1: '重力收敛', 0: '达最大迭代'}.get(int(float(r[4])), '未知')
        except (IndexError, ValueError):
            reason = '未知'
        if not np.isfinite(lag_ms) or lag_ms < 0:
            print('[frame_time]   第%s轮完成（%s）: 无法计算 —— 未启用 use_sim_time, '
                  '帧时间戳(bag 时间)与墙钟不同基准' % (pass_id, reason))
            continue
        print('[frame_time]   第%s轮完成（%s）: 完成帧数据时刻=%.6f s, 当时仿真时刻=%.6f s'
              % (pass_id, reason, t_frame, t_sim))
        print('[frame_time]     系统滞后 = %.1f ms (%.3f s)'
              % (lag_ms, lag_ms / 1000.0))
        shown = True
    if shown:
        print('[frame_time]   注: 此值只反映初始化算完那一刻的滞后; LIO 阶段若单帧耗时'
              '持续超过帧周期, 滞后会继续无上限增长')


def console_lio_summary(win, meta, error, secs):
    """终端分节摘要: LIO 阶段(初始化完成后)状态更新间隔。"""
    if error:
        print('[lio] 提示: %s（页面6 显示空数据; 初始化未收敛或采集未覆盖到 LIO 阶段）' % error)
        return
    print('[lio] LIO 阶段(初始化完成后): 共 %d 次状态更新' % meta['total_rows'])
    if meta['mode'] == 'wall':
        print('[lio] 窗口: 进入LIO后 %.1f s 墙钟内 %d 次状态更新 (有效更新频率 ≈ %.1f Hz)'
              % (secs, meta['rows'],
                 meta['rows'] / secs if secs > 0 else float('nan')))
    else:
        print('[lio] 窗口: 进入LIO后 %.1f s 数据时间内 %d 次状态更新' % (secs, meta['rows']))
        print('[lio] 提示: 旧版8列日志无 t_wall_ms 列, 无法计算真实完成间隔(下图间隔为数据时间差);')
        print('[lio]       重新编译节点并重跑采集后即可获得墙钟完成间隔')
    print('[lio]   帧内净耗时: %s' % stat_line(win['frame_total_ms']))
    print('[lio]   ICP 总: %s' % stat_line(win['icp_ms']))
    print('[lio]   KNN: %s' % stat_line(win['knn_ms']))
    print('[lio]   求解: %s' % stat_line(win['solve_ms']))
    print('[lio]   建图: %s' % stat_line(win['map_inc_ms']))
    print('[lio]   未埋点残差(发布等): %s' % stat_line(win['uncovered_ms']))
    print('[lio]   数据时间间隔: %s' % stat_line(win['data_interval_ms']))
    if win.get('_has_pts'):
        print('[lio]   降采样前原始点数: %s' % stat_line(win['feats_full_size']))
        print('[lio]   降采样后点数: %s' % stat_line(win['feats_down_size']))
        print('[lio]   ESKF 有效点数: %s' % stat_line(win['effect_feat_num']))
        m_full = np.nanmean(win['feats_full_size'])
        m_down = np.nanmean(win['feats_down_size'])
        m_eff = np.nanmean(win['effect_feat_num'])
        if np.isfinite(m_full) and m_full > 0:
            print('[lio]   点数漏斗均值: 原始 %.0f → 降采样 %.0f (保留 %.1f%%) → 有效 %.0f (占降采样 %.1f%%)'
                  % (m_full, m_down, m_down / m_full * 100 if np.isfinite(m_down) else float('nan'),
                     m_eff, m_eff / m_down * 100 if np.isfinite(m_down) and m_down > 0 else float('nan')))
    else:
        print('[lio] 提示: 旧版日志无点数列(effect_feat_num 等 3 列), 页面6 中图显示提示;')
        print('[lio]       重新编译节点并重跑采集后即可获得点数漏斗')
    if meta['mode'] == 'wall':
        print('[lio]   完成间隔: %s' % stat_line(win['interval_ms']))
        print('[lio]   帧外空档(完成间隔-净耗时): %s' % stat_line(win['gap_ms']))
        iv = win['interval_ms'][1:]
        iv = iv[np.isfinite(iv)]
        if iv.size:
            mean_iv = float(iv.mean())
            if mean_iv > INPUT_PERIOD_MS * 1.05:
                print('[lio]   吞吐判定: 完成间隔均值 %.1f ms > 输入周期 33.3 ms'
                      ' → 处理跟不上输入, 积压持续增长' % mean_iv)
            elif np.nanmean(win['frame_total_ms']) > INPUT_PERIOD_MS:
                print('[lio]   吞吐判定: 临界 —— 完成间隔均值 %.1f ms, 但帧内净耗时'
                      '已超 33.3 ms 预算' % mean_iv)
            else:
                print('[lio]   吞吐判定: 保持实时 (完成间隔均值 %.1f ms ≤ 33.3 ms)' % mean_iv)
    # 帧丢弃检测: 数据时间间隔远大于输入周期 => 中间有帧未被处理
    dint = win['data_interval_ms']
    skip = np.isfinite(dint) & (dint > INPUT_PERIOD_MS * 2.0)
    if int(skip.sum()):
        print('[lio] 告警: %d 处数据时间间隔 > 66.7 ms（最大 %.1f ms）——'
              '有帧被丢弃或跳过' % (int(skip.sum()), float(np.nanmax(dint[skip]))))


def write_csv(d, summaries, outdir):
    """导出逐列统计与轮次汇总 CSV。"""
    os.makedirs(outdir, exist_ok=True)
    full = d['row_type'] == 0
    stats_path = os.path.join(outdir, 'frame_time_summary.csv')
    with open(stats_path, 'w') as f:
        f.write('# 列名 n 均值 中位数 P95 最大值\n')
        for name in COLUMNS[IDX['frame_total_ms']:]:
            vals = np.asarray(d[name], dtype=float)
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            if vals.size == 0:
                f.write('%s 0 nan nan nan nan\n' % name)
                continue
            f.write('%s %d %.6f %.6f %.6f %.6f\n' % (
                name, vals.size, vals.mean(), np.median(vals),
                np.percentile(vals, 95), vals.max()))
        for name in ('interval_ms', 'gap_ms', 'icp_net_ms', 'knn_share',
                     'ms_per_valid_pt', 'valid_ratio', 'publish_ms',
                     'uncovered_ms', 'ms_per_down_pt'):
            vals = np.asarray(d[name], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            f.write('%s %d %.6f %.6f %.6f %.6f\n' % (
                name, vals.size, vals.mean(), np.median(vals),
                np.percentile(vals, 95), vals.max()))
    pass_path = os.path.join(outdir, 'pass_summary.csv')
    with open(pass_path, 'w') as f:
        header = ['pass_id', 'rows', 'full_rows', 'imu_en'] + list(SEGMENTS) + \
                 ['build_ms', 'frame_sum_ms', 'pass_total_ms', 'delta_g', 'converged']
        f.write('# ' + ' '.join(header) + '\n')
        for s in summaries:
            f.write('%d %d %d %d ' % (s['pass_id'], s['rows'], s['full'], s['imu_en']))
            f.write(' '.join('%.6f' % s[name] for name in SEGMENTS))
            f.write(' %.6f %.6f %s %s %d\n' % (
                s['build_ms'], s['total_ms'],
                'nan' if not np.isfinite(s['pass_total_ms']) else '%.6f' % s['pass_total_ms'],
                'nan' if not np.isfinite(s['delta_g']) else '%.6f' % s['delta_g'],
                s['converged']))
    return stats_path, pass_path


def write_lio_csv(win, meta, outdir, secs):
    """导出页面6的 LIO 窗口数据: 逐帧明细 + 统计摘要。无窗口数据时返回空列表。"""
    if win is None or win['_n'] == 0:
        return []
    os.makedirs(outdir, exist_ok=True)
    frames_path = os.path.join(outdir, 'lio_window_frames.csv')
    cols = [('time', '数据时刻s'), ('t_wall_ms', '完成墙钟ms'),
            ('interval_ms', '完成间隔ms'), ('data_interval_ms', '数据时间差ms'),
            ('frame_total_ms', '帧内净耗时ms'), ('gap_ms', '帧外空档ms'),
            ('imu_ms', '前向传播去畸变ms'), ('downfilter_ms', '地图分块下采样ms'),
            ('icp_ms', 'ICP总ms'), ('knn_ms', 'KNNms'), ('solve_ms', '求解ms'),
            ('icp_net_ms', 'ICP其余ms'), ('map_inc_ms', '增量建图ms'),
            ('uncovered_ms', '未埋点ms'),
            ('effect_feat_num', 'ESKF有效点数'), ('feats_down_size', '降采样后点数'),
            ('feats_full_size', '降采样前原始点数')]
    with open(frames_path, 'w') as f:
        f.write('# LIO窗口: 进入LIO后%.1fs(%s), %d/%d次状态更新\n'
                % (secs, '按墙钟截取' if meta['mode'] == 'wall' else '旧版日志按数据时间截取',
                   meta['rows'], meta['total_rows']))
        f.write('# ' + ' '.join(name for name, _ in cols) + '\n')
        for i in range(win['_n']):
            f.write(' '.join('%.6f' % win[name][i] for name, _ in cols) + '\n')
    summary_path = os.path.join(outdir, 'lio_window_summary.csv')
    with open(summary_path, 'w') as f:
        f.write('# 列名 n 均值 中位数 P95 最大值\n')
        for name, _ in cols:
            vals = np.asarray(win[name], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            f.write('%s %d %.6f %.6f %.6f %.6f\n' % (
                name, vals.size, vals.mean(), np.median(vals),
                np.percentile(vals, 95), vals.max()))
    return [frames_path, summary_path]


def export_pages(pages, outdir):
    """离线导出各页面 PNG。"""
    os.makedirs(outdir, exist_ok=True)
    written = []
    keys = list(pages.keys())
    for k, key in enumerate(keys):
        axes = pages[key]
        for a in axes:
            a.set_visible(True)
        fig = axes[0].figure
        path = os.path.join(outdir, 'page%d.png' % (k + 1))
        fig.savefig(path, dpi=130)
        written.append(path)
        for a in axes:
            a.set_visible(False)
    return written


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='D-LI-Init 动态初始化每帧耗时分析（单窗口仪表盘）')
    p.add_argument('--log-dir', default=DEFAULT_LOG_DIR,
                   help='日志目录（默认 %s）' % DEFAULT_LOG_DIR)
    p.add_argument('--no-show', action='store_true',
                   help='不弹窗，导出 PNG 与 CSV 后退出')
    p.add_argument('--outdir', default=None,
                   help='--no-show 时的导出目录（默认 <log-dir>/frame_time_analysis）')
    p.add_argument('--pass', dest='pass_id', type=int, default=None,
                   help='只显示第 N 轮的帧级页面（第 5 页仍汇总全部轮次）')
    p.add_argument('--lio-window', type=float, default=LIO_WINDOW_SECS,
                   help='页面6: 进入LIO后的观察窗口长度(秒, 默认 %.0f)' % LIO_WINDOW_SECS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    log_dir = os.path.abspath(args.log_dir)
    outdir = args.outdir or os.path.join(log_dir, 'frame_time_analysis')

    data, error = load_frame_time(os.path.join(log_dir, MAIN_LOG))
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title('D-LI-Init 初始化阶段每帧耗时分析')
    if data is None:
        ax = new_axes(PLOT_RECT, 'err')
        show_empty(ax, error)
        print('[frame_time] %s' % error)
        if args.no_show:
            os.makedirs(outdir, exist_ok=True)
            fig.savefig(os.path.join(outdir, 'page1.png'), dpi=130)
            return 1
        plt.show()
        return 1

    add_derived(data)
    warns = check_data(data)
    iter_d, iter_error = load_iter_time(os.path.join(log_dir, ITER_LOG))
    summaries = pass_summary(data, iter_d)
    console_summary(data, summaries, warns, log_dir, iter_error)
    report_init_lag(log_dir)   # 只打印到终端, 不进入仪表盘

    lio_d, lio_error = load_lio_time(os.path.join(log_dir, LIO_LOG))
    lio_win, lio_meta = None, None
    if lio_d is not None:
        lio_win, lio_meta = lio_window(lio_d, args.lio_window)
    console_lio_summary(lio_win, lio_meta, lio_error, args.lio_window)

    view = data
    suffix = ''
    if args.pass_id is not None:
        sel = data['pass_id'] == args.pass_id
        if not np.any(sel):
            print('[frame_time] 告警: 日志中没有第 %d 轮' % args.pass_id)
        else:
            view = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
            view['_n'] = int(sel.sum())
            add_derived(view)
            suffix = '（仅第 %d 轮）' % args.pass_id

    pages = {}
    pages['1 每帧耗时分解'] = build_page_1(view, suffix)
    pages['2 帧间间隔'] = build_page_2(view, suffix)
    pages['3 ICP 内部构成'] = build_page_3(view, suffix)
    pages['4 归一化指标'] = build_page_4(view, suffix)
    pages['5 轮次汇总'] = build_page_5(data, summaries, '')
    pages['6 LIO后%gs窗口' % args.lio_window] = build_page_6(lio_win, lio_meta, args.lio_window)

    all_axes = [a for axs in pages.values() for a in axs]
    for a in all_axes:
        style_axes(a)
        a.set_visible(False)

    if args.no_show:
        written = export_pages(pages, outdir)
        csvs = list(write_csv(data, summaries, outdir)) + \
            write_lio_csv(lio_win, lio_meta, outdir, args.lio_window)
        print('[frame_time] 已导出 %d 张图: %s' % (len(written), outdir))
        for p in csvs:
            print('[frame_time] 已导出 %s' % p)
        return 1 if warns else 0

    rax = plt.axes([0.012, 0.71, 0.155, 0.22])
    radio = RadioButtons(rax, list(pages.keys()))
    for label in radio.labels:
        label.set_fontsize(TICK_SIZE)

    current = []

    def on_select(label):
        for a in all_axes:
            a.set_visible(False)
        current[:] = pages[label]
        for a in current:
            a.set_visible(True)
        fig.canvas.draw_idle()

    radio.on_clicked(on_select)
    on_select(list(pages.keys())[0])
    plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())
