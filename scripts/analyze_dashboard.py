#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D-LI-Init 动态初始化分析仪表盘 (单窗口 RadioButtons 按钮切换)

读取日志 (由 laserMapping.cpp 写出, 目录: src/D-LI-Init/Log/):
  Li_D_init_fastlio.txt  15列: time vx vy vz bgx bgy bgz gx gy gz flag rtk_t rvx rvy rvz
      flag: 0=初始化前, 1=初始化完成时刻专行(速度/零偏/重力为初始化结果), 2=初始化后
      rtk_*: 帧时刻最近邻RTK速度样本 (ENU, m/s), 无样本时为 nan
  Li_D_init_iter_time.txt 5列: pass_id solve_ms pass_total_ms delta_g converged
      (首轮 delta_g=-1 表示预处理+首次求解, 无收敛判定)

页面: A 初始化质量摘要 | B 初始化后5s速度对比 | C 重力稳定性 | D 动态初始化耗时
坐标系说明: LIO vel_end/gravity 为 camera_init 系(倾斜), RTK 为 ENU 系;
          主要看模长 |v| 与垂直分量对比, 逐分量 xyz 仅参考。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

# ---------- 中文字体与统一可读性主题 ----------
TITLE_SIZE = 24
AXIS_TITLE_SIZE = 20
TICK_SIZE = 22
LEGEND_SIZE = 22
ANNOTATION_SIZE = 22
BAR_LABEL_SIZE = 24
EMPTY_SIZE = 24

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
LOG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'Log')

# 主绘图区 (右侧大区域)
PLOT_RECT = [0.19, 0.09, 0.78, 0.84]
WIN = 5.0  # 初始化后观察窗口 (秒)


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
    """安全读取数值日志；返回 (二维数组或 None, 错误说明)。"""
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


def load_lio():
    """读取主日志；兼容无 RTK 扩展列的旧格式。"""
    path = os.path.join(LOG_DIR, 'Li_D_init_fastlio.txt')
    data, error = safe_loadtxt(path, 11)
    if data is None:
        return None, error
    rtk_t = data[:, 11] if data.shape[1] > 11 else np.full(data.shape[0], np.nan)
    rtk_v = data[:, 12:15] if data.shape[1] >= 15 else np.full((data.shape[0], 3), np.nan)
    return {
        't': data[:, 0], 'v': data[:, 1:4], 'bg': data[:, 4:7],
        'g': data[:, 7:10], 'flag': data[:, 10],
        'rtk_t': rtk_t, 'rtk_v': rtk_v,
    }, None


def load_iter_time():
    path = os.path.join(LOG_DIR, 'Li_D_init_iter_time.txt')
    return safe_loadtxt(path, 5)


def find_init_row(d):
    """flag=1 专行; 无则降级取首个 flag=2 帧前一帧; 再无则取最后一帧"""
    idx1 = np.where(d['flag'] == 1)[0]
    if len(idx1) > 0:
        return idx1[-1], True
    idx2 = np.where(d['flag'] == 2)[0]
    if len(idx2) > 0:
        i = max(idx2[0] - 1, 0)
        return i, False
    return len(d['t']) - 1, False


def vertical_vel(v, g):
    """LIO 垂直速度: v 沿 -gravity 方向投影 (g 指向下方时 v·(-ĝ))"""
    gn = np.linalg.norm(g, axis=1, keepdims=True)
    gn[gn == 0] = 1.0
    return -np.sum(v * (g / gn), axis=1)


def gravity_norm(g):
    """返回单个三维重力向量模长；输入无效时返回 NaN。"""
    value = np.asarray(g, dtype=float)
    if value.ndim != 1 or value.shape != (3,) or not np.all(np.isfinite(value)):
        return np.nan
    return float(np.linalg.norm(value))


def gravity_angle_deg(g, reference):
    """计算单个或批量重力方向相对参考向量的夹角（度）。"""
    value = np.asarray(g, dtype=float)
    ref = np.asarray(reference, dtype=float)
    scalar_input = value.ndim == 1
    if scalar_input:
        value = value.reshape(1, -1)
    if value.ndim != 2 or value.shape[1] != 3 or ref.shape != (3,):
        return np.nan if scalar_input else np.full(value.shape[:1], np.nan)

    value_norm = np.linalg.norm(value, axis=1)
    ref_norm = np.linalg.norm(ref)
    valid = np.all(np.isfinite(value), axis=1) & np.isfinite(value_norm)
    valid &= np.isfinite(ref_norm) & (value_norm > 0) & (ref_norm > 0)
    result = np.full(value.shape[0], np.nan, dtype=float)
    if np.any(valid):
        cosine = np.sum(value[valid] * ref, axis=1) / (value_norm[valid] * ref_norm)
        result[valid] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(result[0]) if scalar_input else result


def gravity_axis_angles_deg(g):
    """返回重力向量与 X、Y、Z 三个坐标正轴的方向夹角（度）。"""
    value = np.asarray(g, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return np.full(3, np.nan, dtype=float)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm <= 0:
        return np.full(3, np.nan, dtype=float)
    cosine = np.clip(value / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def add_bar_labels(ax, bars, fmt='%.3f'):
    """为柱状图增加可读的数值标签。"""
    for bar in bars:
        value = bar.get_height()
        offset = 4 if value >= 0 else -4
        ax.annotate(fmt % value,
                    (bar.get_x() + bar.get_width() / 2.0, value),
                    xytext=(0, offset), textcoords='offset points',
                    ha='center', va='bottom' if value >= 0 else 'top',
                    fontsize=BAR_LABEL_SIZE, fontweight='bold')


def build_page_A(d, init_i, has_flag1):
    """页面 A: 初始化完成时的速度和状态质量摘要。"""
    gap_x = 0.035
    gap_y = 0.06
    width = (PLOT_RECT[2] - gap_x) / 2.0
    height = (PLOT_RECT[3] - gap_y) / 2.0
    left = PLOT_RECT[0]
    right = left + width + gap_x
    bottom = PLOT_RECT[1]
    top = bottom + height + gap_y
    ax_speed = plt.axes([left, top, width, height])
    ax_vertical = plt.axes([right, top, width, height])
    ax_bias = plt.axes([left, bottom, width, height])
    ax_gravity = plt.axes([right, bottom, width, height])

    v0 = d['v'][init_i]
    g0 = d['g'][init_i]
    bg0 = d['bg'][init_i]
    rv = d['rtk_v'][init_i]
    rtk_ok = np.all(np.isfinite(rv))
    lio_speed = np.linalg.norm(v0)
    colors = ['tab:blue', 'tab:orange']
    display_labels = ['动态初始化结果', 'RTK 参考结果']

    speed_values = [lio_speed]
    speed_labels = display_labels[:1]
    if rtk_ok:
        speed_values.append(np.linalg.norm(rv))
        speed_labels.append(display_labels[1])
    speed_x = np.arange(len(speed_values))
    bars = ax_speed.bar(speed_x, speed_values, color=colors[:len(speed_values)],
                        width=0.58)
    add_bar_labels(ax_speed, bars)
    ax_speed.set_xticks(speed_x)
    ax_speed.set_xticklabels(speed_labels)
    ax_speed.set_ylabel('速度模长 (m/s)')
    ax_speed.set_title('速度模长对比')
    ax_speed.grid(axis='y', ls='--', alpha=0.5)
    if not rtk_ok:
        ax_speed.text(0.5, 0.90, '未找到有效的 RTK 参考数据',
                      transform=ax_speed.transAxes, ha='center', va='top',
                      fontsize=ANNOTATION_SIZE, color='dimgray')
    style_axes(ax_speed)

    component_x = np.arange(3)
    component_width = 0.34
    lio_component_bars = ax_vertical.bar(
        component_x - component_width / 2.0, v0, component_width,
        color='tab:blue', label=display_labels[0])
    add_bar_labels(ax_vertical, lio_component_bars)
    if rtk_ok:
        rtk_component_bars = ax_vertical.bar(
            component_x + component_width / 2.0, rv, component_width,
            color='tab:orange', label=display_labels[1])
        add_bar_labels(ax_vertical, rtk_component_bars)
    ax_vertical.axhline(0, color='black', lw=0.8)
    ax_vertical.set_xticks(component_x)
    ax_vertical.set_xticklabels(['X 轴', 'Y 轴', 'Z 轴'])
    ax_vertical.set_ylabel('速度分量 (m/s)')
    ax_vertical.set_title('三轴速度分量对比')
    ax_vertical.grid(axis='y', ls='--', alpha=0.5)
    if not rtk_ok:
        ax_vertical.text(0.5, 0.87, '未找到有效的 RTK 参考数据',
                         transform=ax_vertical.transAxes, ha='center', va='top',
                         fontsize=ANNOTATION_SIZE, color='dimgray')
    style_axes(ax_vertical, legend=True, loc='best')

    bias_x = np.arange(3)
    bias_bars = ax_bias.bar(bias_x, bg0, color='tab:purple')
    add_bar_labels(ax_bias, bias_bars, '%.5f')
    ax_bias.axhline(0, color='black', lw=0.8)
    ax_bias.set_xticks(bias_x)
    ax_bias.set_xticklabels(['X 轴', 'Y 轴', 'Z 轴'])
    ax_bias.set_ylabel('陀螺仪零偏 (rad/s)')
    ax_bias.set_title('陀螺仪三轴零偏')
    ax_bias.grid(axis='y', ls='--', alpha=0.5)
    style_axes(ax_bias)

    gravity_values = gravity_axis_angles_deg(g0)
    gravity_x = np.arange(3)
    valid_gravity = np.isfinite(gravity_values)
    if np.any(valid_gravity):
        gravity_bars = ax_gravity.bar(
            gravity_x[valid_gravity], gravity_values[valid_gravity],
            color=np.asarray(['tab:red', 'tab:green', 'tab:blue'])[valid_gravity],
            width=0.58)
        add_bar_labels(ax_gravity, gravity_bars, '%.2f°')
    else:
        show_empty(ax_gravity, '重力方向无效，无法计算方向角')
    ax_gravity.set_xticks(gravity_x)
    ax_gravity.set_xticklabels(['与 X 正轴夹角', '与 Y 正轴夹角', '与 Z 正轴夹角'])
    ax_gravity.set_ylabel('方向夹角 (°)')
    ax_gravity.set_ylim(0, 190)
    ax_gravity.set_title('重力向量相对三轴正方向的夹角')
    ax_gravity.grid(axis='y', ls='--', alpha=0.5)
    ax_gravity.text(
        0.5, 0.97,
        '0° 表示同向，90° 表示垂直，180° 表示反向；角度属于初始化局部坐标系',
        transform=ax_gravity.transAxes, ha='center', va='top',
        fontsize=ANNOTATION_SIZE, color='dimgray')
    style_axes(ax_gravity)

    status = ('初始化完成时的结果' if has_flag1
              else '未找到专用结果记录，使用邻近状态')
    fig_note = ('初始化结果摘要  t=%.3f  |  %s\n'
                '速度保留原始三轴分量；不同坐标系未经转换时不直接计算逐轴误差。') % (
                    d['t'][init_i], status)
    ax_bias.text(1.05, -0.25, fig_note, transform=ax_bias.transAxes,
                 ha='center', va='top', fontsize=ANNOTATION_SIZE,
                 color='dimgray')
    return [ax_speed, ax_vertical, ax_bias, ax_gravity]


def build_page_B(d, init_i):
    """页面 B: 初始化结果作为 0 秒首点，随后展示 flag=2 正式 LIO 状态。"""
    gap = 0.04
    h1 = PLOT_RECT[3] * 0.62
    h2 = PLOT_RECT[3] * 0.30
    ax1 = plt.axes([PLOT_RECT[0], PLOT_RECT[1] + h2 + gap,
                    PLOT_RECT[2], h1])
    ax2 = plt.axes([PLOT_RECT[0], PLOT_RECT[1],
                    PLOT_RECT[2], h2])

    t0 = d['t'][init_i]
    post_idx = np.where(
        (d['t'] > t0 + 1e-9)
        & (d['t'] <= t0 + WIN)
        & (d['flag'] == 2)
    )[0]
    if len(post_idx) > 0:
        _, unique_pos = np.unique(d['t'][post_idx], return_index=True)
        post_idx = post_idx[np.sort(unique_pos)]
    idx = np.concatenate(([init_i], post_idx))

    t = d['t'][idx] - t0
    v = d['v'][idx]
    g = d['g'][idx]
    rv = d['rtk_v'][idx]
    vnorm = np.linalg.norm(v, axis=1)
    rvn = np.linalg.norm(rv, axis=1)
    vup = vertical_vel(v, g)
    rvup = rv[:, 2] if rv.shape[1] == 3 else np.full(len(t), np.nan)
    ax1.plot(t, vnorm, 'b-', lw=1.5,
             label='LIO 速度模长（0 秒为初始化结果）')
    ax1.plot(t, rvn, 'r-', lw=1.5, label='RTK 参考速度模长')
    ax1.plot(t, vup, 'b--', lw=1.0,
             label='LIO 垂直速度（0 秒为初始化结果）')
    ax1.plot(t, rvup, 'r--', lw=1.0,
             label='RTK 垂直速度（东-北-天坐标，正值向上）')
    ax1.set_ylabel('速度 (m/s)')
    ax1.set_title('初始化后 %d 秒速度对比（实线为模长，虚线为垂直速度）' % int(WIN))
    style_axes(ax1, legend=True, loc='best')
    ax1.grid(ls='--', alpha=0.5)
    ax1.tick_params(labelbottom=False)  # 上图与下图共享横轴, 隐藏上图刻度标签
    res = np.abs(vnorm - rvn)
    res[~np.isfinite(res)] = np.nan
    ax2.plot(t, res, 'g-', lw=1.0)
    ax2.set_xlabel('距初始化完成的时间 (s)  [起始时刻 %.3f]' % t0)
    ax2.set_ylabel('速度模长差值 (m/s)')
    ax2.set_title('速度模长差值（越接近 0 表示两者越一致）')
    ax2.grid(ls='--', alpha=0.5)
    return [ax1, ax2]


def build_page_C(d, t0, g_ref, has_flag1):
    """页面 C: 使用初始化完成后的有效状态评价重力方向稳定性。"""
    gap = 0.05
    h_top = PLOT_RECT[3] * 0.58
    h_bottom = PLOT_RECT[3] * 0.32
    ax_components = plt.axes([PLOT_RECT[0], PLOT_RECT[1] + h_bottom + gap,
                              PLOT_RECT[2], h_top])
    ax_angle = plt.axes([PLOT_RECT[0], PLOT_RECT[1],
                         PLOT_RECT[2], h_bottom])
    m = ((d['t'] >= t0 - 1e-9) & (d['t'] <= t0 + WIN)
         & (d['flag'] == 2))
    t = d['t'][m] - t0
    g = d['g'][m]
    valid = np.all(np.isfinite(g), axis=1) & np.isfinite(t)
    t, g = t[valid], g[valid]
    if len(t) == 0:
        show_empty(ax_components, '初始化完成后的观察窗口内没有有效重力数据')
        show_empty(ax_angle, '没有可用于评价重力方向稳定性的数据')
        return [ax_components, ax_angle]

    ax_components.plot(t, g[:, 0], lw=1.5, label='X 轴分量')
    ax_components.plot(t, g[:, 1], lw=1.5, label='Y 轴分量')
    ax_components.plot(t, g[:, 2], lw=1.5, label='Z 轴分量')
    ax_components.set_ylabel('重力分量 (m/s²)')
    ax_components.set_title('初始化完成后 %d 秒内的重力估计' % int(WIN))
    ax_components.grid(ls='--', alpha=0.5)
    ax_components.tick_params(labelbottom=False)
    style_axes(ax_components, legend=True, loc='upper left', ncol=3)

    ax_norm = ax_components.twinx()
    ax_norm.plot(t, np.linalg.norm(g, axis=1), color='tab:gray', ls='--',
                 lw=1.3, label='重力模长')
    ax_norm.set_ylabel('重力模长 (m/s²)', color='tab:gray')
    ax_norm.tick_params(axis='y', colors='tab:gray', labelsize=TICK_SIZE)
    ax_norm.legend(loc='upper right', fontsize=LEGEND_SIZE)

    angle = gravity_angle_deg(g, g_ref)
    angle_valid = np.isfinite(angle)
    if not np.any(angle_valid):
        show_empty(ax_angle, '初始化完成时的参考重力无效，无法评价方向稳定性')
        return [ax_components, ax_norm, ax_angle]
    angle_t, angle = t[angle_valid], angle[angle_valid]
    mean_angle = np.mean(angle)
    std_angle = np.std(angle)
    max_angle = np.max(angle)
    reference_label = ('初始化完成时的重力方向' if has_flag1
                       else '邻近状态的重力方向')
    ax_angle.plot(angle_t, angle, color='tab:green', lw=1.6,
                  label='相对%s' % reference_label)
    ax_angle.axhline(0, color='black', ls='--', lw=1,
                     label='方向保持不变（0°）')
    ax_angle.fill_between(angle_t, np.maximum(0.0, mean_angle - std_angle),
                          mean_angle + std_angle, color='tab:green', alpha=0.15,
                          label='平均值上下一个标准差')
    ax_angle.set_xlabel('距初始化完成的时间 (s)  [起始时刻 %.3f]' % t0)
    ax_angle.set_ylabel('重力方向变化角 (°)')
    ax_angle.set_title('重力方向变化角：平均 %.4f°，波动 %.4f°，最大 %.4f°' %
                       (mean_angle, std_angle, max_angle))
    ax_angle.grid(ls='--', alpha=0.5)
    style_axes(ax_angle, legend=True, loc='best')
    return [ax_components, ax_norm, ax_angle]


def build_page_D(iter_data, unused_lio_time=None):
    """页面 D: 仅展示动态初始化各轮纯计算耗时。"""
    ax1 = plt.axes([PLOT_RECT[0], 0.54, PLOT_RECT[2], 0.39])
    ax2 = plt.axes([PLOT_RECT[0], 0.09, PLOT_RECT[2], 0.35])
    if iter_data is None or len(iter_data) == 0:
        show_empty(ax1, '无动态初始化迭代耗时数据\n需要 Li_D_init_iter_time.txt（至少 5 列）')
        show_empty(ax2, '无法生成动态初始化总耗时摘要')
        return [ax1, ax2]

    values = iter_data[:, :5]
    valid = np.all(np.isfinite(values[:, :3]), axis=1)
    values = values[valid]
    if len(values) == 0:
        show_empty(ax1, '迭代日志中没有有效耗时记录')
        show_empty(ax2, '无法生成动态初始化总耗时摘要')
        return [ax1, ax2]

    pid = values[:, 0]
    solve_ms = np.clip(values[:, 1], 0, None)
    pass_ms = np.clip(values[:, 2], 0, None)
    delta_g = values[:, 3]
    conv = values[:, 4]
    replay_ms = np.clip(pass_ms - solve_ms, 0, None)

    replay_bars = ax1.bar(pid, replay_ms, 0.62, label='回放处理净耗时', color='tab:blue')
    solve_bars = ax1.bar(pid, solve_ms, 0.62, bottom=replay_ms,
                         label='批量求解耗时', color='tab:orange')
    for p, total in zip(pid, pass_ms):
        ax1.annotate('%.1f' % total, (p, total), xytext=(0, 4),
                     textcoords='offset points', ha='center', va='bottom',
                     fontsize=BAR_LABEL_SIZE, fontweight='bold')
    conv_i = np.where(conv > 0)[0]
    if len(conv_i) > 0:
        ax1.axvline(pid[conv_i[-1]], color='green', ls='--', lw=1.5,
                    label='收敛轮')
    ax1.set_xlabel('优化轮次（首轮不进行重力变化收敛判断）')
    ax1.set_ylabel('计算耗时 (ms)')
    ax1.set_title('动态初始化每轮计算耗时（柱顶为整轮总耗时）')
    ax1.set_xticks(pid.astype(int))
    ax1.grid(axis='y', ls='--', alpha=0.5)
    style_axes(ax1, legend=True, loc='best')

    ax1_delta = ax1.twinx()
    delta_mask = np.isfinite(delta_g) & (delta_g >= 0)
    if np.any(delta_mask):
        ax1_delta.plot(pid[delta_mask], delta_g[delta_mask], 'o--',
                       color='tab:green', lw=1.4,
                       label='本轮重力方向变化')
    ax1_delta.set_ylabel('本轮重力方向变化量')
    style_axes(ax1_delta, legend=np.any(delta_mask), loc='upper right')

    total_pass = np.sum(pass_ms)
    total_solve = np.sum(solve_ms)
    total_replay = np.sum(replay_ms)
    totals = [total_replay, total_solve]
    bars = ax2.barh(['回放处理', '批量求解'], totals,
                    color=['tab:blue', 'tab:orange'], height=0.55)
    for bar, value in zip(bars, totals):
        ratio = 100.0 * value / total_pass if total_pass > 0 else 0.0
        ax2.text(value, bar.get_y() + bar.get_height() / 2,
                 '  %.1f ms (%.1f%%)' % (value, ratio), va='center',
                 fontsize=BAR_LABEL_SIZE, fontweight='bold')
    final_state = '已收敛' if np.any(conv > 0) else '未标记收敛'
    summary = ('总计算耗时 %.1f ms  |  %d 轮  |  平均 %.1f ms/轮  |  %s\n'
               '口径：各轮 pass_total_ms 求和；不含数据累积、bag 等待及初始化后正式 LIO') % (
                   total_pass, len(pid), total_pass / len(pid), final_state)
    ax2.set_xlabel('累计计算耗时 (ms)')
    ax2.set_title(summary)
    ax2.grid(axis='x', ls='--', alpha=0.5)
    style_axes(ax2)
    return [ax1, ax1_delta, ax2]


def main():
    d, lio_error = load_lio()
    iter_data, iter_error = load_iter_time()

    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title('D-LI-Init 动态初始化分析仪表盘')

    if d is None:
        ax = plt.axes(PLOT_RECT)
        show_empty(ax, lio_error)
        print(lio_error)
        plt.show()
        return

    init_i, has_flag1 = find_init_row(d)
    t0 = d['t'][init_i]
    print('日志目录: %s' % LOG_DIR)
    print('共 %d 帧，初始化结果位于第 %d 行，状态编号为 %d，时刻 %.3f'
         % (len(d['t']), init_i, d['flag'][init_i], t0))

    pages = {}
    pages['A 初始化时刻'] = build_page_A(d, init_i, has_flag1)
    pages['B 速度对比'] = build_page_B(d, init_i)
    pages['C 重力'] = build_page_C(d, t0, d['g'][init_i], has_flag1)
    pages['D 耗时分析'] = build_page_D(iter_data, None)
    if iter_error:
        print(iter_error)

    all_axes = [a for axs in pages.values() for a in axs]
    for a in all_axes:
        style_axes(a)
        a.set_visible(False)

    rax = plt.axes([0.012, 0.71, 0.155, 0.25])
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


if __name__ == '__main__':
    main()