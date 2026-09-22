#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# analyze_init_comparison.py — D-LI-Init(A, 动态) vs FAST-LIO2(B, 静态) 初始化状态对比分析
#
# 输入: --run-dir 目录(含 A_events/A_states/B_events/B_states.log 与 rtk_velocity.csv)
# 输出: <run-dir>/analysis/ 下配对CSV、summary.csv、comparison_report.md、*.png
#
# 日志格式(与两侧插桩一致):
#   A_states: t rtype applied cov prop inited imu_en eff vx vy vz bgx bgy bgz gx gy gz
#   B_states: t rtype boxplus cov valid_iter inited feats vx vy vz bgx bgy bgz gx gy gz
#   A_events: init_done t reason vel(3) bg(3) grav(3) pos(3) vel_p(3) bg_p(3) grav_p(3)
#             batch_solution t_first V0(3) V0i(3) gb(3) GL0(3) gi(3)
#             first_post t vel(3) bg(3) grav(3)
#   B_events: init_done t reason vel(3) bg(3) grav(3) pos(3); first_post t vel(3) bg(3) grav(3)
#   rtk csv : t,vx,vy,vz,speed
# =============================================================================
import argparse
import bisect
import math
import os
import sys

import numpy as np

G_NOMINAL = 9.81  # m/s^2 名义重力模长


# ------------------------------- 解析 -------------------------------
def _floats(tokens):
    try:
        return [float(x) for x in tokens]
    except ValueError:
        return None


def parse_states(path):
    """返回 dict: t[], rtype[], flag[], feats[], vel(N,3), bg(N,3), grav(N,3)"""
    t, rtype, flags, feats, vel, bg, grav = [], [], [], [], [], [], []
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            toks = line.split()
            # 列数自适应: A 侧 17 列(5 标志位+feats), B 侧 16 列(4 标志位+feats)
            if len(toks) >= 17:
                nflag, fi = 5, 7          # A: toks[2:7]=flags, toks[7]=feats
            elif len(toks) == 16:
                nflag, fi = 4, 6          # B: toks[2:6]=flags, toks[6]=feats
            else:
                continue
            vals = _floats(toks[0:1] + toks[2:])
            if vals is None:
                continue
            t.append(vals[0])
            rtype.append(toks[1])
            flags.append([int(float(x)) for x in toks[2:2 + nflag]])
            feats.append(int(float(toks[fi])))
            vel.append([float(x) for x in toks[fi + 1:fi + 4]])
            bg.append([float(x) for x in toks[fi + 4:fi + 7]])
            grav.append([float(x) for x in toks[fi + 7:fi + 10]])
    if not t:
        return None
    return dict(t=np.array(t), rtype=rtype, flags=flags, feats=np.array(feats),
                vel=np.array(vel), bg=np.array(bg), grav=np.array(grav))


def parse_events(path):
    """返回 dict: init_done{}, batch_solution{}, first_post{}(每类首条)"""
    ev = {'init_done': None, 'batch_solution': None, 'first_post': None}
    if not os.path.isfile(path):
        return ev
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            toks = line.split()
            kind = toks[0]
            if kind not in ev or ev[kind] is not None:
                continue
            if kind == 'init_done':
                vals = _floats(toks[1:2] + toks[3:])  # 跳过 reason 字符串字段
                if vals is None:
                    continue
                entry = dict(t=vals[0], reason=toks[2],
                             vel=np.array(vals[1:4]),
                             bg=np.array(vals[4:7]),
                             grav=np.array(vals[7:10]),
                             pos=np.array(vals[10:13]))
            else:
                vals = _floats(toks[1:])
                if vals is None:
                    continue
                entry = dict(t=vals[0])
                if kind == 'first_post':
                    entry['vel'] = np.array(vals[1:4])
                    entry['bg'] = np.array(vals[4:7])
                    entry['grav'] = np.array(vals[7:10])
                elif kind == 'batch_solution':
                    entry['V0'] = np.array(vals[1:4])
                    entry['V0_imu'] = np.array(vals[4:7])
                    entry['gyro_bias'] = np.array(vals[7:10])
                    entry['Grav_L0'] = np.array(vals[10:13])
                    entry['grav_imu'] = np.array(vals[13:16])
            ev[kind] = entry
    return ev


def parse_rtk(path):
    """返回 t[], v(N,3)"""
    if not os.path.isfile(path):
        return None
    t, v = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('t,'):
                continue
            toks = line.split(',')
            if len(toks) < 4:
                continue
            vals = _floats(toks[:4])
            if vals is None:
                continue
            t.append(vals[0])
            v.append(vals[1:4])
    if not t:
        return None
    order = np.argsort(t)
    return dict(t=np.array(t)[order], v=np.array(v)[order])


def nearest_rtk(t_q, rtk, tol):
    """最近邻配对, 返回 (idx, |dt|); 超出 tol 返回 (None, dt)"""
    if rtk is None:
        return None, float('nan')
    i = bisect.bisect_left(rtk['t'], t_q)
    best, bestd = None, float('inf')
    for j in (i - 1, i):
        if 0 <= j < len(rtk['t']):
            d = abs(rtk['t'][j] - t_q)
            if d < bestd:
                best, bestd = j, d
    if best is None or bestd > tol:
        return None, (bestd if best is not None else float('nan'))
    return best, bestd


# ------------------------------- 统计 -------------------------------
def _ms(arr):
    if len(arr) == 0:
        return float('nan'), float('nan')
    return float(np.mean(arr)), float(np.std(arr))


def analyze_side(tag, states, events, rtk, t0, win, tol):
    """返回 (paired行列表, summary dict)。paired 列:
    t rel_t rtype dt vx vy vz speed rvx rvy rvz rspeed evx evy evz espeed
    bgx bgy bgz gmag gx gy gz"""
    if states is None:
        return [], dict(side=tag, error='无状态日志')
    # D-LI-Init 迭代式初始化: init_done 时刻同一时间戳会重复写多行
    # (每迭代一遍一行, 首遍未收敛 vel 可达数十 m/s, grav=0)。
    # 同时间戳只保留最后一行(即该时刻收敛后的最终状态)。
    sel = (states['t'] >= t0 - 1e-6) & (states['t'] <= t0 + win)
    idx = np.where(sel)[0]
    if len(idx) > 1:
        keep = []
        for k in range(len(idx)):
            if k > 0 and states['t'][idx[k]] == states['t'][idx[k - 1]]:
                keep.pop()
            keep.append(idx[k])
        idx = np.array(keep)
    paired, unpaired = [], 0
    for i in idx:
        j, dt = nearest_rtk(states['t'][i], rtk, tol)
        if j is None:
            unpaired += 1
            continue
        v = states['vel'][i]
        rv = rtk['v'][j]
        g = states['grav'][i]
        paired.append([states['t'][i], states['t'][i] - t0, states['rtype'][i], dt] +
                      list(v) + [float(np.linalg.norm(v))] + list(rv) +
                      [float(np.linalg.norm(rv))] + list(v - rv) +
                      [float(np.linalg.norm(v) - np.linalg.norm(rv))] +
                      list(states['bg'][i]) +
                      [float(np.linalg.norm(g))] + list(g))
    n_by = {k: sum(1 for i in idx if states['rtype'][i] == k)
            for k in ('post', 'partial_applied', 'invalid', 'predict')}
    s = dict(side=tag, t_init=t0, n_frames=len(idx), n_by=n_by, unpaired=unpaired)
    if paired:
        P = np.array(paired, dtype=object)
        rvel = P[:, 8:11].astype(float)
        err = P[:, 12:15].astype(float)
        spd_err = P[:, 15].astype(float)
        bg = P[:, 16:19].astype(float)
        gmag = P[:, 19].astype(float)
        s['vel_err_mean'] = np.mean(err, axis=0)
        s['vel_err_std'] = np.std(err, axis=0)
        s['speed_err_mean'], s['speed_err_std'] = _ms(spd_err)
        s['speed_rmse'] = float(np.sqrt(np.mean(spd_err ** 2)))
        s['speed_rel_err'] = float(np.mean(np.abs(spd_err) /
                                           np.maximum(np.linalg.norm(rvel, axis=1), 1e-3)))
        s['bg_mean'] = np.mean(bg, axis=0)
        s['bg_std'] = np.std(bg, axis=0)
        s['gmag_mean'], s['gmag_std'] = _ms(gmag)
        s['gmag_err'] = float(np.mean(gmag) - G_NOMINAL)
        gdirs = P[:, 20:23].astype(float)
        n = np.linalg.norm(gdirs, axis=1, keepdims=True)
        n[n == 0] = 1.0
        s['grav_dir_mean'] = np.mean(gdirs / n, axis=0)
    return paired, s


def angle_between(u, v):
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# ------------------------------- 绘图 -------------------------------
def make_plots(outdir, paired_a, paired_b, t0_a, t0_b):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[analyze] matplotlib 不可用, 跳过绘图')
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    for ax, P, name, t0 in ((axes[0], paired_a, 'A(D-LI-Init dynamic)', t0_a),
                            (axes[1], paired_b, 'B(FAST-LIO2 static)', t0_b)):
        if not P:
            ax.set_title(name + ' no data')
            continue
        A = np.array([[r[0], r[1]] + list(r[4:7]) + [r[7]] + list(r[8:11]) + [r[11]]
                      for r in P], dtype=float)
        t = A[:, 0] - t0
        ax.plot(t, A[:, 2], label='vx')
        ax.plot(t, A[:, 3], label='vy')
        ax.plot(t, A[:, 4], label='vz')
        ax.plot(t, A[:, 5], 'k-', lw=1.5, label='|v|')
        ax.plot(t, A[:, 6], '--', label='RTK vx')
        ax.plot(t, A[:, 7], '--', label='RTK vy')
        ax.plot(t, A[:, 8], '--', label='RTK vz')
        ax.plot(t, A[:, 9], 'k--', lw=1.5, label='RTK |v|')
        ax.set_xlabel('t since init done (s)')
        ax.set_ylabel('v (m/s)')
        ax.set_title(name + ' velocity vs RTK')
        ax.legend(fontsize=7, ncol=4)
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'vel_vs_rtk.png'), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, P, name, t0 in ((axes[0], paired_a, 'A', t0_a), (axes[1], paired_b, 'B', t0_b)):
        if not P:
            continue
        # 动态初始化(A)进入 LIO 后重力模长不固定, 静态初始化(B)后模长恒定,
        # 模长对比无参考性, 故改画重力加速度三分量(gx, gy, gz)
        A = np.array([[r[0] - t0, r[20], r[21], r[22]] for r in P], dtype=float)
        ax.plot(A[:, 0], A[:, 1], label='gx')
        ax.plot(A[:, 0], A[:, 2], label='gy')
        ax.plot(A[:, 0], A[:, 3], label='gz')
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_title(name + ' gravity components')
        ax.set_xlabel('t (s)')
        ax.set_ylabel('g (m/s^2)')
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'gravity_components.png'), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, P, name, t0 in ((axes[0], paired_a, 'A', t0_a), (axes[1], paired_b, 'B', t0_b)):
        if not P:
            continue
        A = np.array([[r[0] - t0, r[16], r[17], r[18]] for r in P], dtype=float)
        ax.plot(A[:, 0], A[:, 1], label='bgx')
        ax.plot(A[:, 0], A[:, 2], label='bgy')
        ax.plot(A[:, 0], A[:, 3], label='bgz')
        ax.set_title(name + ' gyro bias (IMU frame)')
        ax.set_xlabel('t (s)')
        ax.set_ylabel('bg (rad/s)')
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'gyro_bias.png'), dpi=130)
    plt.close(fig)


# ------------------------------- 连续状态序列与仪表盘 -------------------------------
def _side_series(states, events, rtk, tol=0.06):
    """全时段连续状态序列(相对 init_done 时刻), 供仪表盘与 CSV 导出。
    与 analyze_side 的配对统计不同: 不限分析窗口、不要求 RTK 配对成功。"""
    if states is None:
        return None
    ev = events['init_done'] or events['first_post']
    if ev is None:
        return None
    t0 = ev['t']
    # 序列起点: first_post(收敛后首个有效后验帧)。此前的帧属于 D-LI-Init
    # 迭代式初始化的中间遍状态(vel 未收敛可达数十 m/s, grav=0), 且
    # init_done 时刻同一时间戳会重复写多行(每迭代一遍一行), 不能进图表。
    ev_start = events['first_post'] or events['init_done']
    m = states['t'] >= ev_start['t']
    # 去重防御: 同一时间戳保留最后一行(即该时刻收敛后的最终状态)
    keep = m.copy()
    for k in range(1, len(m)):
        if m[k] and m[k - 1] and states['t'][k] == states['t'][k - 1]:
            keep[k - 1] = False
    m = keep
    if not m.any():
        return None
    t = states['t'][m] - t0
    bg = states['bg'][m]
    # init 时刻零偏(变化量基准): 优先取事件自带的 bg, 回退到最接近 t=0 的帧
    i0 = int(np.argmin(np.abs(t)))
    bg0 = np.array(ev.get('bg', bg[i0]), dtype=float)
    S = dict(t=t, t0=t0, vel=states['vel'][m], bg=bg,
             grav=states['grav'][m], bg0=bg0)
    S['vmag'] = np.linalg.norm(S['vel'], axis=1)
    S['gmag'] = np.linalg.norm(S['grav'], axis=1)
    S['dbg'] = S['bg'] - bg0
    if rtk is not None:
        tr, vr = [], []
        for ti in states['t'][m]:
            j, _ = nearest_rtk(ti, rtk, tol)
            if j is not None:
                tr.append(ti - t0)
                vr.append(rtk['v'][j])
        S['rtk_t'] = np.array(tr, dtype=float)
        S['rtk_v'] = np.array(vr, dtype=float).reshape(-1, 3)
    else:
        S['rtk_t'] = np.zeros(0)
        S['rtk_v'] = np.zeros((0, 3))
    return S


def write_full_states(outdir, tag, S):
    """导出全时段连续状态 CSV(速度/零偏绝对+变化量/重力, 分量+模长)"""
    path = os.path.join(outdir, '%s_states_full.csv' % tag)
    with open(path, 'w') as f:
        f.write('# t rel_t vx vy vz vmag bgx bgy bgz d_bgx d_bgy d_bgz gx gy gz gmag\n')
        for i in range(len(S['t'])):
            row = [S['t'][i] + S['t0'], S['t'][i],
                   S['vel'][i, 0], S['vel'][i, 1], S['vel'][i, 2], S['vmag'][i],
                   S['bg'][i, 0], S['bg'][i, 1], S['bg'][i, 2],
                   S['dbg'][i, 0], S['dbg'][i, 1], S['dbg'][i, 2],
                   S['grav'][i, 0], S['grav'][i, 1], S['grav'][i, 2], S['gmag'][i]]
            f.write(' '.join('%.9f' % v for v in row) + '\n')
    print('[analyze] %s 侧连续状态序列 %d 帧 -> %s' % (tag, len(S['t']), path))


def _dash_axes(fig_title, nrow):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[analyze] matplotlib 不可用, 跳过仪表盘')
        return None
    fig, axes = plt.subplots(nrow, 1, figsize=(12, 3.2 * nrow), sharex=True)
    fig.suptitle(fig_title, fontsize=13)
    return fig, axes


def _decorate(axes):
    for ax in axes:
        ax.axvline(0, color='r', ls=':', lw=1.0, alpha=0.8)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7, ncol=4)


def make_dashboard(outdir, tag, S, title):
    """单侧仪表盘: 速度(分量+模长+RTK) / 重力(分量) / 零偏(绝对) / 零偏变化量"""
    ret = _dash_axes(title + ' — continuous LIO states (t=0 at init done)', 4)
    if ret is None:
        return
    fig, axes = ret
    import matplotlib.pyplot as plt
    t = S['t']

    ax = axes[0]                                    # 速度
    for i, lb in enumerate(('vx', 'vy', 'vz')):
        ax.plot(t, S['vel'][:, i], lw=0.9, label=lb)
    ax.plot(t, S['vmag'], 'k-', lw=1.8, label='|v|')
    if len(S['rtk_t']) > 0:
        for i, lb in enumerate(('RTK vx', 'RTK vy', 'RTK vz')):
            ax.plot(S['rtk_t'], S['rtk_v'][:, i], '--', lw=0.8, label=lb)
        ax.plot(S['rtk_t'], np.linalg.norm(S['rtk_v'], axis=1),
                'k--', lw=1.5, label='RTK |v|')
    ax.set_ylabel('velocity (m/s)')

    ax = axes[1]                                    # 重力(分量; 动态初始化后模长不固定, 不画模长)
    for i, lb in enumerate(('gx', 'gy', 'gz')):
        ax.plot(t, S['grav'][:, i], lw=0.9, label=lb)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_ylabel('gravity components (m/s^2)')

    ax = axes[2]                                    # 零偏绝对值
    for i, lb in enumerate(('bgx', 'bgy', 'bgz')):
        ax.plot(t, S['bg'][:, i], lw=1.0, label=lb)
    ax.set_ylabel('gyro bias (rad/s)')

    ax = axes[3]                                    # 零偏变化量(相对 init 时刻)
    for i, lb in enumerate(('d_bgx', 'd_bgy', 'd_bgz')):
        ax.plot(t, S['dbg'][:, i], lw=1.0, label=lb)
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_ylabel('gyro bias change (rad/s)')
    ax.set_xlabel('t since init done (s)')

    _decorate(axes)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outdir, 'dashboard_%s.png' % tag), dpi=130)
    plt.close(fig)
    print('[analyze] %s 侧仪表盘 -> %s' % (tag, os.path.join(outdir, 'dashboard_%s.png' % tag)))


def make_compare_dashboard(outdir, SA, SB):
    """A/B 对比仪表盘: 速度模长 / 重力分量 / 零偏变化量模长(重力因 A 模长不固定改为分量)"""
    ret = _dash_axes('A (D-LI-Init, dynamic) vs B (FAST-LIO2, static) — metrics', 3)
    if ret is None:
        return
    fig, axes = ret
    import matplotlib.pyplot as plt

    ax = axes[0]
    ax.plot(SA['t'], SA['vmag'], lw=1.4, label='A |v|')
    ax.plot(SB['t'], SB['vmag'], lw=1.4, label='B |v|')
    if len(SA['rtk_t']) > 0:
        ax.plot(SA['rtk_t'], np.linalg.norm(SA['rtk_v'], axis=1),
                'k--', lw=1.2, label='RTK |v|')
    ax.set_ylabel('|velocity| (m/s)')

    ax = axes[1]                                    # 重力分量(动态初始化后模长不固定, 不画模长)
    for i, lb in enumerate(('gx', 'gy', 'gz')):
        ax.plot(SA['t'], SA['grav'][:, i], lw=1.2, label='A ' + lb)
    for i, lb in enumerate(('gx', 'gy', 'gz')):
        ax.plot(SB['t'], SB['grav'][:, i], '--', lw=1.2, label='B ' + lb)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_ylabel('gravity components (m/s^2)\n(A/B world frames differ, per-axis for reference only)')

    ax = axes[2]
    ax.plot(SA['t'], np.linalg.norm(SA['dbg'], axis=1), lw=1.4, label='A |d_bg|')
    ax.plot(SB['t'], np.linalg.norm(SB['dbg'], axis=1), lw=1.4, label='B |d_bg|')
    ax.set_ylabel('|gyro bias change| (rad/s)')
    ax.set_xlabel('t since init done (s)')

    _decorate(axes)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, 'dashboard_compare.png'), dpi=130)
    plt.close(fig)
    print('[analyze] A/B 对比仪表盘 -> %s' % os.path.join(outdir, 'dashboard_compare.png'))


# ------------------------------- 报告 -------------------------------
def fmt_v(x, prec=4):
    if x is None:
        return 'N/A'
    try:
        if isinstance(x, (list, np.ndarray)):
            return '[' + ', '.join(('%.*f' % (prec, v)) for v in x) + ']'
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return 'NaN'
        return '%.*f' % (prec, x)
    except (TypeError, ValueError):
        return str(x)


def write_report(path, args, ea, eb, sa, sb, extra):
    L = []
    L.append('# D-LI-Init(动态) vs FAST-LIO2(静态) 初始化状态对比报告')
    L.append('')
    L.append('- 运行目录: `%s`' % args.run_dir)
    L.append('- 分析窗口: 初始化完成后 %.1f s; RTK 配对容差 %.3f s' % (args.window, args.tol))
    L.append('')
    for tag, ev, s in (('A (D-LI-Init, 动态)', ea, sa), ('B (FAST-LIO2, 静态)', eb, sb)):
        L.append('## %s' % tag)
        L.append('')
        if 'error' in s:
            L.append('**%s**' % s['error'])
            L.append('')
            continue
        src = 'init_done 事件' if ev['init_done'] else (
            'first_post 事件' if ev['first_post'] else '未找到(错误)')
        reason = ev['init_done']['reason'] if ev['init_done'] else 'N/A'
        L.append('| 项目 | 值 |')
        L.append('|---|---|')
        L.append('| 初始化完成判据 | %s (reason=%s) |' % (src, reason))
        L.append('| 窗口内帧数 | %d (post=%d, partial=%d, invalid=%d, predict=%d, 未配对=%d) |'
                 % (s.get('n_frames', 0), s['n_by']['post'], s['n_by']['partial_applied'],
                    s['n_by']['invalid'], s['n_by']['predict'], s.get('unpaired', 0)))
        if 'vel_err_mean' in s:
            L.append('| 速度误差均值 xyz (m/s) | %s |' % fmt_v(list(s['vel_err_mean'])))
            L.append('| 速度误差标准差 xyz | %s |' % fmt_v(list(s['vel_err_std'])))
            L.append('| 速度模长误差均值±std | %.4f ± %.4f |' % (s['speed_err_mean'], s['speed_err_std']))
            L.append('| 速度模长 RMSE | %.4f m/s |' % s['speed_rmse'])
            L.append('| 陀螺零偏均值 (rad/s) | %s |' % fmt_v(list(s['bg_mean'])))
            L.append('| 陀螺零偏标准差 | %s |' % fmt_v(list(s['bg_std'])))
            L.append('| 重力模长 | %.4f ± %.4f (与9.81差 %+.4f) |'
                     % (s['gmag_mean'], s['gmag_std'], s['gmag_err']))
            L.append('| 重力方向(单位向量均值) | %s |' % fmt_v(list(s['grav_dir_mean'])))
        L.append('')
    L.append('## A/B 交叉对比')
    L.append('')
    ok = 'grav_dir_mean' in sa and 'grav_dir_mean' in sb
    L.append('| 项目 | A(动态) | B(静态) |')
    L.append('|---|---|---|')
    if ok:
        ang = angle_between(sa['grav_dir_mean'], sb['grav_dir_mean'])
        L.append('| 重力方向夹角 | %.3f° | (两侧重力方向不一致源于各自世界系姿态约定, 仅作参考) |' % ang)
        L.append('| 重力模长差 | %+.4f | %.4f - %.4f |' % (
            sa['gmag_mean'] - sb['gmag_mean'], sa['gmag_mean'], sb['gmag_mean']))
        L.append('| 速度模长 RMSE | %.4f | %.4f |' % (sa['speed_rmse'], sb['speed_rmse']))
        L.append('| 速度模长相对误差 | %.2f%% | %.2f%% |' % (
            100 * sa['speed_rel_err'], 100 * sb['speed_rel_err']))
    else:
        L.append('| 数据不足 | 一侧或两侧无有效配对数据 | |')
    L.append('')
    if extra:
        L.append('## 备注')
        L.extend('- ' + e for e in extra)
        L.append('')
    L.append('> 注: A/B 世界坐标系各自独立(原点/姿态由各自初始化确定), 三轴速度与'
             '重力方向跨侧直接作差受姿态约定影响; 速度模长误差与重力模长为旋转不变量, 更具参考意义。')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


# ------------------------------- 主流程 -------------------------------
def main():
    ap = argparse.ArgumentParser(description='初始化状态对比分析')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--rtk', default=None, help='rtk_velocity.csv 路径(默认 <run-dir>/rtk_velocity.csv)')
    ap.add_argument('--window', type=float, default=10.0, help='初始化完成后的分析窗口(s)')
    ap.add_argument('--tol', type=float, default=0.05, help='RTK 最近邻配对时间容差(s)')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    rd = args.run_dir
    outdir = os.path.join(rd, 'analysis')
    os.makedirs(outdir, exist_ok=True)
    extra = []

    rtk_path = args.rtk or os.path.join(rd, 'rtk_velocity.csv')
    rtk = parse_rtk(rtk_path)
    if rtk is None:
        print('[analyze] 警告: RTK 数据缺失(%s), 速度误差统计不可用' % rtk_path)
        extra.append('RTK 数据缺失, 速度误差统计不可用')

    ea = parse_events(os.path.join(rd, 'A_events.log'))
    eb = parse_events(os.path.join(rd, 'B_events.log'))
    sa_all = parse_states(os.path.join(rd, 'A_states.log'))
    sb_all = parse_states(os.path.join(rd, 'B_states.log'))

    def t0_of(ev, st, tag):
        if ev['init_done']:
            return ev['init_done']['t'], 'init_done'
        if ev['first_post']:
            return ev['first_post']['t'], 'first_post'
        return None, None  # 不降级使用首帧: 首帧可能仍是初始化前数据, 定位会失真

    t0a, srca = t0_of(ea, sa_all, 'A')
    t0b, srcb = t0_of(eb, sb_all, 'B')
    if t0a is None or t0b is None:
        print('[analyze] 错误: 至少一侧缺少初始化事件, 分析中止(不伪造数据)')
        sys.exit(1)

    paired_a, sa = analyze_side('A', sa_all, ea, rtk, t0a, args.window, args.tol)
    paired_b, sb = analyze_side('B', sb_all, eb, rtk, t0b, args.window, args.tol)

    hdr = ('t rel_t rtype dt vx vy vz speed rvx rvy rvz rspeed '
           'evx evy evz espeed bgx bgy bgz gmag gx gy gz')
    for tag, P in (('A', paired_a), ('B', paired_b)):
        with open(os.path.join(outdir, '%s_paired.csv' % tag), 'w') as f:
            f.write('# ' + hdr + '\n')
            for r in P:
                f.write(' '.join(('%.9f' % v) if isinstance(v, float) else str(v) for v in r) + '\n')
        print('[analyze] %s 侧配对样本 %d 条' % (tag, len(P)))

    # 全时段连续状态序列(不限分析窗口): 供 CSV 导出与仪表盘
    SA = _side_series(sa_all, ea, rtk, args.tol)
    SB = _side_series(sb_all, eb, rtk, args.tol)
    for tag, S in (('A', SA), ('B', SB)):
        if S is not None:
            write_full_states(outdir, tag, S)

    if not args.no_plot:
        make_plots(outdir, paired_a, paired_b, t0a, t0b)
        titles = {'A': 'A (D-LI-Init, dynamic)', 'B': 'B (FAST-LIO2, static)'}
        for tag, S in (('A', SA), ('B', SB)):
            if S is not None:
                make_dashboard(outdir, tag, S, titles[tag])
        if SA is not None and SB is not None:
            make_compare_dashboard(outdir, SA, SB)

    write_report(os.path.join(rd, 'comparison_report.md'), args, ea, eb, sa, sb, extra)
    print('[analyze] 报告已生成: %s' % os.path.join(rd, 'comparison_report.md'))


if __name__ == '__main__':
    main()
