#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# test_analyzer.py — analyze_init_comparison.py 的合成数据单元测试
# 构造已知真值的 A/B 事件/状态日志与 RTK CSV, 运行分析器后校验:
#   1) 日志解析(16字段状态行/事件行)
#   2) 初始化时刻定位(init_done > first_post > 首帧)
#   3) RTK 最近邻配对数量与容差
#   4) 速度误差/重力模长/零偏统计的数值正确性
#   5) 缺失侧数据时如实报错退出(不伪造)
# =============================================================================
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYZER = os.path.join(HERE, 'analyze_init_comparison.py')

T0A, T0B = 100.0, 120.0          # 两侧初始化完成时刻
WIN, RATE = 10.0, 10.0          # 窗口10s, 10Hz
G = np.array([0.0, 0.0, -9.81])
BG = np.array([0.001, -0.002, 0.0005])
FAILURES = []


def check(name, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    print('[%s] %s %s' % (tag, name, detail))
    if not cond:
        FAILURES.append(name)


def vel_true(t):                 # 1 m/s 递增的已知真值
    return np.array([0.5 * (t - T0A), 1.0, -0.3])


def write_side(run, tag, t0, rtype_seq, v_func, nflag=5):
    ev = open(os.path.join(run, '%s_events.log' % tag), 'w')
    st = open(os.path.join(run, '%s_states.log' % tag), 'w')
    ev.write('# test header\n')
    if nflag == 5:   # A 侧真实格式: 17 列(5 标志位 + feats)
        st.write('# t_state record_type f0 f1 f2 f3 f4 vx vy vz bgx bgy bgz gx gy gz\n')
        flag_str = '1 1 1 1 1'
    else:             # B 侧真实格式: 16 列(4 标志位 + feats)
        st.write('# t_state record_type f0 f1 f2 f3 vx vy vz bgx bgy bgz gx gy gz\n')
        flag_str = '1 1 1 1'
    for k in range(int(WIN * RATE) + 3):
        t = t0 + k / RATE
        v = v_func(t)
        rtype = rtype_seq[k % len(rtype_seq)]
        st.write('%.9f %s %s 500 %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n'
                 % (t, rtype, flag_str, v[0], v[1], v[2], BG[0], BG[1], BG[2], G[0], G[1], G[2]))
    ev.write('init_done %.9f converged %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f '
             '%.9f %.9f %.9f 0 0 0 0 0 0 0 0 0\n'
             % (t0, 0.1, 0.2, 0.3, BG[0], BG[1], BG[2], G[0], G[1], G[2],
                1.0, 2.0, 3.0))
    ev.write('first_post %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n'
             % (t0 + 0.1, 0.1, 0.2, 0.3, BG[0], BG[1], BG[2], G[0], G[1], G[2]))
    ev.close()
    st.close()


def make_run(dirpath):
    os.makedirs(dirpath, exist_ok=True)
    # RTK: 20Hz, 与真值恒差 0.5 m/s(三轴), 用于误差校验
    with open(os.path.join(dirpath, 'rtk_velocity.csv'), 'w') as f:
        f.write('t,vx,vy,vz,speed\n')
        for k in range(int((min(T0A, T0B) - 1) * 20), int((max(T0A, T0B) + WIN + 2) * 20)):
            t = k / 20.0
            v = vel_true(t) + 0.5
            f.write('%.6f,%.6f,%.6f,%.6f,%.6f\n' % (t, v[0], v[1], v[2], np.linalg.norm(v)))
    # A: 全 post; B: post/partial/invalid 轮换(仍计入窗口统计); B 用真实 16 列格式
    write_side(dirpath, 'A', T0A, ['post'], vel_true)
    write_side(dirpath, 'B', T0B, ['post', 'partial_applied', 'invalid'],
               lambda t: vel_true(t) + np.array([0.0, 0.0, 0.0]), nflag=4)


def run_analyzer(run, extra=None):
    cmd = [sys.executable, ANALYZER, '--run-dir', run, '--window', str(WIN), '--tol', '0.05']
    if extra:
        cmd += extra
    return subprocess.run(cmd, capture_output=True, text=True)


def read_paired(run, tag):
    rows = []
    with open(os.path.join(run, 'analysis', '%s_paired.csv' % tag)) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            rows.append(line.split())
    return rows


def test_main():
    run = tempfile.mkdtemp(prefix='init_cmp_test_')
    try:
        make_run(run)
        r = run_analyzer(run, ['--no-plot'])
        check('分析器退出码0', r.returncode == 0, r.stderr[-300:] if r.returncode else '')

        pa, pb = read_paired(run, 'A'), read_paired(run, 'B')
        check('A配对数=窗口内帧数', len(pa) == int(WIN * RATE) + 1, 'got %d' % len(pa))
        check('B配对数>0(部分类型帧也统计)', len(pb) == int(WIN * RATE) + 1, 'got %d' % len(pb))

        # 速度误差: A 的 vel 与 RTK 三轴恒差 -0.5
        errs = np.array([[float(x) for x in row[12:15]] for row in pa])
        check('A三轴速度误差均值=-0.5',
              np.allclose(errs.mean(axis=0), -0.5, atol=1e-6), str(errs.mean(axis=0)))
        check('A三轴速度误差标准差=0',
              np.allclose(errs.std(axis=0), 0.0, atol=1e-6))

        # 配对时间差不超过容差
        dts = [abs(float(row[3])) for row in pa + pb]
        check('配对dt<=容差0.05', max(dts) <= 0.05, 'max=%.4f' % max(dts))

        # 重力模长 9.81, 误差统计
        gm = np.array([float(row[19]) for row in pa])
        check('A重力模长=9.81', np.allclose(gm, 9.81, atol=1e-6))
        bgm = np.array([[float(x) for x in row[16:19]] for row in pa])
        check('A零偏均值=BG', np.allclose(bgm.mean(axis=0), BG, atol=1e-9))

        # 报告存在且包含关键小节
        rep = os.path.join(run, 'comparison_report.md')
        check('报告已生成', os.path.isfile(rep))
        if os.path.isfile(rep):
            txt = open(rep).read()
            for key in ('初始化完成判据', 'init_done', '速度误差均值', '重力模长', 'A/B 交叉对比'):
                check('报告含"%s"' % key, key in txt)
            check('报告post计数', 'post=%d' % (int(WIN * RATE) + 1) in txt)
        else:
            txt = ''

        # 缺失一侧事件 -> 非零退出
        run2 = tempfile.mkdtemp(prefix='init_cmp_test2_')
        make_run(run2)
        os.remove(os.path.join(run2, 'B_events.log'))
        r2 = run_analyzer(run2, ['--no-plot'])
        check('缺失事件侧非零退出', r2.returncode != 0 and '中止' in (r2.stdout + r2.stderr))
        shutil.rmtree(run2, ignore_errors=True)
    finally:
        shutil.rmtree(run, ignore_errors=True)
    print('\n%d 项失败' % len(FAILURES))
    sys.exit(1 if FAILURES else 0)


if __name__ == '__main__':
    test_main()