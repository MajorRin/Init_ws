#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_init_frame_time.py 自检脚本（不依赖真实 bag / ROS）

合成日志 -> 解析 -> 校验 -> 导出，覆盖计划 §6 要求的全部用例：
  1  3 轮 × 30 帧合成日志 -> 90 行, 其中 row_type==0 为 87 行, row_type==1 为 3 行
  2  加法自洽（Σ分项 ≡ frame_total_ms）-> 不告警
  3  -1 哨兵不进任何统计（build_ms 仅在 3 个 row_type==1 行上有限）
  4  跨 pass 的 t 完全重复时仍按 pass_id 正确分轮
  5  knn+solve > icp 越界 -> 只告警不崩溃（无 Traceback, 仍出图出表）
  6  主日志缺失 -> 非零退出
  7  可选日志 Li_D_init_iter_time.txt 缺失 -> 降级但仍出图, 退出码 0
  8  帧间间隔 = t_wall_ms 逐差
  9  采集在末轮中途结束(iter 日志缺该轮) -> 不 KeyError, 该轮留空, 退出码 0
 10  首轮 delta_g=-1 为 C++ 哨兵(无收敛判定) -> 归一为 NaN/未知, 不画假柱
 11  LIO 12列日志(含 t_wall_ms): 完成间隔 = t_wall 逐差, 10s 墙钟窗口截取,
     gap = 间隔-净耗时, 点数漏斗解析/摘要/CSV, page6 三子图, 吞吐判定文案
 12  LIO 旧版8列日志: 无墙钟 -> 完成间隔全 NaN, 按数据时间截窗口, 提示重新采集
 13  缺 LIO 日志 -> 页面6 空数据页, 退出码 0
14  LIO 9列中间版仅墙钟无点数 -> 页面6 中图降级提示, 退出码 0
用法: python3 test_analyzer.py   （退出码 0 = 全部通过）
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if '--no-show' not in sys.argv:          # 让被测模块选 Agg 后端, 避免无显示环境报错
    sys.argv.append('--no-show')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import analyze_init_frame_time as azt

PASSES = 3
FRAMES_PER_PASS = 30
MAIN_LOG = azt.MAIN_LOG
ITER_LOG = azt.ITER_LOG
LIO_LOG = azt.LIO_LOG

_failures = []


def check(name, cond, detail=''):
    print('  [%s] %s%s' % ('PASS' if cond else 'FAIL', name,
                           ('  <- %s' % detail) if (detail and not cond) else ''))
    if not cond:
        _failures.append(name)


def fmt_row(values):
    return ' '.join('%.6f' % v for v in values)


def make_row(t, t_wall, frame_id, pass_id, m, row_type, imu_en,
             total, imu, down, icp, knn, solve, pubod, mapinc, pubtl,
             iters, knn_cnt, rematch, eff, down_size, full_size, st, end, res, build):
    return fmt_row([t, t_wall, frame_id, pass_id, m, row_type, imu_en,
                    total, imu, down, icp, knn, solve, pubod, mapinc, pubtl,
                    iters, knn_cnt, rematch, eff, down_size, full_size,
                    st, end, res, build])


def write_main_log(path, bad_icp=False):
    """合成 3 轮 × 30 帧；每轮首帧 (m==0) 为 row_type=1 的建树跳过帧。"""
    lines = ['# ' + ' '.join(azt.COLUMNS)]
    t_wall = 1000.0
    for p in range(PASSES):
        for m in range(FRAMES_PER_PASS):
            frame_id = p * FRAMES_PER_PASS + m
            t = 1000.0 + 0.1 * m          # 跨轮逐字重复 —— 故意与 pass 无关
            imu_en = 0 if p == 0 else 1   # pass 0 走 Point3D 路径
            imu = 20.0 + 0.1 * m
            down = 5.0
            if m == 0:
                # row_type=1: 建树后 continue。build_ms 在最后一列。
                build = 120.0 + 10.0 * p
                total = imu + down + build
                t_wall += total + 5.0
                lines.append(make_row(
                    t, t_wall, frame_id, p, m, 1, imu_en, total, imu, down,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10000.0, 12000.0,
                    -1, -1, -1, build))
                continue
            knn = 40.0 + 0.5 * m
            solve = 12.0
            icp = 55.0 + 0.5 * m
            if bad_icp and frame_id % 10 == 3:     # 故意越界: knn+solve > icp
                icp = 30.0
            pubod = 3.0
            mapinc = 4.0
            pubtl = 2.0
            total = imu + down + icp + pubod + mapinc + pubtl
            t_wall += total + 5.0
            lines.append(make_row(
                t, t_wall, frame_id, p, m, 0, imu_en, total, imu, down,
                icp, knn, solve, pubod, mapinc, pubtl,
                4, 4, 1, 8000.0, 10000.0, 12000.0,
                20000.0 + 500.0 * frame_id, 20500.0 + 500.0 * frame_id,
                0.05 + 0.001 * m, -1))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_iter_log(path):
    """每轮恰好一行 —— 与 C++ 一致: 三个写入点(2179/2256/2266)之后都紧跟 pass_id++,
    故同一个 pass_id 不会出现两行(实机日志 4 轮 = 4 行, 与此相符)。
    首轮 delta_g 写 -1 哨兵(2178 注释: 预处理+首次求解, 无收敛判定)。"""
    lines = ['# pass_id solve_ms pass_total_ms delta_g converged']
    for p in range(PASSES):
        lines.append(fmt_row([p, 1700.0 + p, 1800.0 + 10.0 * p,
                              -1.0 if p == 0 else 0.5 ** (p + 1),
                              1 if p == PASSES - 1 else 0]))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_lio_log(path, rows=250, has_wall=True, has_pts=True):
    """合成 LIO 阶段(初始化完成后)日志: 积压排空场景 —— 完成间隔 50ms > 输入周期 33.3ms。

    净耗时 42ms(其中 icp=knn28+solve7+其余3, 发布等未埋点 1.08), 数据时间步进
    33.333ms(输入周期), 墙钟步进 50ms(真实完成间隔)。
    12 列为最新版(墙钟+3列点数), 9 列为中间版(仅墙钟), 8 列为旧版。
    点数漏斗: 原始 6000 -> 降采样 2400(保留 40%) -> ESKF 有效 1000(占降采样 41.7%)。
    """
    header = '# time frame_total_ms imu_ms downfilter_ms icp_ms knn_ms solve_ms' \
             ' map_increment_ms'
    if has_wall:
        header += ' t_wall_ms'
    if has_pts:
        header += ' effect_feat_num feats_down_size feats_full_size'
    lines = [header]
    t_data, t_wall = 353.0, 20000.0
    for _ in range(rows):
        imu, down, knn, solve, icp_net, mapinc = 0.3, 0.12, 28.0, 7.0, 3.0, 2.5
        icp = knn + solve + icp_net
        total = imu + down + icp + mapinc + 1.08
        row = [t_data, total, imu, down, icp, knn, solve, mapinc]
        if has_wall:
            row.append(t_wall)
        if has_pts:
            row += [1000, 2400, 6000]
        lines.append(fmt_row(row))
        t_data += 0.033333
        t_wall += 50.0
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_cli(log_dir, outdir, extra=None):
    cmd = [sys.executable, os.path.join(HERE, 'analyze_init_frame_time.py'),
           '--log-dir', log_dir, '--no-show', '--outdir', outdir]
    if extra:
        cmd += extra
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc


def new_dir(root, name):
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    return path


def main():
    root = tempfile.mkdtemp(prefix='frame_time_test_')
    try:
        # ---------- 用例 1-4、8: 正常数据 ----------
        print('用例 1-4、8: 正常合成日志')
        d1 = new_dir(root, 'normal')
        write_main_log(os.path.join(d1, MAIN_LOG))
        write_iter_log(os.path.join(d1, ITER_LOG))

        data, err = azt.load_frame_time(os.path.join(d1, MAIN_LOG))
        check('主日志解析成功', data is not None, err)
        check('总行数=90', data['_n'] == 90, '实际 %s' % data['_n'])
        n0 = int((data['row_type'] == 0).sum())
        n1 = int((data['row_type'] == 1).sum())
        check('row_type 分布 0->87 1->3', (n0, n1) == (87, 3), '实际 %d/%d' % (n0, n1))
        check('pass_id 分轮 3 轮 × 30 帧',
              [int((data['pass_id'] == p).sum()) for p in range(PASSES)] == [30] * PASSES)

        # 用例 4: t 跨轮完全重复
        t_by_pass = [data['t'][data['pass_id'] == p] for p in range(PASSES)]
        check('t 跨 pass 确实逐字重复（构造前提成立）',
              np.array_equal(t_by_pass[0], t_by_pass[1]) and
              np.array_equal(t_by_pass[1], t_by_pass[2]))
        summaries = azt.pass_summary(data, None)
        check('按 pass_id 分轮成功（t 重复不影响分轮）',
              len(summaries) == PASSES and all(s['rows'] == 30 for s in summaries))

        # 用例 3: -1 哨兵
        check('build_ms 仅 3 个有限值（其余为 -1 哨兵）',
              int(np.isfinite(data['build_ms']).sum()) == 3,
              '实际 %d' % int(np.isfinite(data['build_ms']).sum()))
        check('row_type=1 行的 icp_ms 全为 NaN',
              bool(np.all(~np.isfinite(data['icp_ms'][data['row_type'] == 1]))))
        check('row_type=0 行的 icp_ms 全为有限值',
              bool(np.all(np.isfinite(data['icp_ms'][data['row_type'] == 0]))))
        stats = azt.stat_line(data['build_ms'])
        check('-1 未进入统计（build_ms 统计里没有负值）', '-' not in stats, stats)

        # 用例 8: 帧间间隔
        expect = np.diff(data['t_wall_ms'])
        got = data['interval_ms'][1:]
        check('interval_ms == t_wall_ms 逐差', np.allclose(expect, got),
              '最大偏差 %.6g' % np.max(np.abs(expect - got)))
        check('gap_ms == interval - frame_total 且非负',
              bool(np.all(data['gap_ms'][1:] >= -1e-9)))

        # 用例 2: 加法自洽
        warns = azt.check_data(data)
        check('合法数据无自洽告警', warns == [], '; '.join(warns))

        # 派生指标
        check('净ICP == icp - knn - solve',
              bool(np.allclose(data['icp_net_ms'][data['row_type'] == 0],
                               (data['icp_ms'] - data['knn_ms'] -
                                data['solve_ms'])[data['row_type'] == 0])))
        proc = run_cli(d1, os.path.join(d1, 'out'))
        check('CLI(正常数据) 退出码 0', proc.returncode == 0,
              'rc=%d stderr=%s' % (proc.returncode, proc.stderr[-400:]))
        check('CLI 无 Traceback', 'Traceback' not in proc.stderr)
        for name in ['page1.png', 'page2.png', 'page3.png', 'page4.png', 'page5.png',
                     'frame_time_summary.csv', 'pass_summary.csv']:
            check('已导出 %s' % name,
                  os.path.exists(os.path.join(d1, 'out', name)))
        # 各页面必须各自独立的 axes（matplotlib 参数相同时会复用实例，导致多页画在一起）
        import hashlib
        digests = []
        for name in ['page1.png', 'page2.png', 'page3.png', 'page4.png', 'page5.png']:
            with open(os.path.join(d1, 'out', name), 'rb') as fh:
                digests.append(hashlib.md5(fh.read()).hexdigest())
        check('五页图两两不同（无 axes 复用导致的页面重叠）',
              len(set(digests)) == 5,
              '%d 个不同摘要: %s' % (len(set(digests)), digests))
        check('终端摘要含每轮汇总', '每轮汇总' in proc.stdout)
        csv = open(os.path.join(d1, 'out', 'pass_summary.csv')).read().splitlines()
        check('pass_summary.csv 为 3 轮 + 1 表头', len(csv) == PASSES + 1,
              '实际 %d 行' % len(csv))

        # --pass 过滤
        proc_p = run_cli(d1, os.path.join(d1, 'out_pass1'), extra=['--pass', '1'])
        check('--pass 1 退出码 0', proc_p.returncode == 0,
              'stderr=%s' % proc_p.stderr[-400:])

        # 用例 7: 缺 iter 日志 -> 降级但不失败
        print('用例 7: 缺可选日志')
        d7 = new_dir(root, 'no_iter')
        write_main_log(os.path.join(d7, MAIN_LOG))
        proc7 = run_cli(d7, os.path.join(d7, 'out'))
        check('缺 iter 日志仍退出码 0', proc7.returncode == 0,
              'rc=%d stderr=%s' % (proc7.returncode, proc7.stderr[-400:]))
        check('缺 iter 日志有中文提示', '未找到日志' in proc7.stdout, proc7.stdout[-300:])
        data7, _ = azt.load_frame_time(os.path.join(d7, MAIN_LOG))
        s7 = azt.pass_summary(data7, None)
        check('缺 iter 日志时 pass_total_ms 留空(NaN)',
              bool(np.all(~np.isfinite([s['pass_total_ms'] for s in s7]))))
        check('缺 iter 日志仍出图', os.path.exists(os.path.join(d7, 'out', 'page5.png')))

        # 用例 5: 父子口径越界 -> 只告警
        print('用例 5: knn+solve > icp 越界')
        d5 = new_dir(root, 'bad_icp')
        write_main_log(os.path.join(d5, MAIN_LOG), bad_icp=True)
        data5, _ = azt.load_frame_time(os.path.join(d5, MAIN_LOG))
        warns5 = azt.check_data(data5)
        check('越界被检出并告警', len(warns5) >= 1 and 'knn+solve' in ' '.join(warns5),
              str(warns5))
        proc5 = run_cli(d5, os.path.join(d5, 'out'))
        check('越界数据不崩溃（无 Traceback）', 'Traceback' not in proc5.stderr,
              proc5.stderr[-400:])
        check('越界数据仍出图', os.path.exists(os.path.join(d5, 'out', 'page1.png')))
        check('越界数据退出码非 0（明示告警）', proc5.returncode != 0,
              'rc=%d' % proc5.returncode)

        # 残差语义: 小正残差正常(未埋点代码), 大残差告警, 负残差告警(区间重叠)
        def tweak_total(delta, tag):
            # 文件第 1 行为表头 -> 文件行号 = 数据下标 + 1
            path = new_dir(root, tag)
            rows = open(os.path.join(d1, MAIN_LOG)).read().splitlines()
            line = 11                      # 即数据下标 10
            toks = rows[line].split()
            toks[azt.IDX['frame_total_ms']] = '%.6f' % (
                float(toks[azt.IDX['frame_total_ms']]) + delta)
            rows[line] = ' '.join(toks)
            open(os.path.join(path, MAIN_LOG), 'w').write('\n'.join(rows) + '\n')
            data_x, _ = azt.load_frame_time(os.path.join(path, MAIN_LOG))
            return data_x, azt.check_data(data_x)

        data_small, warns_small = tweak_total(0.2, 'res_small')
        check('小正残差(0.2ms)不告警（未埋点代码属正常）', warns_small == [],
              str(warns_small))
        check('uncovered_ms == frame_total - Σ六段',
              np.allclose(data_small['uncovered_ms'][10], 0.2))
        _, warns_big = tweak_total(10.0, 'res_big')
        check('大残差被检出', any('未埋点残差' in w for w in warns_big), str(warns_big))
        _, warns_neg = tweak_total(-0.5, 'res_neg')
        check('负残差(区间重叠)被检出',
              any('重叠或重复累计' in w for w in warns_neg), str(warns_neg))

        # 用例 9: 采集在末轮中途结束 —— iter 日志缺被截断的那一轮
        # 实机复现: frame_time 有第 4 轮(仅 1 帧), iter_time 里没有该轮的行,
        # 曾导致 console_summary KeyError: 'pass_total_ms' 崩溃
        print('用例 9: iter 日志缺末轮（采集截断）')
        d9 = new_dir(root, 'truncated_iter')
        write_main_log(os.path.join(d9, MAIN_LOG))
        write_iter_log(os.path.join(d9, ITER_LOG))
        rows9 = open(os.path.join(d9, ITER_LOG)).read().splitlines()
        open(os.path.join(d9, ITER_LOG), 'w').write(
            '\n'.join(rows9[:1 + (PASSES - 1)]) + '\n')   # 砍掉末轮那一行
        data9, _ = azt.load_frame_time(os.path.join(d9, MAIN_LOG))
        iter9, _ = azt.load_iter_time(os.path.join(d9, ITER_LOG))
        s9 = azt.pass_summary(data9, iter9)          # 旧代码在此 KeyError
        check('缺末轮 iter 行时仍返回全部 %d 轮' % PASSES, len(s9) == PASSES,
              '实际 %d' % len(s9))
        check('缺行的末轮 pass_total_ms/delta_g 留空、converged=-1',
              (not np.isfinite(s9[-1]['pass_total_ms']) and
               not np.isfinite(s9[-1]['delta_g']) and s9[-1]['converged'] == -1),
              str({k: s9[-1][k] for k in ('pass_total_ms', 'delta_g', 'converged')}))
        check('缺行末轮的逐帧合计仍正常算出（帧数据本身没缺）',
              s9[-1]['rows'] > 0 and np.isfinite(s9[-1][azt.SEGMENTS[0]]))
        check('前 %d 轮仍取到 iter 值（未被末轮缺失带偏）' % (PASSES - 1),
              all(np.isfinite(s9[p]['pass_total_ms']) for p in range(PASSES - 1)),
              str([s9[p]['pass_total_ms'] for p in range(PASSES - 1)]))
        proc9 = run_cli(d9, os.path.join(d9, 'out'))
        check('CLI(缺末轮 iter 行) 无 Traceback', 'Traceback' not in proc9.stderr,
              proc9.stderr[-400:])
        check('CLI(缺末轮 iter 行) 退出码 0', proc9.returncode == 0,
              'rc=%d stderr=%s' % (proc9.returncode, proc9.stderr[-400:]))
        check('终端摘要仍打出全部 %d 轮' % PASSES,
              proc9.stdout.count('轮: 逐帧合计') == PASSES,
              '实际 %d' % proc9.stdout.count('轮: 逐帧合计'))

        # 用例 10: 首轮 delta_g=-1 是 C++ 哨兵(无收敛判定), 必须归一成 NaN
        # 它不是测出来的残差(delta_g=abs(Last_g-current_g) 恒 >= 0), 若当真实值用,
        # 终端会打 "Δg=-1.0000"、第 5 页会画出一根 -1 的假柱
        print('用例 10: 首轮 delta_g=-1 哨兵')
        iter10, _ = azt.load_iter_time(os.path.join(d1, ITER_LOG))   # write_iter_log 里 pass 0 即 -1
        check('pass 0 的 delta_g 哨兵已归一为 NaN',
              not np.isfinite(iter10['delta_g'][0]), str(iter10['delta_g'][0]))
        check('pass 0 的 converged 归一为 -1（终端渲染「未知」）',
              iter10['converged'][0] == -1, str(iter10['converged'][0]))
        check('相邻轮的真实 delta_g(0.25) 不被误伤',
              np.isfinite(iter10['delta_g'][1]) and
              abs(iter10['delta_g'][1] - 0.25) < 1e-9, str(iter10['delta_g'][1]))
        check('后续轮的真实 delta_g 全部保留',
              np.all(np.isfinite(iter10['delta_g'][1:])))
        check('converged=1 的末轮未被归一误改',
              iter10['converged'][-1] == 1, str(iter10['converged'][-1]))
        s10 = azt.pass_summary(data, iter10)
        check('pass 0 汇总里 delta_g 为「无数据」而非 -1.0',
              not np.isfinite(s10[0]['delta_g']) and s10[0]['converged'] == -1,
              str({k: s10[0][k] for k in ('delta_g', 'converged')}))
        check('pass 0 的 solved/pass_total 不受哨兵影响',
              np.isfinite(s10[0]['pass_total_ms']))
        csv10 = open(os.path.join(d1, 'out', 'pass_summary.csv')).read().splitlines()
        check('CSV 里哨兵轮 delta_g 记为空(nan)而非 -1.000000',
              'nan' in csv10[1].split(), csv10[1])

        # 用例 11: LIO 阶段(初始化完成后)状态更新完成间隔 —— 页面6
        # 场景为积压排空: 完成间隔 50ms > 输入周期 33.3ms, 净耗时 42ms, 空档 8ms
        print('用例 11: LIO 日志(12列含墙钟+点数)')
        write_lio_log(os.path.join(d1, LIO_LOG))
        lio11, err11 = azt.load_lio_time(os.path.join(d1, LIO_LOG))
        check('LIO 日志解析成功', lio11 is not None, err11)
        check('识别出 t_wall_ms 新列', lio11['_has_wall'])
        check('识别出点数 3 列(12列)', lio11['_has_pts'])
        check('点数列数值正确(有效1000/降采样2400/原始6000)',
              np.allclose(lio11['effect_feat_num'], 1000) and
              np.allclose(lio11['feats_down_size'], 2400) and
              np.allclose(lio11['feats_full_size'], 6000))
        check('完成间隔 == t_wall_ms 逐差',
              np.allclose(lio11['interval_ms'][1:], np.diff(lio11['t_wall_ms'])))
        check('空档 == 完成间隔 - 帧内净耗时',
              np.allclose(lio11['gap_ms'][1:],
                          lio11['interval_ms'][1:] - lio11['frame_total_ms'][1:]))
        check('数据时间差 == time 逐差×1000',
              np.allclose(lio11['data_interval_ms'][1:], np.diff(lio11['time']) * 1000))
        check('icp_net = icp - knn - solve ≥ 0',
              bool(np.all(lio11['icp_net_ms'] >= -1e-9)))
        win11, meta11 = azt.lio_window(lio11, 10.0)
        check('10s 墙钟窗口截取 201/250 行（50ms 间隔×10s）',
              meta11['rows'] == 201 and meta11['mode'] == 'wall',
              '实际 %s/%s mode=%s' % (meta11['rows'], meta11['total_rows'], meta11['mode']))
        proc11 = run_cli(d1, os.path.join(d1, 'out11'))
        check('CLI(LIO 日志) 退出码 0', proc11.returncode == 0,
              'rc=%d stderr=%s' % (proc11.returncode, proc11.stderr[-400:]))
        check('CLI(LIO 日志) 无 Traceback', 'Traceback' not in proc11.stderr)
        check('已导出 page6.png',
              os.path.exists(os.path.join(d1, 'out11', 'page6.png')))
        check('已导出 lio_window_frames.csv',
              os.path.exists(os.path.join(d1, 'out11', 'lio_window_frames.csv')))
        check('已导出 lio_window_summary.csv',
              os.path.exists(os.path.join(d1, 'out11', 'lio_window_summary.csv')))
        frames_csv = open(os.path.join(d1, 'out11', 'lio_window_frames.csv')).read().splitlines()
        check('lio_window_frames.csv = 2 表头 + 201 行', len(frames_csv) == 203,
              '实际 %d 行' % len(frames_csv))
        check('终端含完成间隔摘要', '完成间隔' in proc11.stdout)
        check('终端含吞吐判定文案（50ms>33.3ms 跟不上）', '跟不上输入' in proc11.stdout,
              proc11.stdout[-400:])
        check('终端含点数漏斗摘要（原始→降采样→有效）', '点数漏斗均值' in proc11.stdout,
              proc11.stdout[-400:])
        check('lio_window_frames.csv 表头含点数列', 'effect_feat_num' in frames_csv[1],
              frames_csv[1])
        check('lio_window_frames.csv 含点数数值 6000',
              '6000.000000' in frames_csv[2], frames_csv[2])
        import hashlib
        digests6 = []
        for name in ['page1.png', 'page2.png', 'page3.png', 'page4.png', 'page5.png',
                     'page6.png']:
            with open(os.path.join(d1, 'out11', name), 'rb') as fh:
                digests6.append(hashlib.md5(fh.read()).hexdigest())
        check('六页图两两不同（页面6 未与其他页重叠）', len(set(digests6)) == 6,
              '%d 个不同摘要' % len(set(digests6)))

        # 用例 12: 旧版 8 列日志（无 t_wall_ms）——完成间隔不可算, 明示降级
        print('用例 12: LIO 日志(旧版8列, 无墙钟)')
        d12 = new_dir(root, 'lio_legacy')
        write_main_log(os.path.join(d12, MAIN_LOG))
        write_lio_log(os.path.join(d12, LIO_LOG), has_wall=False, has_pts=False)
        lio12, _ = azt.load_lio_time(os.path.join(d12, LIO_LOG))
        check('旧版识别为无墙钟', not lio12['_has_wall'])
        check('旧版识别为无点数列', not lio12['_has_pts'])
        check('旧版完成间隔全 NaN（不伪造）',
              bool(np.all(~np.isfinite(lio12['interval_ms']))))
        win12, meta12 = azt.lio_window(lio12, 10.0)
        check('旧版按数据时间截窗口（33.3ms×10s 全覆盖250行）',
              meta12['rows'] == 250 and meta12['mode'] == 'data',
              '实际 %s/%s mode=%s' % (meta12['rows'], meta12['total_rows'], meta12['mode']))
        proc12 = run_cli(d12, os.path.join(d12, 'out'))
        check('CLI(旧版LIO) 退出码 0', proc12.returncode == 0,
              'rc=%d stderr=%s' % (proc12.returncode, proc12.stderr[-400:]))
        check('终端提示旧版日志需重新采集', '无 t_wall_ms' in proc12.stdout,
              proc12.stdout[-400:])
        check('终端提示旧版日志无点数列', '无点数列' in proc12.stdout,
              proc12.stdout[-400:])

        # 用例 14: 9 列中间版（仅墙钟, 无点数列）——页面6 中图降级提示
        print('用例 14: LIO 日志(9列中间版, 仅墙钟无点数)')
        d14 = new_dir(root, 'lio_mid')
        write_main_log(os.path.join(d14, MAIN_LOG))
        write_lio_log(os.path.join(d14, LIO_LOG), has_pts=False)
        lio14, err14 = azt.load_lio_time(os.path.join(d14, LIO_LOG))
        check('中间版识别为有墙钟无点数',
              lio14 is not None and lio14['_has_wall'] and not lio14['_has_pts'])
        check('中间版点数列全 NaN（不伪造）',
              bool(np.all(~np.isfinite(lio14['feats_full_size']))) and
              bool(np.all(~np.isfinite(lio14['effect_feat_num']))))
        proc14 = run_cli(d14, os.path.join(d14, 'out'))
        check('CLI(9列中间版) 退出码 0', proc14.returncode == 0,
              'rc=%d stderr=%s' % (proc14.returncode, proc14.stderr[-400:]))
        check('终端提示中间版日志无点数列', '无点数列' in proc14.stdout,
              proc14.stdout[-400:])
        check('9列中间版仍导出 page6.png（中图降级）',
              os.path.exists(os.path.join(d14, 'out', 'page6.png')))

        # 用例 13: 缺 LIO 日志（初始化未完成/采集未覆盖）——页面6 空数据页
        print('用例 13: 缺 LIO 日志')
        proc13 = run_cli(d7, os.path.join(d7, 'out13'))
        check('缺 LIO 日志退出码 0', proc13.returncode == 0,
              'rc=%d stderr=%s' % (proc13.returncode, proc13.stderr[-400:]))
        check('缺 LIO 日志有中文提示', 'Li_D_init_lio_time' in proc13.stdout,
              proc13.stdout[-400:])
        check('缺 LIO 日志仍导出 page6.png（空数据页）',
              os.path.exists(os.path.join(d7, 'out13', 'page6.png')))

        # 用例 6: 主日志缺失
        print('用例 6: 主日志缺失')
        d6 = new_dir(root, 'missing')
        proc6 = run_cli(d6, os.path.join(d6, 'out'))
        check('主日志缺失 -> 退出码非 0', proc6.returncode != 0, 'rc=%d' % proc6.returncode)
        check('主日志缺失 -> 无 Traceback', 'Traceback' not in proc6.stderr,
              proc6.stderr[-400:])
        check('主日志缺失 -> 中文提示', '未找到日志' in proc6.stdout, proc6.stdout[-300:])

        # 空日志
        d0 = new_dir(root, 'empty')
        open(os.path.join(d0, MAIN_LOG), 'w').close()
        proc_e = run_cli(d0, os.path.join(d0, 'out'))
        check('空主日志 -> 退出码非 0 且提示为空', proc_e.returncode != 0 and
              '日志为空' in proc_e.stdout, proc_e.stdout[-300:])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print('')
    if _failures:
        print('结果: %d 项失败 -> %s' % (len(_failures), ', '.join(_failures)))
        return 1
    print('结果: 全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
