#!/usr/bin/env python3
"""同一算法两次实验与 RTK 的离线对比；历史输入只读，不依赖 ROS。"""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path
import sys

import numpy as np
from analyze_init_comparison import parse_events, parse_states, parse_rtk, nearest_rtk


def resolve_rtk(directory, explicit=None):
    """显式路径优先；只追溯本次基线记录的来源，不猜测其他实验。"""
    base = Path(directory).resolve()
    candidates = [Path(explicit)] if explicit else [
        base / 'rtk_velocity.csv', base / 'analysis' / 'rtk_velocity.csv']
    if not explicit:
        summary_path = base / 'analysis' / 'summary.json'
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text())
            except (OSError, ValueError):
                summary = {}
            source = summary.get('rtk') if isinstance(summary, dict) else None
            if isinstance(source, str) and source:
                source = Path(source)
                # 历史版本记录绝对路径；相对路径按报告所在目录解释。
                candidates.append(source if source.is_absolute() else summary_path.parent / source)
    for path in candidates:
        if path.is_file() and path.stat().st_size:
            return path.resolve()
    raise ValueError('缺少 RTK，已检查: %s；请使用 --rtk 指定同一数据包的 CSV' %
                     ', '.join(str(p) for p in candidates))


def load_run(directory, side):
    events = parse_events(os.path.join(directory, side + '_events.log'))
    states = parse_states(os.path.join(directory, side + '_states.log'))
    event = events['init_done'] or events['first_post']
    if states is None or event is None:
        raise ValueError('%s 缺少有效状态或初始化完成事件' % directory)
    start = (events['first_post'] or event)['t']
    if not np.isfinite(start) or not np.isfinite(event['t']):
        raise ValueError('%s 初始化事件时间非有限值' % directory)
    # 全局去重：迭代重放可能产生非相邻的重复时间戳，始终取最终记录。
    last = {float(t): i for i, t in enumerate(states['t'])
            if np.isfinite(t) and t >= max(start, event['t'])}
    indices = [last[t] for t in sorted(last)]
    good = [i for i in indices if states['rtype'][i] == 'post'
            and np.all(np.isfinite(np.r_[states['vel'][i], states['bg'][i], states['grav'][i]]))
            and states['flags'][i][3] == 1]
    if not good:
        raise ValueError('%s 初始化后无有效 post 状态' % directory)
    series = {k: states[k][good] for k in ('t', 'vel', 'bg', 'grav', 'feats')}
    series['speed'] = np.linalg.norm(series['vel'], axis=1)
    series['gmag'] = np.linalg.norm(series['grav'], axis=1)
    return series, {'directory': str(Path(directory).resolve()), 'side': side,
                    'init_time': float(event['t']),
                    'reason': event.get('reason', 'first_post_only'),
                    'valid_post_frames': len(good),
                    'excluded_or_duplicate_frames': len(states['t']) - len(good)}


def compare(before, after, rtk, window, tol):
    start = max(before['t'][0], after['t'][0], rtk['t'][0])
    end = min(start + window, before['t'][-1], after['t'][-1], rtk['t'][-1])
    if end <= start:
        raise ValueError('两次实验与 RTK 没有共同时间窗口；请确认同一数据包和回放段')
    # 使用唯一 RTK 时间戳作共同采样网格，而非两侧独立帧数加权。
    grid = np.flatnonzero((rtk['t'] >= start) & (rtk['t'] <= end))
    pairs = []
    for j in grid:
        t = rtk['t'][j]
        ia, da = nearest_rtk(t, before, tol)
        ib, db = nearest_rtk(t, after, tol)
        if ia is not None and ib is not None:
            pairs.append((j, ia, ib, da, db))
    if not pairs:
        raise ValueError('共同窗口内没有同时满足时间容差的 RTK 配对样本')
    return pairs, dict(start=float(start), end=float(end), duration=float(end - start),
                       requested_window=window, rtk_samples=len(grid), paired_samples=len(pairs),
                       paired_fraction=len(pairs) / len(grid), tolerance=tol)


def metrics(series, indices, reference, dt):
    error = series['speed'][indices] - reference
    return dict(speed_rmse=float(np.sqrt(np.mean(error ** 2))),
                speed_mae=float(np.mean(np.abs(error))),
                speed_error_mean=float(np.mean(error)),
                speed_error_std=float(np.std(error)),
                gravity_norm_mean=float(np.mean(series['gmag'][indices])),
                gravity_norm_std=float(np.std(series['gmag'][indices])),
                gyro_bias_mean=np.mean(series['bg'][indices], axis=0).tolist(),
                gyro_bias_std=np.std(series['bg'][indices], axis=0).tolist(),
                feature_count_median=float(np.median(series['feats'][indices])),
                unique_state_samples=len(set(indices)), max_time_delta=float(max(dt)))


def plot_result(out, times, before, after, ia, ib, reference, start, rtk_velocity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    x = times - start
    for label, data, idx in (('Before', before, ia), ('After', after, ib)):
        axes[0].plot(x, data['speed'][idx], label=label)
        axes[1].plot(x, data['speed'][idx] - reference, label=label)
        axes[2].plot(x, data['gmag'][idx], label=label)
        axes[3].plot(x, data['feats'][idx], label=label)
    axes[0].plot(x, reference, 'k--', label='RTK reference')
    axes[1].axhline(0, color='k', linestyle='--')
    axes[2].axhline(9.81, color='k', linestyle='--', label='Nominal 9.81 (not direction truth)')
    for ax, title in zip(axes, ('Speed (m/s)', 'Speed norm error (m/s)',
                                'Gravity norm (m/s^2)', 'Feature count (same algorithm)')):
        ax.set_ylabel(title)
        ax.grid(True)
        ax.legend()
    axes[-1].set_xlabel('Seconds from common window start')
    fig.suptitle('Parameter tuning: before / after / RTK (common RTK timestamps)')
    fig.tight_layout()
    fig.savefig(out / 'tuning_comparison.png', dpi=140)
    plt.close(fig)

    # 分列展示原生坐标系，避免把不同世界系的分量差误解为真值误差。
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    columns = [('Before: native world frame', before['vel'][ia]),
               ('After: native world frame', after['vel'][ib]),
               ('RTK: native reference frame', rtk_velocity)]
    for col, (title, values) in enumerate(columns):
        axes[0, col].set_title(title)
        for dim, axis in enumerate('xyz'):
            ax = axes[dim, col]
            ax.plot(x, values[:, dim])
            ax.set_ylabel('v_%s (m/s)' % axis)
            ax.grid(True)
        axes[-1, col].set_xlabel('Seconds from common window start')
    fig.suptitle('Velocity components: frames NOT aligned; component differences are NOT errors')
    fig.tight_layout()
    fig.savefig(out / 'velocity_components.png', dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for col, (label, data, idx) in enumerate((('Before', before, ia), ('After', after, ib))):
        axes[0, col].set_title(label + ': native world frame')
        for dim, axis in enumerate('xyz'):
            ax = axes[dim, col]
            ax.plot(x, data['grav'][idx, dim])
            ax.set_ylabel('g_%s (m/s^2)' % axis)
            ax.grid(True)
        axes[-1, col].set_xlabel('Seconds from common window start')
    fig.suptitle('Estimated gravity components: frames NOT aligned; no RTK gravity truth')
    fig.tight_layout()
    fig.savefig(out / 'gravity_components.png', dpi=140)
    plt.close(fig)


def positive(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('必须为有限正数')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', required=True)
    parser.add_argument('--candidate-run', required=True)
    parser.add_argument('--side', choices=('A', 'B'), default='A')
    parser.add_argument('--rtk', help='默认查找 baseline 的 RTK 副本或历史报告记录的来源')
    parser.add_argument('--output-dir', required=True, help='必须不存在，避免覆盖已有结果')
    parser.add_argument('--window', type=positive, default=10.0)
    parser.add_argument('--tol', type=positive, default=0.05)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, IndexError, ImportError) as exc:
        print('[tuning] 错误: %s' % exc, file=sys.stderr)
        return 1
    return 0


def run(args):
    before, meta_a = load_run(args.baseline_run, args.side)
    after, meta_b = load_run(args.candidate_run, args.side)
    rtk_path = str(resolve_rtk(args.baseline_run, args.rtk))
    rtk = parse_rtk(rtk_path)
    if rtk is None:
        raise ValueError('缺少有效 RTK 数据: ' + rtk_path)
    finite = np.isfinite(rtk['t']) & np.all(np.isfinite(rtk['v']), axis=1)
    rtk = {k: v[finite] for k, v in rtk.items()}
    _, unique = np.unique(rtk['t'], return_index=True)
    rtk = {k: v[unique] for k, v in rtk.items()}
    if not len(rtk['t']):
        raise ValueError('RTK 无有限样本')
    pairs, common = compare(before, after, rtk, args.window, args.tol)
    jr, ia, ib = (np.array([p[k] for p in pairs], dtype=int) for k in range(3))
    da, db = (np.array([p[k] for p in pairs]) for k in (3, 4))
    reference = np.linalg.norm(rtk['v'][jr], axis=1)
    ma = metrics(before, ia, reference, da)
    mb = metrics(after, ib, reference, db)
    improvement = ((ma['speed_rmse'] - mb['speed_rmse']) / ma['speed_rmse'] * 100
                   if ma['speed_rmse'] > 0 else None)
    warnings = ['必须使用同一 bag、相同时间基准和对应回放段；历史日志无法独立证明数据来源一致。',
                '仅使用初始化后的有效 post 帧；全局重复时间戳保留最后记录。',
                '共同 RTK 时刻分别最近邻匹配两侧，可能重复使用状态帧，不插值、不外推。',
                '速度模长为主指标；未对齐世界系，不计算跨实验三轴速度或重力方向误差。',
                '分量图及 CSV 使用各自原生坐标系，仅展示分量演化；RTK 不提供重力方向真值。',
                'max_iter 表示达到上限，不表示收敛；重力模长接近9.81也不证明方向正确。',
                '不从状态日志推算计算耗时；初始化时间为传感器时间，不是墙钟耗时。']
    if Path(args.baseline_run).resolve() == Path(args.candidate_run).resolve():
        warnings.append('两侧使用相同目录：仅为自对比校验，不是调参收益。')
    if common['duration'] < args.window - 1e-6:
        warnings.append('可用共同窗口不足请求时长，已截短。')
    if common['paired_fraction'] < 0.8:
        warnings.append('共同配对覆盖率不足80%，误差统计可能有选择偏差。')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(rtk_path, out / 'rtk_velocity.csv')
    summary = dict(before=dict(meta_a, **ma), after=dict(meta_b, **mb), common=common,
                   rtk=str((out / 'rtk_velocity.csv').resolve()), rtk_source=rtk_path,
                   speed_rmse_improvement_percent=improvement,
                   warnings=warnings)
    with (out / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'before', 'after'])
        for key in ma:
            writer.writerow([key, ma[key], mb[key]])
    with (out / 'paired.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rtk_t', 'before_t', 'after_t', 'before_dt', 'after_dt',
                         'rtk_speed', 'before_speed', 'after_speed', 'before_error', 'after_error',
                         'before_gmag', 'after_gmag', 'before_feats', 'after_feats'] +
                        [prefix + axis for prefix in ('before_v', 'after_v', 'rtk_v',
                                                      'before_g', 'after_g') for axis in 'xyz'])
        for k, (j, a, b, d1, d2) in enumerate(pairs):
            writer.writerow([rtk['t'][j], before['t'][a], after['t'][b], d1, d2, reference[k],
                             before['speed'][a], after['speed'][b],
                             before['speed'][a] - reference[k], after['speed'][b] - reference[k],
                             before['gmag'][a], after['gmag'][b], before['feats'][a], after['feats'][b]] +
                            before['vel'][a].tolist() + after['vel'][b].tolist() + rtk['v'][j].tolist() +
                            before['grav'][a].tolist() + after['grav'][b].tolist())
    lines = ['调参前后对比（%s）' % args.side,
             '调参前: %s；结束原因: %s' % (meta_a['directory'], meta_a['reason']),
             '调参后: %s；结束原因: %s' % (meta_b['directory'], meta_b['reason']),
             '共同窗口: %.9f 至 %.9f，配对 %d/%d' %
             (common['start'], common['end'], common['paired_samples'], common['rtk_samples']),
             '速度模长RMSE: %.6f -> %.6f m/s' % (ma['speed_rmse'], mb['speed_rmse']),
             'RMSE改善率(正为改善): %s' % ('N/A (原误差为0)' if improvement is None else '%.3f%%' % improvement)]
    lines += ['注意: ' + w for w in warnings]
    (out / 'comparison_report.txt').write_text('\n'.join(lines) + '\n')
    if not args.no_plot:
        plot_result(out, rtk['t'][jr], before, after, ia, ib, reference, common['start'], rtk['v'][jr])
    print('\n'.join(lines))
    print('[tuning] 输出目录: %s' % out)


if __name__ == '__main__':
    sys.exit(main())