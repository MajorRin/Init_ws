#!/usr/bin/env python3
"""运行: python3 -m unittest discover -s scripts/init_comparison -p test_tuning_comparison.py"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tuning_test_')
        self.root = Path(self.temp.name)
        self.before = self.root / 'before'
        self.after = self.root / 'after'
        self.make_run(self.before, 4.0, 100.0)
        self.make_run(self.after, 2.0, 100.2)
        self.rtk = self.before / 'rtk_velocity.csv'
        self.rtk.write_text('t,vx,vy,vz,speed\n' + ''.join(
            '%.3f,2,0,0,2\n' % (100 + k * 0.1) for k in range(31)))

    def tearDown(self):
        self.temp.cleanup()

    def make_run(self, path, speed, start, side='A'):
        path.mkdir(exist_ok=True)
        (path / (side + '_events.log')).write_text(
            'init_done %.3f max_iter 0 0 0 0 0 0 0 0 -9.81 0 0 0\n' % start)
        flags = '1 1 1 1 1' if side == 'A' else '1 1 1 1'
        rows = ['%.3f post %s 500 %.3f 0 0 0 0 0 0 0 -9.81\n' %
                (start + k * 0.1, flags, speed) for k in range(21)]
        # 非相邻重复时间戳的旧状态，不得进入最终统计。
        old = '%.3f post %s 500 99 0 0 0 0 0 0 0 -9.81\n' % (start + 0.5, flags)
        (path / (side + '_states.log')).write_text(old + ''.join(rows))

    def command(self, output, extra=()):
        return [sys.executable, str(HERE / 'analyze_tuning_comparison.py'),
                '--baseline-run', str(self.before), '--candidate-run', str(self.after),
                '--output-dir', str(output), '--no-plot'] + list(extra)

    def execute(self, command):
        return subprocess.run(command, capture_output=True, text=True, timeout=60)

    def test_metrics_and_readonly(self):
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for d in (self.before, self.after) for p in d.iterdir()}
        out = self.root / 'analysis'
        result = self.execute(self.command(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((out / 'summary.json').read_text())
        self.assertAlmostEqual(summary['before']['speed_rmse'], 2)
        self.assertAlmostEqual(summary['after']['speed_rmse'], 0)
        self.assertAlmostEqual(summary['speed_rmse_improvement_percent'], 100)
        self.assertEqual(summary['before']['reason'], 'max_iter')
        self.assertGreaterEqual(summary['common']['start'], 100.2)
        for p, digest in before.items():
            self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(), digest)
        self.assertNotEqual(self.execute(self.command(out)).returncode, 0)

    def test_missing_rtk_and_no_overlap(self):
        self.assertNotEqual(self.execute(self.command(self.root / 'missing',
                            ['--rtk', str(self.root / 'absent.csv')])).returncode, 0)
        self.make_run(self.after, 2, 200)
        result = self.execute(self.command(self.root / 'no_overlap'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('共同时间窗口', result.stderr)

    def test_sparse_frames_and_invalid_final_record(self):
        states = self.after / 'A_states.log'
        # After 为5Hz并偏移20ms；RTK为10Hz，窄容差应排除未同时配对样本。
        rows = ['%.3f post 1 1 1 1 1 500 2 0 0 0 0 0 0 0 -9.81\n' %
                (100.22 + k * 0.2) for k in range(10)]
        states.write_text(''.join(rows) +
                          '100.620 invalid 0 0 0 1 1 500 99 0 0 0 0 0 0 0 -9.81\n')
        out = self.root / 'sparse'
        result = self.execute(self.command(out, ['--tol', '0.03']))
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((out / 'summary.json').read_text())
        self.assertLess(summary['common']['paired_fraction'], 1)
        self.assertAlmostEqual(summary['after']['speed_rmse'], 0)
        self.assertLessEqual(summary['after']['max_time_delta'], 0.03)
        self.assertEqual(summary['after']['valid_post_frames'], 9)
        result = self.execute(self.command(self.root / 'too_strict', ['--tol', '0.001']))
        self.assertNotEqual(result.returncode, 0)

    def test_static_format(self):
        self.make_run(self.before, 4, 100, 'B')
        self.make_run(self.after, 2, 100.2, 'B')
        result = self.execute(self.command(self.root / 'static', ['--side', 'B']))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_components_and_plot_output(self):
        out = self.root / 'plots'
        command = [arg for arg in self.command(out) if arg != '--no-plot']
        result = self.execute(command)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ('tuning_comparison.png', 'velocity_components.png', 'gravity_components.png'):
            self.assertEqual((out / name).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        with (out / 'paired.csv').open() as f:
            row = next(csv.DictReader(f))
        for key, value in {'before_vx': 4, 'after_vx': 2, 'rtk_vx': 2,
                           'before_vy': 0, 'after_vz': 0, 'rtk_vz': 0,
                           'before_gz': -9.81, 'after_gz': -9.81}.items():
            self.assertAlmostEqual(float(row[key]), value)
        self.assertEqual((out / 'rtk_velocity.csv').read_bytes(), self.rtk.read_bytes())

    def test_historical_rtk_reference_and_chain(self):
        # 复现旧 tuning 目录：没有 RTK 副本，只有报告记录的来源。
        analysis = self.after / 'analysis'
        analysis.mkdir()
        report = analysis / 'summary.json'
        report.write_text(json.dumps({'rtk': str(self.rtk)}))
        base = ['bash', str(HERE / 'run_init_comparison.sh'), '--mode', 'tuning',
                '--baseline-run', str(self.after), '--candidate-run', str(self.before), '--no-plot']
        out = self.root / 'resolved'
        result = self.execute(base + ['--output-dir', str(out)])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((out / 'rtk_velocity.csv').read_bytes(), self.rtk.read_bytes())
        self.assertFalse((self.after / 'rtk_velocity.csv').exists())
        self.assertFalse((out / 'A_node.pid').exists())
        bad = self.execute(base + ['--rtk', str(self.root / 'missing.csv'),
                                   '--output-dir', str(self.root / 'explicit_bad')])
        self.assertNotEqual(bad.returncode, 0)  # 显式错误不能被自动回退掩盖。
        # 模拟新回放目录已有状态：上游引用失效后仍可独立作为基线。
        self.make_run(out, 2, 100.2)
        report.write_text(json.dumps({'rtk': '/absent/old-source.csv'}))
        self.rtk.write_text('')
        command = self.command(self.root / 'chain')
        command[command.index('--baseline-run') + 1] = str(out)
        result = self.execute(command)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # 只有 analysis 副本的直接 Python 输出也能恢复来源。
        from analyze_tuning_comparison import resolve_rtk
        self.assertEqual(resolve_rtk(out / 'analysis'), (out / 'analysis' / 'rtk_velocity.csv').resolve())
        self.assertNotEqual(self.execute(base + ['--output-dir', str(self.root / 'stale')]).returncode, 0)

    def test_shell_offline_and_error_propagation(self):
        base = ['bash', str(HERE / 'run_init_comparison.sh'), '--mode', 'tuning',
                '--baseline-run', str(self.before), '--candidate-run', str(self.after),
                '--bag', '/nonexistent/offline.bag', '--no-plot']
        out = self.root / 'shell_output'
        result = self.execute(base + ['--output-dir', str(out)])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((out / 'analysis' / 'summary.json').is_file())
        self.assertFalse((out / 'A_node.pid').exists())
        self.make_run(self.after, 2, 200)
        failed = self.execute(base + ['--output-dir', str(self.root / 'bad_output')])
        self.assertNotEqual(failed.returncode, 0, failed.stdout)
        for extra in (['--window', '-1'], ['--tol', 'nan'], ['--offset']):
            result = self.execute(base + extra)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()