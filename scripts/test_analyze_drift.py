"""Run: python3 -m unittest discover -s src/D-LI-Init/scripts -p test_analyze_drift.py -v"""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import analyze_drift as drift


def frame(number=1, time=11, **changes):
    data = {key: None for key in drift.SCALARS}
    data.update({key: 0 for key in drift.COUNTS})
    data.update({key: 1 for key in drift.FLAGS})
    data.update({key: [None] * size for key, size in drift.VECTORS.items()})
    data.update(schema=1, frame=number, time=time, stage="post_dynamic", status="post",
                reason="matched", points=100, effective=50, iterations=3, searches=2,
                position=[number, 0, 0], velocity=[1, 0, 0], quaternion_wxyz=[1, 0, 0, 0],
                gravity=[0, 0, -9.81], covariance_diagonal=list(range(24)))
    data.update(changes)
    return "I0920 12:00:00.000 123 file.cpp:1] [DLI_FRAME] " + json.dumps(data) + "\n"


def event(name="init_done", time=10):
    return "[DLI_EVENT] " + json.dumps(dict(schema=1, event=name, time=time, reason="test")) + "\n"


def s2_snapshot(number=1):
    return {"position": [number, 0, 0], "velocity": [1, 0, 0], "gravity": [0, 0, -9.81],
            "bias_g": [0, 0, 0], "bias_a": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0],
            "covariance_diagonal": list(range(24)), "covariance_finite": 1,
            "covariance_nonnegative": 1}


def frame2(number=1, time=11, **changes):
    data = dict(schema=2, run_id="rid", sequence=number, steady_ms=0.0, time=time,
                data_time=time, pass_id=0, batch_id=0, ros_time=time, lag_ms=0.0,
                lag_s=0.0, clock_mode="sim", lag_reason="valid", playback_rate=1.0,
                playback_state="playing", frame=number, cache_index=0,
                stage="post_dynamic", status="post", reason="matched", applied=1,
                committed=1, propagated=1, points=100, effective=50, iterations=3,
                searches=2, relative_time=time - 10, effective_ratio=0.5,
                raw_points=1072, undistorted_points=1072, map_points=100,
                publish_ms=1.0, build_ms=1.0, before_propagation=s2_snapshot(number),
                after_propagation=s2_snapshot(number), after_update=s2_snapshot(number),
                covariance_source="post", state_frame="world", residual_source="post",
                information_source="post", residual=0.1, condition=2.0, cov_lidar=0.01,
                frame_ms=10.0, imu_ms=1.0, down_ms=1.0, icp_ms=1.0, match_ms=1.0,
                solve_ms=1.0, map_ms=1.0, diagnostic_timing_scope="frame",
                rotation_correction_deg=0.0, position_correction=[0, 0, 0],
                position=[number, 0, 0], velocity=[1, 0, 0], gravity=[0, 0, -9.81],
                bias_g=[0, 0, 0], bias_a=[0, 0, 0], quaternion_wxyz=[1, 0, 0, 0],
                covariance_diagonal=list(range(24)),
                pose_information_eigenvalues=[0.0] * 6,
                pose_information_diagonal=[0.0] * 6, speed=1.0, gravity_norm=9.81,
                vertical_velocity=0.0, rtk_time=time, rtk_time_difference=0.0,
                rtk_found=1, rtk_valid=1, rtk_reason="ok", rtk_frame="enu",
                rtk_velocity=[1, 0, 0], rtk_speed=1.0, rtk_axis_error=0.0,
                rtk_alignment_reason="ok", diag_scope_ms=1.0)
    data.update(changes)
    return "[DLI_FRAME] " + json.dumps(data) + "\n"


def event2(name="init_done", time=10, **changes):
    data = dict(schema=2, run_id="rid", sequence=1, steady_ms=0.0, time=time,
                data_time=time, pass_id=0, batch_id=0, event=name, reason="test")
    data.update(changes)
    return "[DLI_EVENT] " + json.dumps(data) + "\n"


def batch2(stage, pass_id=0, batch_id=0, **changes):
    data = dict(schema=2, run_id="rid", sequence=1, steady_ms=0.0, time=10.0,
                data_time=10.0, pass_id=pass_id, batch_id=batch_id, stage=stage,
                refinement=0, window_start=0.0, window_end=10.0, window_size=6,
                cache_size=6, configured_length=60, estimate_acc_bias=1)
    if stage == "batch_end":
        data.update(refinement=None, initial_velocity=[1, 0, 0],
                    initial_velocity_time=0.0, initial_velocity_frame="world",
                    initial_speed=9.8, gravity=[0, 0, -9.81], gravity_norm=9.8,
                    gravity_frame="world", bias_g=[0, 0, 0], bias_a=[0, 0, 0],
                    bias_a_source="fixed_not_estimated", delta_v=0.1,
                    delta_v_reason="converged", velocity_threshold=0.5, finite=1,
                    observations=6, quality_reason="ok", velocity_frame="world",
                    velocity_transform_reason="identity", velocities=[],
                    batch_wall_ms=1.0, batch_algorithm_ms=1.0, observer_ms=1.0,
                    pass_wall_ms=1.0, timing_scope="pass",
                    serialization_ms_excludes_log_write=0.1)
    elif stage == "gyro_bias":
        data.update(decomposition_status=0, solve_status=0, solution_finite=1,
                    system_finite=1, normal_equation_relative_residual=0.0,
                    residual_reason="ok", solution=[0.0] * 6,
                    gravity=[0, 0, -9.81], gravity_reason="ok")
    else:
        data.update(decomposition_status=0, solve_status=0, solution_finite=1,
                    system_finite=1, normal_equation_relative_residual=0.0,
                    residual_reason="ok", solution=[0.0] * 9,
                    gravity=[0, 0, -9.81], gravity_reason="ok",
                    velocity_frame="world", velocity_transform_reason="identity",
                    velocities=[], gravity_tangent_increment=[0.0, 0.0])
    data.update(changes)
    return "[DLI_BATCH] " + json.dumps(data) + "\n"


def queue2(**changes):
    data = dict(schema=2, run_id="rid", sequence=1, steady_ms=0.0, time=10.0,
                data_time=10.0, pass_id=0, batch_id=0, ros_time=10.0, lag_ms=0.0,
                lag_s=0.0, clock_mode="sim", lag_reason="valid", playback_rate=1.0,
                playback_state="playing", initialized=1, lidar_queue=3, imu_queue=4,
                latest_lidar=10.0, processed=9.0, latest_imu=9.9)
    data.update(changes)
    return "[DLI_QUEUE] " + json.dumps(data) + "\n"


def edit2(line, drop=None, **changes):
    head, _, payload = line.partition("] ")
    data = json.loads(payload)
    if drop:
        data.pop(drop, None)
    data.update(changes)
    return head + "] " + json.dumps(data) + "\n"


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.warning_context = warnings.catch_warnings(record=True)
        self.seen_warnings = self.warning_context.__enter__()
        warnings.simplefilter("always", RuntimeWarning)

    def tearDown(self):
        self.warning_context.__exit__(None, None, None)
        self.temp.cleanup()

    def log(self, text, name="run.INFO"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_strict_dirty_lines(self):
        dirty = ['[DLI_FRAME] {bad}\n', frame().replace('"schema": 1', '"schema": 2'),
                 frame().replace('"residual": null', '"residual": NaN'),
                 frame().replace('"schema": 1', '"schema": 1, "schema": 1'),
                 frame().replace('"velocity": [1, 0, 0]', '"velocity": [1]'),
                 event(time=None), event(time=0)]
        run = drift.load_run(self.log(event("startup", 0) + event() + ''.join(dirty) + frame()))
        self.assertEqual(run.counts["bad_lines"], len(dirty))
        self.assertEqual(len(run.frames), 1)
        self.assertTrue(run.frames[0]["valid_motion"])
        self.assertTrue(self.seen_warnings)

    def test_event_order_and_timestamp(self):
        run = drift.load_run(self.log(event("startup", 0) + frame(1, 9) + event() +
                                      frame(2, 9.5) + frame(3, 10) + frame(4, 11)))
        self.assertEqual([r["valid_motion"] for r in run.frames], [False, False, True, True])
        self.assertEqual(run.frames[-1]["dt"], 1)
        late = drift.load_run(self.log(frame(1, 11) + event(), "late.INFO"))
        self.assertFalse(late.frames[0]["valid_motion"])

    def test_missing_event_and_direct_lio(self):
        run = drift.load_run(self.log(frame()))
        self.assertFalse(run.frames[0]["valid_motion"])
        path = self.log(frame(stage="direct_lio"), "direct.INFO")
        self.assertFalse(drift.load_run(path).frames[0]["valid_motion"])
        direct = drift.load_run(path, include_direct=True)
        self.assertTrue(direct.frames[0]["valid_motion"])
        self.assertIn("no_init_done", direct.origin_kind)

    def test_rotation_symlink_multi_startup(self):
        self.log(event("startup", 0) + event() + frame(), "nested/run.INFO.1")
        latest = self.log(frame(2, 12), "nested/run.INFO")
        (self.root / "alias.INFO").symlink_to(latest)
        self.log(event("startup", 0), "ignored.WARNING")
        run = drift.load_run(self.root)
        self.assertEqual(len(run.files), 2)
        self.assertEqual([r["frame"] for r in run.frames], [1, 2])
        self.assertTrue(run.frames[1]["connected"])
        self.log(event("startup", 0) + frame(3, 13), "z.INFO")
        with self.assertRaisesRegex(ValueError, "多次 startup"):
            drift.load_run(self.root)

    def test_duplicates_conflicts_nonmonotonic(self):
        path = self.log(event() + frame() + frame() + frame(position=[99, 0, 0]) +
                        frame(2, 12) + frame(3, 12) + frame(4, 11.5) + frame(5, 13))
        run = drift.load_run(path)
        self.assertEqual(run.counts["duplicates"], 1)
        self.assertEqual(run.counts["conflicts"], 1)
        self.assertEqual(run.counts["invalid_or_nonmonotonic"], 2)
        self.assertFalse(run.frames[0]["valid_motion"])
        self.assertTrue(all(np.isnan(r["step_m"]) for r in run.frames))

    def test_no_posterior_null_and_gaps(self):
        rows = [frame(1, 11), frame(2, 12, status="skipped"), frame(3, 13),
                frame(4, 14, status="partial_applied", committed=0),
                frame(5, 15, status="predict", applied=0), frame(6, 16, effective=0),
                frame(7, None), frame(8, 18, position=[None, 0, 0]), frame(9, 19)]
        run = drift.load_run(self.log(event() + ''.join(rows)))
        self.assertTrue(all(np.isnan(r["step_m"]) for r in run.frames))
        x, y = drift.series(run, "position", True)
        self.assertTrue(np.isnan(x).any())
        self.assertTrue(np.isnan(y).any())
        only_predict = drift.load_run(self.log(event() + frame(status="predict"), "predict.INFO"))
        summary = drift.run_summary(only_predict)
        self.assertEqual(summary["valid_posteriors"], 0)
        self.assertIsNone(summary["connected_path_length_m"])

    def test_headless_comparison_cli_and_export(self):
        a = self.log(event("startup", 0) + event(time=10) + frame(1, 11) + frame(2, 12), "a.INFO")
        b = self.log(event("startup", 0) + event(time=100) + frame(1, 101) + frame(2, 102), "b.INFO")
        output = self.root / "output"
        with patch.dict(os.environ, {"DISPLAY": "", "WAYLAND_DISPLAY": ""}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(drift.main(["--log", str(a), "--opt", str(b), "--output", str(output),
                                         "--lio-window", "10.0", "--export-data"]), 0)
        import matplotlib
        self.assertEqual(matplotlib.get_backend().lower(), "agg")
        self.assertEqual({p.name for p in output.iterdir()}, {"summary.json", "frames.csv"})
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["runs"]["log"]["time_origin"], 10)
        self.assertEqual(summary["runs"]["opt"]["time_origin"], 100)
        with (output / "frames.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["relative_time"] for r in rows], ["1", "2", "1", "2"])
        self.assertEqual(rows[0]["cov_G_2"], "23")
        self.assertEqual(rows[0]["residual"], "nan")
        self.assertNotIn("NaN", (output / "summary.json").read_text())
        self.assertTrue(drift.configure_backend(True))

    def test_bad_comparison_and_no_frames(self):
        a = self.log(event() + frame())
        b = self.log(frame(), "noevent.INFO")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(drift.main(["--log", str(a), "--opt", str(b), "--no-show"]), 2)
        with self.assertRaisesRegex(ValueError, "没有可解析"):
            drift.load_run(self.log("unrelated\n", "empty.INFO"))

    def test_timestamp_rotation_across_directories(self):
        self.log(event("startup", 0) + event() + frame(),
                 "z/host.INFO.20260920-100000.123")
        self.log(frame(2, 12), "a/host.INFO.20260920-110000.123")
        run = drift.load_run(self.root)
        self.assertEqual([r["frame"] for r in run.frames], [1, 2])
        self.assertTrue(run.frames[1]["connected"])

    def test_dirty_line_breaks_path_and_late_startup(self):
        run = drift.load_run(self.log(event() + frame() + '[DLI_FRAME] bad\n' + frame(2, 12)))
        self.assertFalse(run.frames[1]["connected"])
        with self.assertRaisesRegex(ValueError, "startup 出现在"):
            drift.load_run(self.log(frame() + event("startup", 0), "late.INFO"))

    def test_no_posterior_headless_export(self):
        path = self.log(frame(status="predict", position=[None]*3))
        output = self.root / "empty_motion"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(drift.main(["--log", str(path), "--output", str(output), "--no-show"]), 0)
            self.assertFalse(output.exists())
            self.assertEqual(drift.main(["--log", str(path), "--output", str(output),
                                         "--no-show", "--export-data"]), 0)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["runs"]["log"]["valid_posteriors"], 0)
        self.assertEqual({p.name for p in output.iterdir()}, {"summary.json", "frames.csv"})

    def test_imu_excitation_loader_and_page(self):
        rows = []
        t = 100.0
        for _ in range(300):  # 3 s at 100 Hz; first 2 s stationary (acc in g units)
            if t < 102.0:
                acc, gyro = [0.01, -0.02, 1.005], [0.001, -0.001, 0.002]
            else:
                acc, gyro = [0.51, -0.02, 1.005], [0.0, 0.0, 0.2]
            rows.append(f"{t:.3f}  {gyro[0]} {gyro[1]} {gyro[2]}  "
                        f"{acc[0]} {acc[1]} {acc[2]}")
            t += 0.01
        (self.root / "legacy_logs").mkdir()
        (self.root / "legacy_logs" / "imu.txt").write_text("\n".join(rows) + "\n",
                                                           encoding="utf-8")
        run = drift.load_run(self.log(event("startup", 0) + event() + frame()))
        data = drift.load_imu_excitation(run)
        self.assertIsNone(data["error"])
        self.assertEqual(data["unit"], "g")
        self.assertTrue(data["stationary"])
        self.assertAlmostEqual(data["gravity_g"][2], 1.005, places=6)
        self.assertLess(data["dyn_mag"][:200].max(), 0.1)      # static window: no motion
        self.assertAlmostEqual(data["dyn_mag"][250:].mean(), 0.5 * 9.805, delta=0.15)
        self.assertIs(drift.load_imu_excitation(run), data)    # cached on the run
        self.assertIsNone(data.get("win_t0"))   # schema1 log: no solver window info
        # schema2 run carrying a DLI_BATCH solver window: window stats + shading
        sub = self.root / "schema2"
        (sub / "legacy_logs").mkdir(parents=True)
        (sub / "legacy_logs" / "imu.txt").write_text("\n".join(rows) + "\n",
                                                     encoding="utf-8")
        log2 = sub / "run.INFO"
        log2.write_text(event2(time=103.0) + frame2(1, 102.5, rtk_time=102.5) +
                       batch2("linear_initial", window_start=102.0, window_end=103.0,
                              time=103.0, data_time=103.0, sequence=2),
                       encoding="utf-8")
        run2 = drift.load_run(log2)
        data2 = drift.load_imu_excitation(run2)
        self.assertIsNone(data2["error"])
        self.assertAlmostEqual(data2["win_t0"], -1.0, places=3)
        self.assertAlmostEqual(data2["win_t1"], 0.0, places=3)
        self.assertGreaterEqual(data2["win_count"], 2)
        self.assertAlmostEqual(data2["win_acc_mean"], 0.5 * 9.805, delta=0.15)
        self.assertGreater(data2["win_gyro_mean"], 0.1)
        # re-logged window replays: the dynamic-init pass loop re-writes the solver
        # window; the loader must de-duplicate by timestamp and restore monotonic t
        replayed = rows + rows + rows            # 3 identical copies of the whole file
        (sub / "legacy_logs" / "imu.txt").write_text("\n".join(replayed) + "\n",
                                                     encoding="utf-8")
        run3 = drift.load_run(log2)
        data3 = drift.load_imu_excitation(run3)
        self.assertIsNone(data3["error"])
        self.assertEqual(data3["count"], 300)            # duplicates removed
        self.assertEqual(data3["replayed_logs"], 2)     # two backward jumps
        self.assertLess(np.diff(data3["t"]).max(), 1.0)  # monotonic after de-dup
        self.assertEqual(int((np.diff(data3["t"]) < 0).sum()), 0)
        self.assertAlmostEqual(data3["dyn_mag"][250:].mean(), 0.5 * 9.805, delta=0.15)
        drift.configure_backend(True)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        drift.plot_page(fig, "initialization", 3, [run])
        self.assertTrue(fig.axes)
        self._audit_english(fig)
        plt.close(fig)
        fig2 = plt.figure()
        drift.plot_page(fig2, "initialization", 3, [run2])
        self.assertTrue(fig2.axes)
        self._audit_english(fig2)
        plt.close(fig2)

    def test_imu_excitation_missing_log_renders_empty(self):
        drift.configure_backend(True)
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as other:
            path = Path(other) / "run.INFO"
            path.write_text(event("startup", 0) + event() + frame(), encoding="utf-8")
            run = drift.load_run(path)
            data = drift.load_imu_excitation(run)
            self.assertIsNotNone(data["error"])
            fig = plt.figure()
            drift.plot_page(fig, "initialization", 3, [run])
            self.assertTrue(fig.axes)
            plt.close(fig)

    def _cjk(self, text):
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    def _audit_english(self, fig):
        import matplotlib.text
        for item in fig.findobj(match=matplotlib.text.Text):
            text = item.get_text()
            if text:
                self.assertFalse(self._cjk(text), f"Chinese text in figure: {text}")

    def _all_pages(self, viewer):
        for section in drift.SECTION_NAMES:
            for page in range(len(drift.SECTION_PAGES[section])):
                viewer.show(section, page)
                yield section, page

    def test_viewer_buttons_and_click(self):
        drift.configure_backend(True)
        import matplotlib.pyplot as plt
        run = drift.load_run(self.log(event() + frame()))
        drift.add_plot_fields(run)
        import matplotlib
        self.assertEqual(matplotlib.rcParams["keymap.save"], [])
        # Viewer must suppress a toolbar even if a caller re-enables it globally.
        with matplotlib.rc_context({"toolbar": "toolbar2", "keymap.save": ["s"]}):
            viewer = drift.Viewer([run], plt)
        self.addCleanup(plt.close, viewer.fig)
        self.assertIsNone(viewer.fig.canvas.manager.toolbar)
        callbacks = viewer.fig.canvas.callbacks.callbacks.get("key_press_event", {})
        self.assertNotIn(viewer.fig.canvas.manager.key_press_handler_id, callbacks)
        # Three first-level sections; every section/page pair must render.
        self.assertEqual(sum(len(drift.SECTION_PAGES[s]) for s in drift.SECTION_NAMES), 11)
        for section, page in self._all_pages(viewer):
            self.assertEqual(len(viewer.section_buttons), 3)
            self.assertEqual(len(viewer.page_buttons), len(drift.SECTION_PAGES[section]))
            self.assertTrue(viewer.fig.axes)
        # Schema1 log has no initialization passes: only the order-axis control exists.
        self.assertEqual(len(viewer.controls), 1)
        viewer.show("lio", 1)
        ax = next(a for a in viewer.fig.axes if getattr(a, "_drift_time", False))
        viewer.on_click(SimpleNamespace(inaxes=ax, xdata=1))
        self.assertIn("frame=1", viewer.info.get_text())
        self.assertIn("post_dynamic/post", viewer.info.get_text())
        # Axis-mode toggle switches the shared x-axis semantic and redraws.
        viewer.toggle_order(None)
        self.assertTrue(viewer.order_mode)
        self.assertIn("processing order", viewer.controls[0].label.get_text().lower())
        viewer.toggle_order(None)
        self.assertFalse(viewer.order_mode)
        plt.close(viewer.fig)

    def test_schema2_dashboard_pages_pass_filter_and_english(self):
        drift.configure_backend(True)
        import matplotlib.pyplot as plt
        text = (event2("startup", 0) + event2() +
                frame2(1, 11, stage="initial_lgo", pass_id=0) +
                frame2(2, 12, stage="cache_reprocess", pass_id=1) +
                frame2(3, 13) +
                batch2("solved", pass_id=0) + batch2("batch_end", pass_id=0) +
                batch2("solved", pass_id=1) + batch2("batch_end", pass_id=1) +
                queue2())
        run = drift.load_run(self.log(text))
        drift.add_plot_fields(run)
        viewer = drift.Viewer([run], plt)
        self.addCleanup(plt.close, viewer.fig)
        # Schema2 batches/frames provide pass ids: pass-filter control is present.
        self.assertEqual(len(viewer.controls), 2)
        self.assertIn("all passes", viewer.controls[1].label.get_text())
        for section, page in self._all_pages(viewer):
            self._audit_english(viewer.fig)
        # Pass filter narrows initialization frames to one pass and redraws.
        viewer.show("initialization", 0)
        viewer.toggle_pass(None)
        self.assertEqual(viewer.pass_filter, 0)
        self.assertIn("0", viewer.controls[1].label.get_text())
        self._audit_english(viewer.fig)
        # Reprocess quality page 2 is the frame-level view of the selected pass.
        viewer.show("initialization", 2)
        self._audit_english(viewer.fig)
        ctx_frames = [r for r in run.frames
                      if r["stage"] == "cache_reprocess" and r["pass_id"] == 0]
        self.assertFalse(ctx_frames)  # fixture cache frames belong to pass 1 only
        viewer.toggle_pass(None)
        self.assertEqual(viewer.pass_filter, 1)
        viewer.show("initialization", 0)
        self._audit_english(viewer.fig)
        viewer.toggle_pass(None)
        self.assertIsNone(viewer.pass_filter)
        plt.close(viewer.fig)

    def test_dashboard_empty_data_uses_english_notes(self):
        drift.configure_backend(True)
        import matplotlib.pyplot as plt
        run = drift.load_run(self.log(event() + frame()))
        drift.add_plot_fields(run)
        viewer = drift.Viewer([run], plt)
        self.addCleanup(plt.close, viewer.fig)
        viewer.show("initialization", 0)
        notes = [t.get_text() for t in viewer.fig.findobj(match=plt.Text)
                 if "No data available" in t.get_text()]
        self.assertTrue(notes)
        self._audit_english(viewer.fig)
        plt.close(viewer.fig)


    def test_lio_covariance_symlog_ticks_visible(self):
        # symlog 默认 LogFormatterSciNotation 在数据跨度不足一个数量级时
        # 返回空标签; _symlog_axis 须为每个协方差面板补齐可见 y 刻度。
        drift.configure_backend(True)
        import matplotlib.pyplot as plt
        cov = [1e-6, 2e-6, 4e-6] * 8      # span < 1 decade, all non-negative
        frames = ''.join(frame2(i, 10 + i, covariance_diagonal=cov)
                         for i in range(1, 6))
        run = drift.load_run(self.log(event2() + frames))
        drift.add_plot_fields(run)
        fig = plt.figure()
        self.addCleanup(plt.close, fig)
        drift.plot_page(fig, "lio", 3, [run])
        fig.canvas.draw()
        self.assertEqual(len(fig.axes), 8)
        for ax in fig.axes:
            lo, hi = ax.get_ylim()
            labels = [t.get_text() for t in ax.get_yticklabels()
                      if t.get_text() and lo <= t.get_position()[1] <= hi]
            self.assertGreaterEqual(len(labels), 2,
                                    "symlog covariance panel lost its y ticks")
        # 大跨度数据: 子刻度密度自适应稀疏, 标签不得叠成一团。
        wide = [1e-6, 1e-4, 1e-2] * 8      # 4 decades, all non-negative
        frames = ''.join(frame2(i, 10 + i, covariance_diagonal=wide)
                         for i in range(1, 6))
        run = drift.load_run(self.log(event2() + frames))
        drift.add_plot_fields(run)
        fig = plt.figure()
        self.addCleanup(plt.close, fig)
        drift.plot_page(fig, "lio", 3, [run])
        fig.canvas.draw()
        for ax in fig.axes:
            lo, hi = ax.get_ylim()
            labels = [t.get_text() for t in ax.get_yticklabels()
                      if t.get_text() and lo <= t.get_position()[1] <= hi]
            self.assertGreaterEqual(len(labels), 2)
            self.assertLessEqual(len(labels), 8,
                                 "symlog covariance panel ticks are crowded")
        # 窄跨度全正数据: symlog 视区放不下两个 decade 刻度, 退化为线性轴。
        narrow = [7.7e-6, 8.9e-6, 1.0e-5] * 8   # span ~0.11 decade
        frames = ''.join(frame2(i, 10 + i, covariance_diagonal=narrow)
                         for i in range(1, 6))
        run = drift.load_run(self.log(event2() + frames))
        drift.add_plot_fields(run)
        fig = plt.figure()
        self.addCleanup(plt.close, fig)
        drift.plot_page(fig, "lio", 3, [run])
        fig.canvas.draw()
        for ax in fig.axes:
            lo, hi = ax.get_ylim()
            labels = [t.get_text() for t in ax.get_yticklabels()
                      if t.get_text() and lo <= t.get_position()[1] <= hi]
            self.assertGreaterEqual(len(labels), 2)

    def test_default_no_files_and_config_overrides(self):
        path = self.log(event() + frame() + frame(2, 12))
        output = self.root / "not_created"
        before = set(self.root.rglob("*"))
        text = io.StringIO()
        config = dict(log=str(path), output=str(output), lio_window=0.5,
                      show=True, export_data=False)
        with patch.dict(drift.CONFIG, config), patch.dict(os.environ, {"DISPLAY": "", "WAYLAND_DISPLAY": ""}), \
                contextlib.redirect_stdout(text), patch.object(drift, "Viewer") as viewer:
            self.assertEqual(drift.main([]), 0)
            viewer.assert_not_called()
        self.assertEqual(set(self.root.rglob("*")), before)
        self.assertIn("窗口未打开", text.getvalue())
        self.assertIn("有效后验 0", text.getvalue())
        with patch.dict(drift.CONFIG, dict(config, log="missing", export_data=True)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(drift.main(["--log", str(path), "--lio-window", "2", "--no-show",
                                         "--no-export-data"]), 0)
        self.assertFalse(output.exists())
        other = self.root / "explicit"
        with patch.dict(drift.CONFIG, config), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(drift.main(["--output", str(other), "--lio-window", "2",
                                         "--export-data", "--no-show"]), 0)
        summary = json.loads((other / "summary.json").read_text())["runs"]["log"]
        self.assertEqual(summary["valid_posteriors"], 2)
        self.assertFalse(output.exists())
        self.assertEqual({p.name for p in other.iterdir()}, {"summary.json", "frames.csv"})

    def test_invalid_windows_and_errors_write_nothing(self):
        path = self.log(event() + frame())
        output = self.root / "invalid"
        before = set(self.root.rglob("*"))
        for value in ("0", "-1", "nan", "inf", "-inf", "oops"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    drift.main(["--log", str(path), "--output", str(output),
                                "--export-data", "--lio-window=" + value])
                self.assertEqual(error.exception.code, 2)
        self.assertEqual(set(self.root.rglob("*")), before)
        empty = self.log("unrelated\n", "empty.INFO")
        noevent = self.log(frame(), "noevent.INFO")
        for extra in (["--log", str(empty)], ["--log", str(self.root / "missing")],
                      ["--log", str(path), "--opt", str(noevent)]):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(drift.main(extra + ["--output", str(output), "--export-data"]), 2)
            self.assertFalse(output.exists())

    def test_window_boundaries_retain_diagnostics(self):
        path = self.log(frame(1, 9, stage="initializing") + event() + frame(2, 10) +
                        frame(3, 10.5) + frame(4, 11) + frame(5, 11.000001) + frame(6, 12))
        run = drift.load_run(path, lio_window=1)
        self.assertEqual([r["selected"] for r in run.frames], [False, True, True, True, False, False])
        self.assertEqual([r["connected"] for r in run.frames], [False, False, True, True, False, False])
        summary = drift.run_summary(run)
        self.assertEqual(summary["frames"], 6)
        self.assertEqual(summary["connected_steps"], 2)
        self.assertEqual(summary["connected_path_length_m"], 2)
        self.assertEqual(summary["diagnostics_all_frames"]["effective"]["count"], 6)
        window = summary["lio_window"]
        self.assertEqual(window["outside_window_frames"], 2)
        self.assertEqual(window["valid_coverage"], {"first_s": 0, "last_s": 1, "span_s": 1})
        self.assertEqual(run.frames[0]["stage"], "initializing")
        _, motion = drift.series(run, "position", True)
        self.assertTrue(np.isnan(motion[-1]).all())

    def test_window_does_not_reconnect_or_fabricate_coverage(self):
        path = self.log(event() + frame(1, 10.5) + frame(2, 12) + frame(3, 10.75))
        run = drift.load_run(path, lio_window=1)
        self.assertFalse(any(r["connected"] for r in run.frames))
        self.assertIsNone(drift.run_summary(run)["connected_path_length_m"])
        self.assertEqual(drift.window_summary(run)["valid_coverage"]["span_s"], 0)
        empty = drift.load_run(self.log(event() + frame(1, 12), "outside.INFO"), lio_window=1)
        self.assertIsNone(drift.window_summary(empty)["valid_coverage"]["first_s"])

    # ---- schema2 (unified protocol) ----

    def test_schema2_segmented_monotonicity(self):
        reprocess = ''.join(frame2(i, 9 + i, stage="cache_reprocess", pass_id=0, frame=i)
                            for i in (1, 2, 3))
        revisit = ''.join(frame2(i + 3, 9 + i, stage="cache_reprocess", pass_id=1,
                                frame=i + 3, relative_time=99.5 + i) for i in (1, 2, 3))
        posts = frame2(7, 13) + frame2(8, 14)
        # 跨轮(pass 0 -> 1)重访相同雷达时间合法, 不计非单调。
        run = drift.load_run(self.log(event2() + reprocess + revisit + posts))
        self.assertEqual(run.schema, 2)
        self.assertEqual(run.counts["invalid_or_nonmonotonic"], 0)
        self.assertTrue(all(r["valid_motion"] for r in run.frames if r["stage"] == "post_dynamic"))
        # prepare_run 覆盖 relative_time, 保留原始 emitted_relative_time。
        self.assertEqual(run.frames[3]["relative_time"], 0.0)
        self.assertEqual(run.frames[3]["emitted_relative_time"], 100.5)
        # 同一 (stage, pass) 段内时间回退仍计非单调, 并附分段语义注记。
        regress = frame2(7, 9.5, stage="cache_reprocess", pass_id=1, frame=7)
        bad = drift.load_run(self.log(event2() + reprocess + revisit + regress +
                                      frame2(8, 13) + frame2(9, 14)))
        self.assertEqual(bad.counts["invalid_or_nonmonotonic"], 1)
        self.assertTrue(any("跨轮重处理" in m for m in bad.messages))

    def test_schema2_batch_metrics_failure_split(self):
        ends = (batch2("batch_end", pass_id=0, initial_speed=5.6, delta_v=0.1) +
                batch2("batch_end", pass_id=1, batch_id=1, initial_speed=9.8, delta_v=4.2) +
                batch2("batch_end", pass_id=2, batch_id=2, initial_speed=None,
                       delta_v=None, finite=0, quality_reason="quality_warning"))
        solved = (batch2("linear_initial", pass_id=0) + batch2("gyro_bias", pass_id=0) +
                  batch2("linear_initial", pass_id=1) + batch2("gyro_bias", pass_id=1) +
                  batch2("linear_initial", pass_id=2, system_finite=0, solution_finite=0) +
                  batch2("gyro_bias", pass_id=2, decomposition_status=1, solve_status=1,
                         system_finite=0, solution_finite=0) +
                  batch2("gravity_refine", pass_id=2, batch_id=1, refinement=1,
                         decomposition_status=1, solve_status=1,
                         system_finite=0, solution_finite=0))
        run = drift.load_run(self.log(event2() + frame2() + solved + ends))
        metrics = drift.run_summary(run)["initialization"]
        self.assertEqual(metrics["run_id"], "rid")
        self.assertEqual(metrics["init_done_reason"], "test")
        self.assertEqual(metrics["pass_count"], 3)
        self.assertEqual(metrics["batch_solved_observations"], 7)
        self.assertEqual(metrics["batch_solve_failures"], 3)
        self.assertEqual(metrics["batch_solve_nonfinite"], 3)
        self.assertEqual(metrics["batch_solve_status_nonzero"], 2)
        self.assertEqual(metrics["final_pass"]["pass_id"], 2)
        self.assertIsNone(metrics["final_pass"]["initial_speed"])
        self.assertEqual(metrics["last_finite_pass"]["pass_id"], 1)
        self.assertEqual(metrics["last_finite_pass"]["delta_v"], 4.2)

    def test_schema2_queue_heartbeat(self):
        run = drift.load_run(self.log(event2() + frame2() +
                                      queue2(lidar_queue=5, imu_queue=7)))
        metrics = drift.run_summary(run)["initialization"]
        self.assertEqual(run.counts["DLI_QUEUE"], 1)
        self.assertEqual(metrics["queue_heartbeats"], 1)
        self.assertEqual(metrics["queue_depth_max"], {"lidar": 5, "imu": 7})

    def test_schema2_strict_and_multi_run_rejected(self):
        dirty = [edit2(frame2(), drop="lag_reason"),
                 edit2(frame2(), before_propagation={"position": [1, 0, 0]}),
                 edit2(batch2("gyro_bias"), refinement=None),
                 edit2(batch2("batch_end"), refinement=0)]
        run = drift.load_run(self.log(event2() + frame2() + ''.join(dirty)))
        self.assertEqual(run.counts["bad_lines"], 4)
        self.assertEqual(len(run.frames), 1)
        with self.assertRaisesRegex(ValueError, "多个 run_id"):
            drift.load_run(self.log(event2() + frame2() + edit2(frame2(), run_id="other")))
        with self.assertRaisesRegex(ValueError, "混入不同 schema"):
            drift.load_run(self.log(event2() + frame2() + frame()))

    def test_schema2_export_nested_snapshot_single_cell(self):
        path = self.log(event2() + frame2(1, 11) + frame2(2, 12, rtk_velocity=None))
        output = self.root / "schema2_out"
        with patch.dict(os.environ, {"DISPLAY": "", "WAYLAND_DISPLAY": ""}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(drift.main(["--log", str(path), "--output", str(output),
                                         "--export-data", "--no-show"]), 0)
        summary = json.loads((output / "summary.json").read_text())
        init = summary["runs"]["log"]["initialization"]
        self.assertIn("batch_solve_nonfinite", init)
        self.assertIn("last_finite_pass", init)
        with (output / "frames.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        cell = json.loads(rows[0]["before_propagation"])
        self.assertEqual(cell["position"], [1, 0, 0])
        self.assertEqual(cell["covariance_finite"], 1)
        # 同一字段 null 标量与定长数组混合: 列并集, 缺失格留空。
        self.assertEqual(rows[0]["rtk_velocity_0"], "1")
        self.assertEqual(rows[1]["rtk_velocity"], "nan")
        self.assertEqual(rows[1]["rtk_velocity_0"], "")
        self.assertEqual(rows[0]["rtk_velocity"], "")

    def test_per_pass_aggregate_collapses_reprocess(self):
        # 三轮重处理同一窗口: 每轮 2 帧 effective, 聚合后每轮一个均值点。
        frames = [frame2(1, 11, stage="cache_reprocess", pass_id=0, effective=10),
                  frame2(2, 12, stage="cache_reprocess", pass_id=0, effective=20),
                  frame2(3, 11, stage="cache_reprocess", pass_id=1, effective=30),
                  frame2(4, 12, stage="cache_reprocess", pass_id=1, effective=50),
                  frame2(5, 11, stage="cache_reprocess", pass_id=2, effective=None)]
        run = drift.load_run(self.log(event2() + ''.join(frames)))
        pts = drift.per_pass_aggregate(run.frames, "effective")
        self.assertEqual(pts, [(0, 15.0), (1, 40.0)])  # null 值跳过, 不当零
        vec = drift.per_pass_aggregate(run.frames, "velocity", component=0)
        self.assertEqual(vec, [(0, 1.0), (1, 1.0), (2, 1.0)])

    def test_schema2_handoff_quality_metrics(self):
        # pass 0-1 有限, pass 2 起非有限; post 帧 effective=0 且状态 null。
        ends = (batch2("batch_end", pass_id=0, finite=1) +
                batch2("batch_end", pass_id=1, finite=1) +
                batch2("batch_end", pass_id=2, finite=0, initial_speed=None,
                       delta_v=None, quality_reason="quality_warning"))
        frames = (frame2(1, 11, effective=0, status="partial_applied",
                         reason="no_effective_points", velocity=[None] * 3,
                         gravity=[None] * 3, position=[None] * 3) +
                  frame2(2, 12, effective=0, status="partial_applied",
                         reason="no_effective_points", velocity=[None] * 3) +
                  frame2(3, 13, effective=5))
        run = drift.load_run(self.log(event2() + frames + ends))
        metrics = drift.run_summary(run)["initialization"]
        self.assertEqual(metrics["batch_finite_pass_count"], 2)
        self.assertEqual(metrics["first_nonfinite_pass"], 2)
        self.assertEqual(metrics["post_dynamic_frames"], 3)
        self.assertEqual(metrics["post_effective_zero_frames"], 2)
        self.assertEqual(metrics["post_state_null_frames"], 2)
        self.assertEqual(metrics["post_reason_counts"],
                         {"no_effective_points": 2, "matched": 1})


if __name__ == "__main__":
    unittest.main()