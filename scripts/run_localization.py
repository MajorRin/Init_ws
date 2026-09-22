#!/usr/bin/env python3
"""隔离回放与诊断采集；通常通过同目录 Shell 入口加载 ROS 环境后运行。"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import xmlrpc.client

# ================= 用户配置区：通常只需修改这里 =================
# 命令行显式参数优先于这里的配置；布尔开关填写 True 或 False。
CONFIG = {
    "bag": "/home/mao/rosbag/HZ_LiDAR_Dataset/AVIA/rosbag_all_19700101_080314.bag",
    "start": 250.2,           # 从 bag 开头偏移多少秒开始回放
    "duration": 20.0,         # 雷达回放窗口长度，单位：秒（与分析窗口独立）
    "rate": 1.0,              # 回放倍率，1.0 为原速
    "drain_timeout": 120.0,   # 回放结束后等待雷达队列排空的最长秒数
    "init_grace": 30.0,       # bag 播完后允许初始化仍未收敛的额外挂钟秒数；超时强制收尾
    "imu_tail": 0.5,          # 窗口后仅补播 IMU 的秒数，0 表示禁用
    "rviz": False,            # 是否打开 RViz 窗口
    "record": False,          # 是否录制估计里程计和注册点云（可能较大）
    "analysis": True,         # 回放后自动分析；不生成图片
    "show_dashboard": True,   # 打开交互仪表盘；无桌面时仅终端摘要
    "export_data": False,     # 显式启用才导出 JSON/CSV 统计
    "lio_window": 10.0,       # 初始化交接后观察窗口，不延长雷达回放
    "output": "",             # 留空：自动创建时间戳目录；否则填新目录路径
}
# 输出目录必须不存在，避免覆盖历史结果；支持 ~，相对路径基于启动目录。
# ================= 用户配置区结束 =================


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="隔离 ROS 回放并采集初始化与 LIO glog；不修改算法参数",
        epilog="无参数运行使用文件顶部 CONFIG；命令行参数覆盖配置。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("bag", nargs="?", default=CONFIG["bag"], help="输入 bag 路径")
    for option, help_text in (
            ("start", "bag 起始偏移秒"), ("duration", "雷达回放长度秒"),
            ("rate", "回放倍率"), ("drain_timeout", "排空超时秒"),
            ("init_grace", "bag 播完后允许初始化未收敛的额外挂钟秒数"),
            ("imu_tail", "窗口后仅补播 IMU 秒数，0 禁用"),
            ("lio_window", "LIO 分析观察窗口秒数，不延长回放")):
        p.add_argument("--" + option.replace("_", "-"), type=float,
                       default=CONFIG[option], help=help_text)
    for key, positive, negative, help_text in (
            ("rviz", "--rviz", "--no-rviz", "打开 RViz"),
            ("record", "--record", "--no-record", "录制里程计和注册点云"),
            ("analysis", "--analysis", "--no-analysis", "回放后自动分析，不生成图片"),
            ("show_dashboard", "--show-dashboard", "--no-show", "打开交互仪表盘"),
            ("export_data", "--export-data", "--no-export-data", "导出 JSON/CSV 统计")):
        group = p.add_mutually_exclusive_group()
        group.add_argument(positive, dest=key, action="store_true", help=help_text)
        group.add_argument(negative, dest=key, action="store_false", help="禁用：" + help_text)
        p.set_defaults(**{key: CONFIG[key]})
    p.add_argument("--output", type=Path,
                   default=Path(CONFIG["output"]) if CONFIG["output"] else None,
                   help="新建运行目录，不覆盖已有目录")
    p.add_argument("--self-test", action="store_true", help="仅执行同文件模拟测试，不启动 ROS")
    a = p.parse_args(argv)
    values = (a.start, a.duration, a.rate, a.drain_timeout, a.init_grace, a.imu_tail, a.lio_window)
    if (not all(math.isfinite(v) for v in values) or min(a.start, a.imu_tail) < 0
            or min(a.duration, a.rate, a.drain_timeout, a.init_grace, a.lio_window) <= 0):
        p.error("起始偏移和 IMU 补播时长必须非负，其余时间/倍率必须为有限正数")
    return a


def file_identity(path, with_hash=True):
    stat = path.stat()
    result = dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                  ctime_ns=stat.st_ctime_ns, inode=stat.st_ino, device=stat.st_dev)
    if with_hash:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def analysis_command(a, scripts, run):
    command = [sys.executable, str(scripts / "analyze_drift.py"),
               "--log", str(run / "glog"), "--output", str(run / "analysis"),
               "--lio-window", str(a.lio_window)]
    if not a.show_dashboard:
        command.append("--no-show")
    if a.export_data:
        command.append("--export-data")
    return command


class TimeoutTransport(xmlrpc.client.Transport):
    """只限制本次 XML-RPC，不能修改导入者的全局 socket 超时。"""
    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = 2
        return connection


def rpc(uri):
    return xmlrpc.client.ServerProxy(uri, transport=TimeoutTransport())


def node_pid(master, name):
    answer = master.lookupNode("/drift_runner", name)
    if answer[0] != 1:
        raise RuntimeError(f"节点未注册: {name}")
    with rpc(answer[2]) as node:
        answer = node.getPid("/drift_runner")
    if answer[0] != 1 or not isinstance(answer[2], int) or answer[2] <= 0:
        raise RuntimeError(f"节点无有效 PID: {name}")
    return answer[2]


def stop(proc):
    # Only our own process groups; never pkill/killall unrelated mapping nodes.
    for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=timeout)
            # The group leader may exit while its children are still running.
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
        except subprocess.TimeoutExpired:
            pass


class DiagnosticReader:
    """有界增量读取普通文件；不等待下一行，不使用阻塞管道。"""
    def __init__(self, run):
        self.run = run
        self.positions = {}
        self.summary = None

    def forward_summary(self):
        if self.summary is not None:
            return
        files = [self.run / "launch.log"] + sorted((self.run / "glog").glob("*INFO*"))
        for path in files:
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
            key = (str(path), stat.st_ino)
            offset, partial = self.positions.get(key, (0, b""))
            if stat.st_size < offset:
                offset, partial = 0, b""
            with path.open("rb") as stream:
                stream.seek(offset)
                block = stream.read(262144)
                offset = stream.tell()
            lines = (partial + block).split(b"\n")
            self.positions[key] = (offset, lines.pop()[-262144:])
            for raw in lines:
                line = raw.decode("utf-8", errors="replace")
                if "[DLI_EVENT]" not in line:
                    continue
                try:
                    event = json.loads(line.split("[DLI_EVENT]", 1)[1].strip())
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("event") in ("init_done", "init_complete", "initialization_complete"):
                    self.summary = event
                    print("初始化完成摘要 [DLI_EVENT] " + json.dumps(event, ensure_ascii=False), flush=True)
                    return

    def init_done(self):
        return self.summary is not None

    def heartbeat(self):
        files = [f for f in (self.run / "glog").glob("*INFO*") if not f.is_symlink()]
        if not files:
            return None, None
        path = max(files, key=lambda x: x.stat().st_mtime_ns)
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 131072))
            lines = stream.read(131072).decode("utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            if "[DLI_QUEUE]" in line:
                try:
                    q = json.loads(line.split("[DLI_QUEUE]", 1)[1].strip())
                    if isinstance(q, dict) and all(k in q for k in ("initialized", "lidar_queue", "latest_lidar", "latest_imu")):
                        return line, q
                except ValueError:
                    pass
        return None, None


def subscriptions(master):
    state = master.getSystemState("/drift_runner")
    if state[0] != 1:
        raise RuntimeError("无法读取私有 master 的订阅状态")
    return dict(state[2][1])


def check_health(master, core, launch, mapping_pid, recorder=None, recorder_pid=None):
    for name, proc in (("roscore", core), ("roslaunch", launch), ("record", recorder)):
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"{name} 提前退出，状态 {proc.returncode}")
    if node_pid(master, "/laserMapping") != mapping_pid:
        raise RuntimeError("映射实际节点 PID 已改变")
    if recorder is not None and node_pid(master, "/drift_recorder") != recorder_pid:
        raise RuntimeError("录包实际节点 PID 已改变")


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    a = parse_args(argv)
    if a.self_test:
        return self_test()
    scripts = Path(__file__).resolve().parent
    workspace = scripts.parents[2]
    bag = Path(a.bag).expanduser().resolve()
    binary = workspace / "devel/lib/lidar_imu_init/lidar_imu_init"
    if not bag.is_file() or not binary.is_file():
        print("输入 bag 或 lidar_imu_init 可执行文件不存在，请检查路径并编译真实包。", file=sys.stderr)
        return 2
    run = (a.output or scripts / "drift_runs" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")).expanduser().resolve()
    try:
        # Read/hash source configuration without changing any algorithm parameters.
        identities = {"binary": file_identity(binary), "input": file_identity(bag, False),
                      "config": file_identity(scripts.parent / "config/avia.yaml"),
                      "launch": file_identity(scripts.parent / "launch/livox_avia.launch")}
        run.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"准备失败（不覆盖已有目录）: {exc}", file=sys.stderr)
        return 2
    metadata = dict(bag=str(bag), start=a.start, duration=a.duration, rate=a.rate,
                    imu_tail=a.imu_tail, binary_mtime=binary.stat().st_mtime,
                    identities=identities, options={k: str(v) if isinstance(v, Path) else v
                                                    for k, v in vars(a).items()},
                    started_at=datetime.datetime.now().astimezone().isoformat(),
                    drain_status="not_started", commands=[], process_exit_codes={},
                    capture_exit_code=None, analysis_exit_code=None,
                    analysis_status="not_started", display_status="not_requested")
    processes, streams = [], []
    reader = DiagnosticReader(run)
    old_signal = signal.getsignal(signal.SIGTERM)
    code = 1

    def save():
        (run / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    def start(name, command):
        stream = (run / f"{name}.log").open("w")
        streams.append(stream)
        metadata["commands"].append(command)
        save()
        proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        processes.append((name, proc))
        metadata.setdefault("process_pids", {})[name] = proc.pid
        save()
        return proc

    try:
        signal.signal(signal.SIGTERM, interrupted)
        for name in ("glog", "init_comparison", "legacy_logs", "ros"):
            (run / name).mkdir()
        # Private master only; never connect to or clean up existing ROS sessions.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = dict(os.environ, ROS_MASTER_URI=f"http://127.0.0.1:{port}",
                   ROS_IP="127.0.0.1", ROS_HOME=str(run / "ros"), ROS_LOG_DIR=str(run / "ros"),
                   INIT_CMP_DIR=str(run / "init_comparison"), DLI_LOG_DIR=str(run / "legacy_logs"),
                   OMP_NUM_THREADS="1", OMP_WAIT_POLICY="PASSIVE")
        env.pop("ROS_HOSTNAME", None)
        metadata["master"] = env["ROS_MASTER_URI"]
        save()
        print(f"运行目录: {run}", flush=True)
        core = start("roscore", ["roscore", "-p", str(port)])
        master = rpc(env["ROS_MASTER_URI"])
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if core.poll() is not None:
                raise RuntimeError("独立 roscore 启动失败")
            try:
                if master.getPid("/drift_runner")[0] == 1:
                    break
            except (OSError, xmlrpc.client.Error):
                pass
            time.sleep(.25)
        else:
            raise RuntimeError("独立 ROS master 启动超时")
        launch = start("launch", ["roslaunch", str(scripts.parent / "launch/livox_avia.launch"),
                       "use_sim_time:=true", "drift_diagnostics:=true",
                       f"glog_dir:={run / 'glog'}", f"rviz:={'true' if a.rviz else 'false'}"])
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            if launch.poll() is not None or core.poll() is not None:
                raise RuntimeError("映射 launch 或 master 提前退出")
            try:
                required_topics = []
                for key in ("/common/lid_topic", "/common/imu_topic"):
                    value = master.getParam("/drift_runner", key)
                    if value[0] != 1 or not isinstance(value[2], str) or not value[2]:
                        raise RuntimeError(f"输入话题参数未就绪: {key}")
                    required_topics.append("/" + value[2].lstrip("/"))
                subscribers = subscriptions(master)
                mapping_pid = node_pid(master, "/laserMapping")
                if (all("/laserMapping" in subscribers.get(t, []) for t in required_topics)
                        and reader.heartbeat()[1]):
                    break
            except (OSError, RuntimeError, xmlrpc.client.Error):
                pass
            reader.forward_summary()
            time.sleep(.25)
        else:
            raise RuntimeError("映射实际节点、输入订阅或诊断心跳未就绪")
        metadata["mapping_pid"] = mapping_pid
        # Verify roslaunch resolved the same binary we fingerprinted (Linux workspace).
        actual_binary = Path(f"/proc/{mapping_pid}/exe").resolve(strict=True)
        if actual_binary != binary.resolve():
            raise RuntimeError(f"映射实际二进制与预期不符: {actual_binary}")
        metadata["actual_binary"] = file_identity(actual_binary)
        subprocess.run(["rosparam", "dump", str(run / "parameters.yaml")],
                       env=env, check=True, timeout=10)
        metadata["parameters"] = file_identity(run / "parameters.yaml")
        recorder, recorder_pid = None, None
        record_topics = ["/lidar_init_aft_mapped_to_init", "/lidar_init_cloud_registered", "/clock"]
        if a.record:
            recorder = start("record", ["rosbag", "record", "__name:=drift_recorder",
                             "-O", str(run / "estimate.bag")] + record_topics)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                check_health(master, core, launch, mapping_pid)
                if recorder.poll() is not None:
                    raise RuntimeError("录包进程就绪前退出")
                try:
                    recorder_pid = node_pid(master, "/drift_recorder")
                    subscribers = subscriptions(master)
                    if (all("/drift_recorder" in subscribers.get(t, []) for t in record_topics)
                            and (run / "estimate.bag.active").is_file()):
                        break
                except (OSError, RuntimeError, xmlrpc.client.Error):
                    pass
                reader.forward_summary()
                time.sleep(.25)
            else:
                raise RuntimeError("录包节点、订阅或输出文件未就绪")
            metadata["recorder_pid"] = recorder_pid
        save()

        def monitor():
            reader.forward_summary()
            check_health(master, core, launch, mapping_pid, recorder, recorder_pid)
            if recorder is not None:
                subscribers = subscriptions(master)
                if not all("/drift_recorder" in subscribers.get(t, []) for t in record_topics):
                    raise RuntimeError("录包输入订阅丢失")
                if not (run / "estimate.bag.active").is_file():
                    raise RuntimeError("录包活动文件丢失")

        player = start("play", ["rosbag", "play", str(bag), "--clock", "-s", str(a.start),
                                "--duration", str(a.duration), "-r", str(a.rate), "-d", "2"])
        while player.poll() is None:
            monitor()
            time.sleep(.25)
        if player.returncode:
            raise RuntimeError(f"rosbag play 退出码 {player.returncode}")
        # Supply IMU coverage beyond the cut without adding any new lidar frames.
        if a.imu_tail > 0:
            tail = start("imu_tail", ["rosbag", "play", str(bag), "--clock",
                         "-s", str(a.start + a.duration), "--duration", str(a.imu_tail),
                         "-r", str(a.rate), "-d", "2", "--topics", required_topics[1]])
            while tail.poll() is None:
                monitor()
                time.sleep(.25)
            if tail.returncode:
                raise RuntimeError(f"IMU 补播退出码 {tail.returncode}")
        metadata["drain_status"] = "waiting"
        save()
        drain_deadline = time.monotonic() + a.drain_timeout
        graceful_deadline = time.monotonic() + a.init_grace
        previous, stable, received = None, 0, None
        while time.monotonic() < drain_deadline:
            monitor()
            token, q = reader.heartbeat()
            now = time.monotonic()
            if token and token != previous:
                previous = token
                same_input = received == (q["latest_lidar"], q["latest_imu"])
                received = (q["latest_lidar"], q["latest_imu"])
                stable = stable + 1 if q["initialized"] in (True, 1) and q["lidar_queue"] == 0 and same_input else 0
                metadata["last_queue"] = q
                if stable >= 3:
                    metadata["drain_status"] = "empty_stable"
                    code = 0
                    break
            if not reader.init_done() and now > graceful_deadline:
                # bag 已播完但 init_done 始终未到：不允许无限悬停，按"初始化未终止"收尾。
                metadata["drain_status"] = "init_unterminated"
                metadata["init_unterminated"] = {
                    "grace_s": a.init_grace,
                    "last_queue": metadata.get("last_queue"),
                    "note": "bag 播完后 init_grace 秒内未收到 init_done，强制收尾",
                }
                code = 1
                break
            time.sleep(.25)
        else:
            metadata["drain_status"] = "timeout_incomplete"
            print("排空超时：保留日志，但不能宣称数据完整；检查最后队列和 IMU 覆盖。", file=sys.stderr)
    except (OSError, RuntimeError, subprocess.SubprocessError, xmlrpc.client.Error) as exc:
        metadata["error"] = str(exc)
        metadata["drain_status"] = "failed_incomplete"
        print(f"失败: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        metadata["drain_status"] = "interrupted_incomplete"
        code = 130
    finally:
        # Repeated Ctrl-C/SIGTERM must not strand our private child process groups.
        old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            for name, proc in reversed(processes):
                try:
                    stop(proc)
                except OSError as exc:
                    metadata.setdefault("cleanup_errors", []).append(f"{name}: {exc}")
                    code = code or 1
                metadata["process_exit_codes"][name] = proc.poll()
                if proc.poll() is None:
                    code = code or 1
            for stream in streams:
                stream.close()
            try:
                reader.forward_summary()
            except OSError as exc:
                metadata["summary_error"] = str(exc)
            metadata["initialization_summary"] = reader.summary
            if a.record and any(name == "record" for name, _ in processes):
                complete = (run / "estimate.bag").is_file() and not (run / "estimate.bag.active").exists()
                metadata["record_status"] = "finalized" if complete else "incomplete"
                if not complete:
                    code = code or 1
            metadata["capture_exit_code"] = code
            metadata["exit_code"] = code
            save()
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_signal)
    if a.analysis and code != 130:
        command = analysis_command(a, scripts, run)
        metadata["analysis_command"] = command
        metadata["analysis_status"] = "running"
        metadata["display_status"] = "requested_not_verified" if a.show_dashboard else "disabled"
        save()
        try:
            result = subprocess.run(command)
            metadata["analysis_exit_code"] = result.returncode
            metadata["analysis_status"] = "completed" if result.returncode == 0 else "failed"
            if result.returncode and code == 0:
                code = result.returncode
        except OSError as exc:
            metadata["analysis_error"] = str(exc)
            metadata["analysis_status"] = "failed"
            code = code or 1
        except KeyboardInterrupt:
            metadata["analysis_status"] = "interrupted"
            code = 130
    else:
        metadata["analysis_status"] = "skipped"
    metadata["exit_code"] = code
    metadata["finished_at"] = datetime.datetime.now().astimezone().isoformat()
    save()
    if reader.summary is None:
        print("未收到初始化完成事件（init_done）；不推断或补造初始化结果。", file=sys.stderr)
    print(f"采集结束: {metadata['drain_status']}，结果目录: {run}")
    return code


def self_test():
    """离线模拟：不启动 ROS，不写运行文件，不接触用户配置。"""
    import contextlib
    import io
    import runpy
    import unittest
    from unittest import mock

    class CaptureTests(unittest.TestCase):
        def test_import_has_no_side_effects(self):
            with contextlib.ExitStack() as stack:
                for target in ("subprocess.Popen", "pathlib.Path.mkdir",
                               "signal.signal", "socket.socket",
                               "socket.setdefaulttimeout",
                               "argparse.ArgumentParser.parse_args"):
                    stack.enter_context(mock.patch(target, side_effect=AssertionError(target)))
                stack.enter_context(mock.patch.object(sys, "argv", [__file__, "--invalid"]))
                module = runpy.run_path(__file__, run_name="capture_import_check")
            self.assertTrue(callable(module["main"]))

        def test_analysis_options(self):
            a = parse_args([])
            self.assertEqual(a.duration, CONFIG["duration"])
            self.assertEqual(a.lio_window, CONFIG["lio_window"])
            command = analysis_command(a, Path("scripts"), Path("run"))
            self.assertEqual(command[2:8], ["--log", "run/glog", "--output",
                                           "run/analysis", "--lio-window", str(a.lio_window)])
            a = parse_args(["--no-show", "--export-data", "--lio-window", "7"])
            command = analysis_command(a, Path("scripts"), Path("run"))
            self.assertEqual(command[-2:], ["--no-show", "--export-data"])
            a = parse_args(["--show-dashboard", "--no-export-data"])
            command = analysis_command(a, Path("scripts"), Path("run"))
            self.assertNotIn("--no-show", command)
            self.assertNotIn("--export-data", command)

        def test_invalid_cli(self):
            for argv in (["--duration", "0"], ["--rate", "nan"],
                         ["--start", "-1"], ["--imu-tail", "-1"],
                         ["--lio-window", "inf"], ["--drain-timeout", "0"],
                         ["--record", "--no-record"]):
                with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as exc:
                        parse_args(argv)
                    self.assertEqual(exc.exception.code, 2)

        def test_actual_node_health(self):
            core, launch, recorder = [mock.Mock(returncode=None) for _ in range(3)]
            for proc in (core, launch, recorder):
                proc.poll.return_value = None
            with mock.patch.dict(globals(), node_pid=mock.Mock(return_value=42)):
                check_health(None, core, launch, 42)
                with self.assertRaisesRegex(RuntimeError, "PID"):
                    check_health(None, core, launch, 43)
                with self.assertRaisesRegex(RuntimeError, "PID"):
                    check_health(None, core, launch, 42, recorder, 43)
                recorder.poll.return_value, recorder.returncode = 1, 1
                with self.assertRaisesRegex(RuntimeError, "record"):
                    check_health(None, core, launch, 42, recorder, 42)
            with mock.patch.dict(globals(), node_pid=mock.Mock(side_effect=OSError("dead"))):
                with self.assertRaises(OSError):
                    check_health(None, core, launch, 42)

        def test_summary_partial_and_duplicate(self):
            reader = DiagnosticReader(Path("/virtual"))
            event = b'[DLI_EVENT] {"event":"init_complete","reason":"done"}'
            content = [event[:20]]
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(Path, "glob", return_value=[]))
                stack.enter_context(mock.patch.object(Path, "is_symlink", return_value=False))
                stack.enter_context(mock.patch.object(Path, "is_file", return_value=True))
                stack.enter_context(mock.patch.object(Path, "stat", side_effect=lambda:
                    mock.Mock(st_ino=1, st_size=len(content[0]))))
                stack.enter_context(mock.patch.object(Path, "open", side_effect=lambda *a:
                    io.BytesIO(content[0])))
                output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                reader.forward_summary()
                self.assertIsNone(reader.summary)
                content[0] = event + b"\n" + event + b"\n"
                reader.forward_summary()
                reader.forward_summary()
                self.assertEqual(reader.summary["reason"], "done")
                self.assertEqual(output.getvalue().count("初始化完成摘要"), 1)

        def test_heartbeat_partial_and_malformed(self):
            path = mock.Mock()
            path.is_symlink.return_value = False
            path.stat.return_value = mock.Mock(st_size=256, st_mtime_ns=1)
            payload = (b'[DLI_QUEUE] {"initialized":1,"lidar_queue":0,'
               b'"latest_lidar":20,"latest_imu":21}\n'
                       b'[DLI_QUEUE] invalid\n[DLI_QUEUE] {"initialized":')
            path.open.return_value = io.BytesIO(payload)
            with mock.patch.object(Path, "glob", return_value=[path]):
                token, queue = DiagnosticReader(Path("/virtual")).heartbeat()
            self.assertEqual(queue["latest_imu"], 21)
            self.assertIsNotNone(token)

        def test_cleanup_targets_only_owned_group(self):
            proc = mock.Mock(pid=12345)
            with mock.patch.object(os, "killpg", side_effect=[None, ProcessLookupError]) as kill:
                stop(proc)
            self.assertEqual(kill.call_args_list,
                             [mock.call(12345, signal.SIGINT), mock.call(12345, 0)])
            proc.wait.assert_called_once_with(timeout=15)

        def test_cleanup_escalates_for_surviving_children(self):
            proc = mock.Mock(pid=12345)
            with mock.patch.object(os, "killpg") as kill:
                stop(proc)
            self.assertEqual([call.args[1] for call in kill.call_args_list],
                             [signal.SIGINT, 0, signal.SIGTERM, 0, signal.SIGKILL, 0])

        def test_start_failure_preserves_metadata_and_signals(self):
            snapshots = []
            before = signal.getsignal(signal.SIGTERM)
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(Path, "is_file", return_value=True))
                stack.enter_context(mock.patch.object(Path, "mkdir"))
                stack.enter_context(mock.patch.object(Path, "stat", return_value=mock.Mock(st_mtime=1)))
                stack.enter_context(mock.patch.object(Path, "open", return_value=io.StringIO()))
                stack.enter_context(mock.patch.object(Path, "write_text", side_effect=lambda text, **kw:
                    snapshots.append(json.loads(text))))
                stack.enter_context(mock.patch.dict(globals(), file_identity=mock.Mock(return_value={})))
                stack.enter_context(mock.patch.object(socket, "socket", return_value=mock.MagicMock()))
                launch = stack.enter_context(mock.patch.object(subprocess, "Popen", side_effect=OSError("missing")))
                stack.enter_context(mock.patch.object(DiagnosticReader, "forward_summary"))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                code = main(["--no-analysis", "--no-record", "--output", "/virtual/run"])
            self.assertEqual(code, 1)
            self.assertEqual(snapshots[-1]["capture_exit_code"], 1)
            self.assertEqual(snapshots[-1]["analysis_status"], "skipped")
            self.assertEqual(signal.getsignal(signal.SIGTERM), before)
            launch.assert_called_once()
            self.assertTrue(launch.call_args.kwargs["start_new_session"])

        def test_existing_output_is_never_overwritten(self):
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(Path, "is_file", return_value=True))
                stack.enter_context(mock.patch.dict(globals(), file_identity=mock.Mock(return_value={})))
                stack.enter_context(mock.patch.object(Path, "mkdir", side_effect=FileExistsError("existing")))
                launch = stack.enter_context(mock.patch.object(subprocess, "Popen"))
                write = stack.enter_context(mock.patch.object(Path, "write_text"))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                self.assertEqual(main(["--output", "/virtual/run"]), 2)
            launch.assert_not_called()
            write.assert_not_called()

        def test_file_identity(self):
            path = mock.Mock()
            path.stat.return_value = mock.Mock(st_size=3, st_mtime_ns=2,
                                               st_ctime_ns=1, st_ino=4, st_dev=5)
            path.resolve.return_value = Path("/input")
            path.open.return_value = io.BytesIO(b"abc")
            identity = file_identity(path)
            self.assertEqual(identity["sha256"], hashlib.sha256(b"abc").hexdigest())
            self.assertEqual(identity["inode"], 4)
            path.open.reset_mock()
            self.assertNotIn("sha256", file_identity(path, False))
            path.open.assert_not_called()

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CaptureTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

