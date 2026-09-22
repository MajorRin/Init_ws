#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐环节剖析两 run 初始化批解: ICP旋转 -> gyro_bias解 -> 线性速度/重力解

从 glog 的 DLI_BATCH 记录提取每 pass 的:
  - 窗口 [start,end], 帧数, 逐帧时间/速度/ICP旋转矩阵
  - gyro_bias 阶段: delta_bg
  - linear_initial 阶段: v_0, g, 逐帧速度
并与 imu.txt 的角速度/比力对照, 定位失效环节。
"""
import json, math, re, sys
import numpy as np

R1 = "/home/mao/Init_ws/src/D-LI-Init/scripts/drift_runs/20260921_174946_912530"
R2 = "/home/mao/Init_ws/src/D-LI-Init/scripts/drift_runs/20260921_180216_909199"
G1 = R1 + "/glog/lidar_imu_init.mao.mao.log.INFO.20260921-174947.3024056"
G2 = R2 + "/glog/lidar_imu_init.mao.mao.log.INFO.20260921-180217.3026090"

def load_batches(path):
    batches = {}
    pat = re.compile(r"\[DLI_BATCH\] (\{.*\})\s*$")
    with open(path, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            j = json.loads(m.group(1))
            key = (j["pass_id"], j["batch_id"])
            d = batches.setdefault(key, {"pass_id": j["pass_id"], "batch_id": j["batch_id"]})
            stage = j["stage"]
            if stage in ("gyro_bias", "linear_initial", "batch_end"):
                d[stage] = j
    return batches

def so3_log_angle(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    return math.acos(c)

def rotvec(R):
    """SO3 旋转矩阵 -> 旋转向量"""
    a = so3_log_angle(R)
    if a < 1e-9:
        return np.zeros(3)
    w = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*math.sin(a))
    return w * a

def frame_R_vec(entry):
    ts, Rs, vs = [], [], []
    for v in entry.get("velocities", []):
        ts.append(v["time"])
        Rs.append(np.array(v["R_local_to_batch_column_major"]).reshape(3, 3))
        vel = v.get("velocity")
        vs.append(np.array(vel) if vel is not None else None)
    return np.array(ts), Rs, vs

def load_imu(path):
    return np.loadtxt(path)

def interp_gyro(imu, ts):
    """imu.txt 列序: t gx gy gz ax ay az (陀螺在前, 角速度 rad/s)"""
    t = imu[:, 0]
    out = np.zeros((len(ts), 3))
    for k in range(3):
        out[:, k] = np.interp(ts, t, imu[:, 1 + k])
    return out

def imu_integrate_rot(imu, t0, t1, step=0.002):
    """IMU 角速度数值积分得相对旋转 (bg=0)"""
    Rg = np.eye(3)
    for tt in np.arange(t0, t1, step):
        w = np.array([np.interp(tt, imu[:, 0], imu[:, 1 + k]) for k in range(3)])
        th = np.linalg.norm(w) * step
        if th > 1e-12:
            k_ = w / np.linalg.norm(w)
            K = np.array([[0, -k_[2], k_[1]], [k_[2], 0, -k_[0]], [-k_[1], k_[0], 0]])
            Rg = Rg @ (np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K)
    return Rg

def evolution_table(name, glog, run_dir):
    """全 pass 演化表: 每行一 pass, 关键环节指标"""
    batches = load_batches(glog)
    print("=" * 100)
    print(f"### {name}: {len(batches)} passes 演化表")
    print(f"{'pass':>4} {'|dbg|':>7} {'dbg_y':>8} {'|v0|lin':>8} {'|g|lin':>7} "
          f"{'|v0|ref':>8} {'|g|ref':>6} {'bg_y':>8} {'delta_v':>8} {'resid':>9} {'finite':>6}")
    for key in sorted(batches.keys()):
        d = batches[key]
        gb = d.get("gyro_bias"); li = d.get("linear_initial"); be = d.get("batch_end")
        if not (gb and li):
            continue
        dbg = np.array(gb["solution"])
        x = np.array(li["solution"])
        v0l = np.linalg.norm(x[0:3]); gl = np.linalg.norm(x[-3:])
        resid = li.get("normal_equation_relative_residual")
        fin = li.get("system_finite")
        if be:
            bg = np.array(be["bias_g"])
            print(f"{d['pass_id']:>4} {np.linalg.norm(dbg):>7.4f} {dbg[1]:>8.4f} {v0l:>8.3f} {gl:>7.3f} "
                  f"{be['initial_speed']:>8.3f} {be['gravity_norm']:>6.3f} {bg[1]:>8.4f} "
                  f"{(be['delta_v'] if be['delta_v'] is not None else float('nan')):>8.3f} "
                  f"{(resid if resid is not None else float('nan')):>9.2e} {fin:>6}")
        else:
            print(f"{d['pass_id']:>4} {np.linalg.norm(dbg):>7.4f} {dbg[1]:>8.4f} {v0l:>8.3f} {gl:>7.3f} "
                  f"{'-':>8} {'-':>6} {'-':>8} {'-':>8} "
                  f"{(resid if resid is not None else float('nan')):>9.2e} {fin:>6}")

def analyze(name, glog, run_dir, max_pass=4):
    batches = load_batches(glog)
    imu = load_imu(run_dir + "/legacy_logs/imu.txt")
    print("=" * 100)
    print(f"### {name}: {len(batches)} passes")
    for key in sorted(batches.keys())[:max_pass]:
        d = batches[key]
        gb = d.get("gyro_bias"); li = d.get("linear_initial"); be = d.get("batch_end")
        if not (gb and li):
            continue
        pid = d["pass_id"]
        ws, we = li["window_start"], li["window_end"]
        n = li["window_size"]
        ts, Rs, vs = frame_R_vec(li)
        x = np.array(li["solution"])
        v0 = x[0:3]; g = x[-3:]
        dbg = np.array(gb["solution"])
        dt = np.diff(ts)

        # 环节1: ICP 相对旋转 vs IMU 积分旋转 的误差
        err_deg = []
        err_vec = []
        for i in range(len(ts) - 1):
            Rrel = Rs[i].T @ Rs[i + 1]
            Rimu = imu_integrate_rot(imu, ts[i], ts[i + 1])
            e = rotvec(Rimu.T @ Rrel)          # ICP 相对 IMU 的旋转误差向量(rad)
            err_deg.append(math.degrees(np.linalg.norm(e)))
            err_vec.append(e)
        err_deg = np.array(err_deg)
        err_vec = np.array(err_vec)
        # 环节2: gyro_bias 解; 用误差向量直接折算对照
        # solve_Rot_bias_gyro: min || J*dbg - 2*vec(δq_pre^-1 q_icp) ||, J≈-0.5*dt*I
        # 等价 dbg ≈ LS拟合: -2*e/dt
        dbg_implied = -2 * np.sum(err_vec / dt[:, None], axis=0) / len(dt)
        # 环节3: 线性解
        vn = [np.linalg.norm(v) for v in vs if v is not None]
        print(f"\n-- pass {pid}  窗口 [{ws:.3f}, {we:.3f}] {n}帧 dt_mean={dt.mean()*1000:.1f}ms")
        print(f"   [环节1 ICP旋转] 逐帧ICP-IMU旋转误差: mean={err_deg.mean():.2f}° max={err_deg.max():.2f}°")
        print(f"        误差向量均值={np.array2string(err_vec.mean(0), precision=4)} rad/帧")
        print(f"   [环节2 gyro_bias] 求解delta_bg={np.array2string(dbg, precision=4)} |dbg|={np.linalg.norm(dbg):.4f} rad/s")
        print(f"        由ICP-IMU误差折算 dbg≈{np.array2string(dbg_implied, precision=4)}")
        print(f"   [环节3 线性解] |v0|={np.linalg.norm(v0):.3f} m/s v0={np.array2string(v0, precision=2)}")
        print(f"        |g|={np.linalg.norm(g):.3f} m/s² g={np.array2string(g, precision=2)}")
        print(f"        逐帧|v_i|: min={min(vn):.2f} max={max(vn):.2f} mean={np.mean(vn):.2f}")
        if be:
            print(f"   [精化后batch_end] |v0|={be['initial_speed']:.3f} |g|={be['gravity_norm']:.3f} "
                  f"bg={np.array2string(np.array(be['bias_g']), precision=4)} delta_v={be['delta_v']}")
    return batches

if __name__ == "__main__":
    b1 = analyze("174946 (跑飞)", G1, R1, max_pass=3)
    b2 = analyze("180216 (正常)", G2, R2, max_pass=3)
    evolution_table("174946 (跑飞)", G1, R1)
    evolution_table("180216 (正常)", G2, R2)