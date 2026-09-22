#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_rtk.py — 从 bag 中离线提取 RTK 速度序列(伪真值), 供初始化对比分析配对使用。

用法:
  python3 extract_rtk.py <bag路径> [-o 输出csv] [--topic /dji_osdk_ros/rtk_velocity] [--max-pairs N]

输出: CSV, 列: t,vx,vy,vz,speed
  t     — RTK 消息时间戳(秒, 取 header.stamp, 无则取 ROS 时间)
  vx..vz— RTK 速度(ENU, m/s)
  speed — 速度模长

设计依据: .joycode/specs/dli-fastlio2-init-state-comparison/design.md 2.4 节
"""
import argparse
import csv
import os
import sys

DEFAULT_TOPIC = "/dji_osdk_ros/rtk_velocity"
DEFAULT_OUT = "rtk_velocity.csv"


def extract(bag_path, topic, out_path):
    try:
        import rosbag
    except ImportError:
        print("[extract_rtk] 错误: 无法导入 rosbag, 请在 ROS 环境下运行(source setup.bash)",
              file=sys.stderr)
        return 1

    if not os.path.isfile(bag_path):
        print("[extract_rtk] 错误: bag 不存在: %s" % bag_path, file=sys.stderr)
        return 1

    rows = []
    with rosbag.Bag(bag_path, "r") as bag:
        # 话题存在性检查
        topics = bag.get_type_and_topic_info()[1]
        if topic not in topics:
            print("[extract_rtk] 错误: bag 中不存在话题 %s, 现有话题:" % topic,
                  file=sys.stderr)
            for t, info in topics.items():
                print("  %s (%s, %d msgs)" % (t, info.msg_type, info.message_count),
                      file=sys.stderr)
            return 1

        for _, msg, bag_t in bag.read_messages(topics=[topic]):
            # 优先 header.stamp; 缺失时退回 bag 记录时间并给出警告
            stamp = msg.header.stamp.to_sec()
            if stamp > 0:
                t = stamp
            else:
                print("[extract_rtk] 警告: RTK 消息无 header.stamp, 退回 bag 记录时间",
                      file=sys.stderr)
                t = bag_t.to_sec()
            vx, vy, vz = msg.vector.x, msg.vector.y, msg.vector.z
            rows.append((t, vx, vy, vz, (vx * vx + vy * vy + vz * vz) ** 0.5))

    rows.sort(key=lambda r: r[0])
    with open(out_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["t", "vx", "vy", "vz", "speed"])
        for r in rows:
            w.writerow(["%.9f" % r[0], "%.9f" % r[1], "%.9f" % r[2], "%.9f" % r[3],
                        "%.9f" % r[4]])
    print("[extract_rtk] 提取 %d 条 RTK 速度样本 -> %s" % (len(rows), out_path))
    if rows:
        print("[extract_rtk] 时间范围: [%.3f, %.3f] s" % (rows[0][0], rows[-1][0]))
    return 0


def main():
    p = argparse.ArgumentParser(description="离线提取 RTK 速度序列")
    p.add_argument("bag", help="bag 文件路径")
    p.add_argument("-o", "--output", default=DEFAULT_OUT, help="输出 CSV 路径")
    p.add_argument("--topic", default=DEFAULT_TOPIC, help="RTK 速度话题")
    args = p.parse_args()
    sys.exit(extract(args.bag, args.topic, args.output))


if __name__ == "__main__":
    main()