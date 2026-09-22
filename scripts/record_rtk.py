#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTK 速度录制脚本
订阅 /dji_osdk_ros/rtk_velocity (geometry_msgs/Vector3Stamped, 单位 m/s, ENU地理坐标系)
写入 Log/rtk_velocity.txt, 格式: time vx vy vz
用法: 回放 rosbag (use_sim_time) 时并行运行:
    python3 record_rtk.py [--topic /dji_osdk_ros/rtk_velocity] [--out ../Log/rtk_velocity.txt]
"""
import argparse
import os

import rospy
from geometry_msgs.msg import Vector3Stamped


def main():
    parser = argparse.ArgumentParser(description="Record RTK velocity to text file")
    parser.add_argument("--topic", type=str, default="/dji_osdk_ros/rtk_velocity")
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Log", "rtk_velocity.txt")
    parser.add_argument("--out", type=str, default=default_out)
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    f = open(out_path, "w")

    count = [0]

    def cb(msg):
        # Vector3Stamped: header.stamp + vector(x,y,z), 单位 m/s, ENU
        t = msg.header.stamp.to_sec()
        if t <= 0:  # 无时间戳时退回当前ROS时间
            t = rospy.get_time()
        f.write("%.9f %.6f %.6f %.6f\n" % (t, msg.vector.x, msg.vector.y, msg.vector.z))
        count[0] += 1
        if count[0] % 100 == 0:
            rospy.loginfo("rtk recorded %d msgs", count[0])

    rospy.init_node("record_rtk_velocity", anonymous=True)
    rospy.Subscriber(args.topic, Vector3Stamped, cb, queue_size=2000)
    rospy.loginfo("recording %s -> %s", args.topic, out_path)

    rospy.spin()
    f.close()
    rospy.loginfo("done, total %d msgs -> %s", count[0], out_path)


if __name__ == "__main__":
    main()