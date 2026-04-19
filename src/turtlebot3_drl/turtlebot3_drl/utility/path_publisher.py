#!/usr/bin/env python3
# Subscribes to /ground_truth_odom and publishes an accumulated nav_msgs/Path
# on /robot_path. The trail is cleared whenever two consecutive samples are
# more than `jump_threshold` meters apart.

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped


class PathPublisher(Node):
    def __init__(self):
        super().__init__('drl_path_publisher')
        self.max_poses = 2000
        self.jump_threshold = 0.5
        self.min_step = 0.02

        self.path = Path()
        self.path.header.frame_id = 'map'
        self.last_x = None
        self.last_y = None

        self.create_subscription(
            Odometry, 'ground_truth_odom', self.odom_cb, qos_profile_sensor_data)
        self.pub = self.create_publisher(Path, 'robot_path', 10)

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.last_x is not None:
            dx = x - self.last_x
            dy = y - self.last_y
            dist = math.hypot(dx, dy)
            if dist > self.jump_threshold:
                self.path.poses.clear()
            elif dist < self.min_step:
                return

        self.last_x = x
        self.last_y = y

        ps = PoseStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = 'map'
        ps.pose = msg.pose.pose
        self.path.poses.append(ps)
        if len(self.path.poses) > self.max_poses:
            del self.path.poses[: len(self.path.poses) - self.max_poses]

        self.path.header.stamp = msg.header.stamp
        self.pub.publish(self.path)


def main():
    rclpy.init()
    node = PathPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
