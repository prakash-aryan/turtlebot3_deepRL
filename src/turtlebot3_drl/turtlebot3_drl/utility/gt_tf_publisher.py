#!/usr/bin/env python3
# Broadcasts map->odom as world_T_base * (odom_T_base)^-1, computed from
# /ground_truth_odom (world frame) and /odom (DiffDrive integration).

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import tf_transformations
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


def _pose_to_mat(pose):
    t = [pose.position.x, pose.position.y, pose.position.z]
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    m = tf_transformations.translation_matrix(t)
    m = m @ tf_transformations.quaternion_matrix(q)
    return m


def _mat_to_transform(m, parent, child, stamp):
    t = tf_transformations.translation_from_matrix(m)
    q = tf_transformations.quaternion_from_matrix(m)
    ts = TransformStamped()
    ts.header.stamp = stamp
    ts.header.frame_id = parent
    ts.child_frame_id = child
    ts.transform.translation.x = t[0]
    ts.transform.translation.y = t[1]
    ts.transform.translation.z = t[2]
    ts.transform.rotation.x = q[0]
    ts.transform.rotation.y = q[1]
    ts.transform.rotation.z = q[2]
    ts.transform.rotation.w = q[3]
    return ts


class GroundTruthTfPublisher(Node):
    def __init__(self):
        super().__init__('drl_gt_tf_publisher')
        self.tf_br = TransformBroadcaster(self)
        self.static_br = StaticTransformBroadcaster(self)

        s = TransformStamped()
        s.header.stamp = self.get_clock().now().to_msg()
        s.header.frame_id = 'map'
        s.child_frame_id = 'world'
        s.transform.rotation.w = 1.0
        self.static_br.sendTransform(s)

        self.latest_odom = None
        self.create_subscription(Odometry, 'odom', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, 'ground_truth_odom', self.gt_cb, qos_profile_sensor_data)

    def odom_cb(self, msg):
        self.latest_odom = msg

    def gt_cb(self, msg):
        if self.latest_odom is None:
            return
        world_T_base = _pose_to_mat(msg.pose.pose)
        odom_T_base = _pose_to_mat(self.latest_odom.pose.pose)
        try:
            base_T_odom = tf_transformations.inverse_matrix(odom_T_base)
        except Exception:
            return
        map_T_odom = world_T_base @ base_T_odom
        self.tf_br.sendTransform(
            _mat_to_transform(map_T_odom, 'map', 'odom', msg.header.stamp))


def main():
    rclpy.init()
    node = GroundTruthTfPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
