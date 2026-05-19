#!/usr/bin/env python3
# Deterministic scenario evaluator. Drives a trained policy through a
# fixed list of (start_pose, goal_pose) scenarios with the gz-sim clock
# reset to 0 at the start of each one, so every algorithm sees the
# obstacle animator replay the same trajectory from t=0. Output is a
# per-scenario CSV plus a printed summary table.

import copy
import csv
import math
import os
import sys
import time

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Bool, Empty
from visualization_msgs.msg import Marker, MarkerArray
from turtlebot3_msgs.srv import DrlStep
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from ros_gz_interfaces.msg import Entity, WorldControl

from ..common import utilities as util
from ..common.settings import ENABLE_STACKING, ENABLE_VISUAL
from ..common.storagemanager import StorageManager
from ..drl_agent.dqn import DQN
from ..drl_agent.ddpg import DDPG
from ..drl_agent.td3 import TD3
from ..drl_agent.redq import REDQ


OUTCOME_STR = {
    1: 'SUCCESS',
    2: 'COLL_WALL',
    3: 'COLL_OBST',
    4: 'TIMEOUT',
    5: 'TUMBLE',
    0: 'UNKNOWN',
}

# Text-color (r, g, b) per outcome. Start dot is always green, goal dot
# is always red; only the status text color reflects success / failure.
TEXT_RGB = {
    'RUNNING':   (1.0, 1.0, 0.0),     # yellow while in-flight
    'SUCCESS':   (0.1, 0.9, 0.1),     # green
    'COLL_WALL': (1.0, 0.1, 0.1),     # red
    'COLL_OBST': (1.0, 0.5, 0.0),     # orange
    'TIMEOUT':   (0.6, 0.6, 0.6),     # gray
    'TUMBLE':    (0.7, 0.0, 1.0),     # purple
    'UNKNOWN':   (1.0, 1.0, 1.0),     # white
}


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _spin_until(node, future, timeout_sec=5.0):
    deadline = time.perf_counter() + timeout_sec
    while rclpy.ok() and not future.done():
        if time.perf_counter() > deadline:
            return None
        rclpy.spin_once(node, timeout_sec=0.05)
    return future.result()


def load_scenarios():
    share = get_package_share_directory('turtlebot3_drl')
    path = os.path.join(share, 'eval', 'scenarios.yaml')
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data['scenarios']


class EvalRunner(Node):
    def __init__(self, algorithm, load_session, load_episode):
        super().__init__(f'{algorithm}_eval_runner')
        self.algorithm = algorithm
        self.load_session = load_session
        self.episode = int(load_episode)
        self.device = util.check_gpu()
        self.sim_speed = util.get_simulation_speed(util.stage)
        print(f"[eval] stage={util.stage} algo={algorithm} session={load_session} eps={self.episode}")

        if algorithm == 'dqn':
            self.model = DQN(self.device, self.sim_speed)
        elif algorithm == 'ddpg':
            self.model = DDPG(self.device, self.sim_speed)
        elif algorithm == 'td3':
            self.model = TD3(self.device, self.sim_speed)
        elif algorithm == 'redq':
            self.model = REDQ(self.device, self.sim_speed)
        else:
            quit(f"invalid algorithm '{algorithm}', choose dqn/ddpg/td3/redq")

        self.sm = StorageManager(algorithm, load_session, self.episode, self.device, util.stage)
        del self.model
        self.model = self.sm.load_model()
        self.model.device = self.device
        self.sm.load_weights(self.model.networks)
        print(f"[eval] loaded {load_session} eps={self.episode}: {self.model.get_model_parameters()}")

        self.step_comm_client = self.create_client(DrlStep, 'step_comm')

        world_name = f'drl_stage{util.stage}'
        self.gazebo_control = self.create_client(ControlWorld, f'/world/{world_name}/control')
        self.set_pose_client = self.create_client(SetEntityPose, f'/world/{world_name}/set_pose')

        # Latched goal publisher so the env sees the goal even if it
        # subscribes after we publish.
        goal_qos = QoSProfile(depth=1,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL,
                              reliability=ReliabilityPolicy.RELIABLE)
        self.goal_pub = self.create_publisher(Pose, 'drl_goal_pose', goal_qos)

        # Tell drl_gazebo to stop generating random goals on task_succeed/fail.
        eval_qos = QoSProfile(depth=1,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL,
                              reliability=ReliabilityPolicy.RELIABLE)
        self.eval_active_pub = self.create_publisher(Bool, '/eval_mode_active', eval_qos)
        self.eval_active_pub.publish(Bool(data=True))

        # Phase-reset bridge to the obstacle_animator gz plugin. We
        # *don't* reset the gz-sim clock between scenarios — that
        # would tear down the dynamically-spawned robot's joints and
        # the DiffDrive plugin would stop driving the wheels. Each
        # Empty msg tells every animator instance to treat the current
        # simTime as t=0 in its keyframe cycle.
        self.phase_reset_pub = self.create_publisher(
            Empty, '/obstacle_phase_reset', 1)

        # RViz overlay: start arrow + scenario/outcome text marker.
        marker_qos = QoSProfile(depth=1,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                reliability=ReliabilityPolicy.RELIABLE)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/eval_markers', marker_qos)

        # cmd_vel zero pulse — keeps the robot from coasting between
        # scenarios since SetEntityPose teleports the pose but does not
        # reset wheel joint velocities.
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 1)
        # Give latched pub a moment to deliver and subscribers a moment to register.
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.05)

    # ----- gz-sim control helpers ------------------------------------

    def _call_control(self, pause=None, timeout=5.0):
        req = ControlWorld.Request()
        ctl = WorldControl()
        if pause is not None:
            ctl.pause = pause
        req.world_control = ctl
        while not self.gazebo_control.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('ControlWorld service not available, waiting...')
        fut = self.gazebo_control.call_async(req)
        _spin_until(self, fut, timeout)

    def _set_robot_pose(self, x, y, yaw, timeout=5.0):
        req = SetEntityPose.Request()
        req.entity = Entity(name='burger', type=Entity.MODEL)
        p = Pose()
        p.position.x = float(x)
        p.position.y = float(y)
        p.position.z = 0.01
        qx, qy, qz, qw = yaw_to_quat(float(yaw))
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
        req.pose = p
        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('SetEntityPose service not available, waiting...')
        fut = self.set_pose_client.call_async(req)
        _spin_until(self, fut, timeout)

    def _publish_goal(self, x, y):
        p = Pose()
        p.position.x = float(x)
        p.position.y = float(y)
        p.position.z = 0.0
        p.orientation.w = 1.0
        self.goal_pub.publish(p)

    def _publish_eval_markers(self, sc, idx, total, status, duration=None):
        tr, tg, tb = TEXT_RGB.get(status, TEXT_RGB['UNKNOWN'])
        now = self.get_clock().now().to_msg()

        def _dot(marker_id, x, y, rgb):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'eval'
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.20
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 0.95
            return m

        start_dot = _dot(0, sc['start']['x'], sc['start']['y'], (0.1, 0.9, 0.1))
        goal_dot  = _dot(1, sc['goal']['x'],  sc['goal']['y'],  (1.0, 1.0, 1.0))

        text = Marker()
        text.header.frame_id = 'map'
        text.header.stamp = now
        text.ns = 'eval'
        text.id = 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        # Fixed world anchor along the camera's image-bottom direction
        # (default orbit view: yaw=π, pitch=1.1) so the label always
        # sits at the bottom of the RViz viewport.
        text.pose.position.x = -2.8
        text.pose.position.y = 0.0
        text.pose.position.z = 0.3
        text.pose.orientation.w = 1.0
        text.scale.z = 0.35
        text.color.r = tr
        text.color.g = tg
        text.color.b = tb
        text.color.a = 1.0
        algo = self.algorithm.upper()
        if status == 'RUNNING':
            text.text = f"{algo}  {idx}/{total}"
        elif duration is not None:
            text.text = f"{algo}  {idx}/{total}  {duration:.1f}s  {status}"
        else:
            text.text = f"{algo}  {idx}/{total}  {status}"

        self.marker_pub.publish(MarkerArray(markers=[start_dot, goal_dot, text]))

    def _step(self, action, prev_action):
        req = DrlStep.Request()
        req.action = action
        req.previous_action = prev_action
        while not self.step_comm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('step_comm service not available, waiting...')
        fut = self.step_comm_client.call_async(req)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if fut.done():
                res = fut.result()
                return res.state, res.reward, res.done, res.success, res.distance_traveled

    # ----- scenario setup --------------------------------------------

    def setup_scenario(self, sc, idx, total):
        # Pause sim so teleport + goal publish land before physics
        # steps. Tell the obstacle animators to restart their phase
        # cycle from "now"; teleport the robot; publish the goal;
        # then unpause. The sim clock keeps running monotonically,
        # which preserves the dynamically-spawned robot's joints.
        self._call_control(pause=True)
        self.phase_reset_pub.publish(Empty())
        self._set_robot_pose(sc['start']['x'], sc['start']['y'], sc['start']['yaw'])
        self._publish_goal(sc['goal']['x'], sc['goal']['y'])
        self._publish_eval_markers(sc, idx, total, 'RUNNING')
        # Drain any pending callbacks so the env's goal_pose_callback
        # fires while still paused.
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.02)
        self._call_control(pause=False)
        # Brief settle so the env receives clock + odom under the new
        # pose before the first step_comm call.
        time.sleep(0.5)

    # ----- episode loop ----------------------------------------------

    def run_episode(self, sc):
        # First step_comm with empty action = init_episode in env code.
        state, _, _, _, _ = self._step([], [0.0, 0.0])
        if ENABLE_STACKING:
            frame_buffer = [0.0] * (self.model.state_size * self.model.stack_depth * self.model.frame_skip)
            state = [0.0] * (self.model.state_size * (self.model.stack_depth - 1)) + list(state)

        action_past = [0.0, 0.0]
        step_count = 0
        reward_sum = 0.0
        episode_start = time.perf_counter()
        outcome = 0
        distance_traveled = 0.0

        while rclpy.ok():
            action = self.model.get_action(state, False, step_count, ENABLE_VISUAL)
            action_current = action
            if self.algorithm == 'dqn':
                action_current = self.model.possible_actions[action]
            next_state, reward, done, succ, dist = self._step(action_current, action_past)
            action_past = copy.deepcopy(action_current)
            reward_sum += reward
            if ENABLE_STACKING:
                frame_buffer = frame_buffer[self.model.state_size:] + list(next_state)
                next_state = []
                for depth in range(self.model.stack_depth):
                    start = self.model.state_size * (self.model.frame_skip - 1) + (self.model.state_size * self.model.frame_skip * depth)
                    next_state += frame_buffer[start: start + self.model.state_size]
            state = copy.deepcopy(next_state)
            step_count += 1
            if done:
                outcome = succ
                distance_traveled = dist
                break
            time.sleep(self.model.step_time)

        duration = time.perf_counter() - episode_start
        # Stop the robot before the next scenario. Publish a zero
        # cmd_vel and let the DiffDrive plugin act on it for a few
        # physics steps before we freeze the world — otherwise the
        # wheels keep spinning across the teleport into the next start.
        self.cmd_vel_pub.publish(Twist())
        time.sleep(0.3)
        self.cmd_vel_pub.publish(Twist())
        self._call_control(pause=True)
        return outcome, step_count, reward_sum, distance_traveled, duration

    # ----- top-level --------------------------------------------------

    def run(self):
        scenarios = load_scenarios()
        out_path, comparison_path = self._open_eval_log()
        print(f"[eval] writing per-scenario log to {out_path}")
        print(f"[eval] {'#':<6} {'outcome':<10} {'steps':>6} {'dist':>7} {'dur':>7}")

        results = []
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['scenario_id', 'tag', 'difficulty',
                             'start_x', 'start_y', 'start_yaw',
                             'goal_x', 'goal_y',
                             'outcome', 'outcome_str',
                             'steps', 'duration_sec', 'distance_m', 'reward_sum'])
            total = len(scenarios)
            for idx, sc in enumerate(scenarios, start=1):
                self.setup_scenario(sc, idx, total)
                outcome, steps, reward_sum, dist, dur = self.run_episode(sc)
                self._publish_eval_markers(
                    sc, idx, total,
                    OUTCOME_STR.get(outcome, 'UNKNOWN'), duration=dur)
                results.append((sc, outcome, steps, reward_sum, dist, dur))
                writer.writerow([
                    sc['id'], sc.get('tag', ''), sc.get('difficulty', ''),
                    sc['start']['x'], sc['start']['y'], sc['start']['yaw'],
                    sc['goal']['x'], sc['goal']['y'],
                    outcome, OUTCOME_STR.get(outcome, str(outcome)),
                    steps, f"{dur:.3f}", f"{dist:.3f}", f"{reward_sum:.2f}",
                ])
                f.flush()
                print(f"[eval] {idx:>2}/{total:<3} "
                      f"{OUTCOME_STR.get(outcome,str(outcome)):<10} {steps:>6} {dist:>7.2f} {dur:>7.2f}")

        self._print_summary(results)
        self._append_comparison(comparison_path, results)
        print(f"[eval] appended summary row to {comparison_path}")

    def _open_eval_log(self):
        # Per-run CSV lives next to the model checkpoint, like _test_*.txt.
        datetime = time.strftime("%Y%m%d-%H%M%S")
        fname = f"_eval_stage{util.stage}_eps{self.episode}_{datetime}.csv"
        out_path = os.path.join(self.sm.session_dir, fname)
        # Top-level cross-algo aggregator sits at the model root so a
        # td3 / ddpg / redq run all append to the same file.
        model_root = os.path.join(os.getenv('DRLNAV_BASE_PATH'),
                                  'src', 'turtlebot3_drl', 'model')
        comparison_path = os.path.join(model_root, '__eval_comparison.csv')
        return out_path, comparison_path

    def _append_comparison(self, comparison_path, results):
        n = len(results)
        succ = sum(1 for (_, o, *_rest) in results if o == 1)
        steps_succ = [s for (_, o, s, *_r) in results if o == 1]
        dist_succ  = [d for (_, o, _s, _r, d, _du) in results if o == 1]
        dur_succ   = [du for (_, o, _s, _r, _d, du) in results if o == 1]
        header = ['datetime', 'algorithm', 'session', 'episode',
                  'scenarios_passed', 'total', 'success_rate',
                  'mean_steps_success', 'mean_dist_success', 'mean_dur_success']
        row = [
            time.strftime("%Y%m%d-%H%M%S"), self.algorithm, self.load_session, self.episode,
            succ, n, f"{succ/n:.4f}" if n else "0",
            f"{np.mean(steps_succ):.2f}" if steps_succ else '',
            f"{np.mean(dist_succ):.3f}" if dist_succ else '',
            f"{np.mean(dur_succ):.3f}" if dur_succ else '',
        ]
        write_header = not os.path.exists(comparison_path)
        os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
        with open(comparison_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)

    def _print_summary(self, results):
        n = len(results)
        succ = sum(1 for (_, o, *_rest) in results if o == 1)
        by_diff = {}
        for sc, o, *_rest in results:
            d = sc.get('difficulty', 'unknown')
            slot = by_diff.setdefault(d, [0, 0])
            slot[1] += 1
            if o == 1:
                slot[0] += 1
        print(f"[eval] ------ summary ------")
        print(f"[eval] algo={self.algorithm} session={self.load_session} eps={self.episode}")
        print(f"[eval] success: {succ}/{n} ({100.0*succ/max(n,1):.1f}%)")
        for d, (ok, tot) in sorted(by_diff.items()):
            print(f"[eval]   {d:<8} {ok}/{tot} ({100.0*ok/max(tot,1):.1f}%)")


def main(args=sys.argv[1:]):
    if len(args) < 3:
        quit("usage: ros2 run turtlebot3_drl eval_agent <algo> <load_session> <load_episode>")
    rclpy.init(args=args)
    runner = EvalRunner(args[0], args[1], args[2])
    try:
        runner.run()
    finally:
        runner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
