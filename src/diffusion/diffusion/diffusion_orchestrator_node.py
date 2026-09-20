#!/usr/bin/env python3
"""Diffusion Policy pick-and-place orchestrator for one action goal per cycle."""

from __future__ import annotations

import argparse
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from custom_msgs.action import PickAndPlace


ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
HOME = [0.0, -1.5708, 1.5708, 0.0, 1.5708, 0.0]
DIFFUSION_ENABLED_TOPIC = "/isaaclab/diffusion/enabled"
PREPARE_SERVICE = "/diffusion_ros_adapter/prepare"


class CycleCancelled(RuntimeError):
    pass


class DiffusionOrchestrator(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("diffusion_orchestrator")
        for name, value in [
            ("cycle_timeout", args.cycle_timeout),
            ("leave_timeout", args.leave_timeout),
            ("home_leave_threshold", args.home_leave_threshold),
            ("home_return_tolerance", args.home_return_tolerance),
            ("home_settle_seconds", args.home_settle_seconds),
            ("reset_settle_seconds", args.reset_settle_seconds),
            ("spawn_settle_seconds", args.spawn_settle_seconds),
            ("prepare_timeout", args.prepare_timeout),
            ("service_timeout", args.service_timeout),
        ]:
            self.declare_parameter(name, value)

        self._cycle_timeout = float(self.get_parameter("cycle_timeout").value)
        self._leave_timeout = float(self.get_parameter("leave_timeout").value)
        self._home_leave_threshold = float(self.get_parameter("home_leave_threshold").value)
        self._home_return_tolerance = float(self.get_parameter("home_return_tolerance").value)
        self._home_settle_seconds = float(self.get_parameter("home_settle_seconds").value)
        self._reset_settle = float(self.get_parameter("reset_settle_seconds").value)
        self._spawn_settle = float(self.get_parameter("spawn_settle_seconds").value)
        self._prepare_timeout = float(self.get_parameter("prepare_timeout").value)
        self._service_timeout = float(self.get_parameter("service_timeout").value)

        self._state_group = MutuallyExclusiveCallbackGroup()
        self._state_lock = threading.Lock()
        self._joint_positions: dict[str, float] = {}
        self.create_subscription(
            JointState, "/joint_states", self._joint_callback, 10, callback_group=self._state_group
        )
        self._enable_pub = self.create_publisher(Bool, DIFFUSION_ENABLED_TOPIC, 10)
        self._reset_cli = self.create_client(Trigger, "/reset")
        self._spawn_cli = self.create_client(Trigger, "/spawn_cube")
        self._prepare_cli = self.create_client(Trigger, PREPARE_SERVICE)

        self._cancel_requested = False
        self._goal_active = False
        self._goal_lock = threading.Lock()
        self._action_server = ActionServer(
            self,
            PickAndPlace,
            "/pick_and_place",
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )
        self.get_logger().info("Diffusion orchestrator ready (/pick_and_place)")

    def _joint_callback(self, msg: JointState) -> None:
        with self._state_lock:
            for name, position in zip(msg.name, msg.position):
                if name in ARM_JOINTS:
                    self._joint_positions[name] = float(position)

    def _arm_snapshot(self) -> list[float] | None:
        with self._state_lock:
            if any(name not in self._joint_positions for name in ARM_JOINTS):
                return None
            return [self._joint_positions[name] for name in ARM_JOINTS]

    def _goal_callback(self, goal_request):
        with self._goal_lock:
            if self._goal_active:
                self.get_logger().warning("Rejecting concurrent PickAndPlace goal")
                return GoalResponse.REJECT
            self._goal_active = True
            self._cancel_requested = False
        self.get_logger().info("Diffusion PickAndPlace goal accepted")
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().warning("Diffusion PickAndPlace cancel requested")
        self._cancel_requested = True
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        result = PickAndPlace.Result()
        try:
            self._publish_enabled(False)
            time.sleep(0.1)
            self._assert_not_cancelled()
            self._feedback(goal_handle, 1, "Resetting...")
            if not self._call_trigger(self._reset_cli, "reset"):
                raise RuntimeError("reset failed")
            self._sleep(self._reset_settle)
            self._feedback(goal_handle, 2, "Spawning cube...")
            if not self._call_trigger(self._spawn_cli, "spawn_cube"):
                raise RuntimeError("spawn_cube failed")
            self._sleep(self._spawn_settle)
            self._feedback(goal_handle, 3, "Preparing policy...")
            if not self._call_trigger(self._prepare_cli, "prepare", self._prepare_timeout):
                raise RuntimeError("policy prepare failed")
            self._feedback(goal_handle, 4, "Running policy...")
            self._publish_enabled(True)
            if not self._wait_leave_home():
                raise TimeoutError(f"arm did not leave home within {self._leave_timeout:.1f}s")
            if not self._wait_return_home():
                raise TimeoutError(f"arm did not return home within {self._cycle_timeout:.1f}s")
            self._publish_enabled(False)
            self._feedback(goal_handle, 11, "Cycle complete")
            goal_handle.succeed()
            result.success = True
            result.message = "Diffusion cycle completed"
        except CycleCancelled:
            self._publish_enabled(False)
            goal_handle.canceled()
            result.success = False
            result.message = "Canceled"
        except Exception as exc:
            self._publish_enabled(False)
            goal_handle.abort()
            result.success = False
            result.message = str(exc)
            self.get_logger().error(f"Diffusion cycle failed: {exc}")
        finally:
            with self._goal_lock:
                self._goal_active = False
        return result

    def _wait_leave_home(self) -> bool:
        end = time.monotonic() + self._leave_timeout
        while time.monotonic() < end:
            self._assert_not_cancelled()
            arm = self._arm_snapshot()
            if arm is not None and max(abs(value - home) for value, home in zip(arm, HOME)) > self._home_leave_threshold:
                self.get_logger().info("Arm left home")
                return True
            time.sleep(0.05)
        return False

    def _wait_return_home(self) -> bool:
        end = time.monotonic() + self._cycle_timeout
        settled_since: float | None = None
        while time.monotonic() < end:
            self._assert_not_cancelled()
            arm = self._arm_snapshot()
            now = time.monotonic()
            near_home = arm is not None and max(abs(value - home) for value, home in zip(arm, HOME)) < self._home_return_tolerance
            if near_home:
                if settled_since is None:
                    settled_since = now
                elif now - settled_since >= self._home_settle_seconds:
                    self.get_logger().info("Arm returned home and settled")
                    return True
            else:
                settled_since = None
            time.sleep(0.05)
        return False

    def _publish_enabled(self, enabled: bool) -> None:
        self._enable_pub.publish(Bool(data=enabled))

    def _feedback(self, goal_handle, step: int, status: str) -> None:
        feedback = PickAndPlace.Feedback()
        feedback.current_step = step
        feedback.status = status
        goal_handle.publish_feedback(feedback)
        self.get_logger().info(f"[{step}/11] {status}")

    def _call_trigger(self, client, name: str, timeout: float | None = None) -> bool:
        timeout = self._service_timeout if timeout is None else timeout
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"{name} service unavailable")
            return False
        future = client.call_async(Trigger.Request())
        end = time.monotonic() + timeout
        while not future.done() and time.monotonic() < end:
            self._assert_not_cancelled()
            time.sleep(0.02)
        if not future.done():
            self.get_logger().error(f"{name} service timeout")
            return False
        response = future.result()
        if response is None or not response.success:
            self.get_logger().error(f"{name} failed: {response.message if response else 'no response'}")
            return False
        return True

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._assert_not_cancelled()
            time.sleep(0.05)

    def _assert_not_cancelled(self) -> None:
        if self._cancel_requested:
            raise CycleCancelled("cancelled by user")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle-timeout", type=float, default=60.0)
    parser.add_argument("--leave-timeout", type=float, default=15.0)
    parser.add_argument("--home-leave-threshold", type=float, default=0.3)
    parser.add_argument("--home-return-tolerance", type=float, default=0.2)
    parser.add_argument("--home-settle-seconds", type=float, default=1.0)
    parser.add_argument("--reset-settle-seconds", type=float, default=1.0)
    parser.add_argument("--spawn-settle-seconds", type=float, default=1.0)
    parser.add_argument("--prepare-timeout", type=float, default=20.0)
    parser.add_argument("--service-timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    rclpy.init()
    node = DiffusionOrchestrator(parse_args())
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
