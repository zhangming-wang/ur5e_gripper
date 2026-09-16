#!/usr/bin/env python3
"""ACT pick-and-place 编排节点 — 一个 goal 跑一轮策略。

流程:
  step 1  reset      → enable(False) + /reset
  step 2  spawn      → /spawn_cube
  step 3  prepare    → /act_ros_adapter/prepare（预热 chunk 缓冲）
  step 4  running    → enable(True)
          判定: 先离开 home，再回到 home 并静止一段时间
  step 11 done       → enable(False) → succeed
  超时/取消          → enable(False) → abort / canceled

与 MoveIt 版编排节点提供同名 ``/pick_and_place``，两种模式接口一致。
"""

from __future__ import annotations

import argparse
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
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
# 与训练数据终点 / control_node.init_pos 一致
HOME = [0.0, -1.5708, 1.5708, 0.0, 1.5708, 0.0]
ACT_ENABLED_TOPIC = "/isaaclab/act/enabled"
PREPARE_SERVICE = "/act_ros_adapter/prepare"


class CycleCancelled(RuntimeError):
    pass


class ActOrchestrator(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("act_orchestrator")

        # declare_parameter 的默认值来自 CLI；--ros-args -p 仍可覆盖
        self.declare_parameter("cycle_timeout", args.cycle_timeout)
        self.declare_parameter("leave_timeout", args.leave_timeout)
        self.declare_parameter("home_leave_threshold", args.home_leave_threshold)
        self.declare_parameter("home_return_tolerance", args.home_return_tolerance)
        self.declare_parameter("home_settle_seconds", args.home_settle_seconds)
        self.declare_parameter("reset_settle_seconds", args.reset_settle_seconds)
        self.declare_parameter("spawn_settle_seconds", args.spawn_settle_seconds)
        self.declare_parameter("prepare_timeout", args.prepare_timeout)
        self.declare_parameter("service_timeout", args.service_timeout)

        self._cycle_timeout = float(self.get_parameter("cycle_timeout").value)
        self._leave_timeout = float(self.get_parameter("leave_timeout").value)
        self._home_leave_threshold = float(self.get_parameter("home_leave_threshold").value)
        self._home_return_tolerance = float(self.get_parameter("home_return_tolerance").value)
        self._home_settle_seconds = float(self.get_parameter("home_settle_seconds").value)
        self._reset_settle = float(self.get_parameter("reset_settle_seconds").value)
        self._spawn_settle = float(self.get_parameter("spawn_settle_seconds").value)
        self._prepare_timeout = float(self.get_parameter("prepare_timeout").value)
        self._service_timeout = float(self.get_parameter("service_timeout").value)

        # /joint_states 放在独立回调组：action 的 execute 回调会阻塞执行器线程，
        # 同组互斥会导致订阅被饿死（与 bridge_node 相同的处理）。
        self._state_group = MutuallyExclusiveCallbackGroup()
        self._state_lock = threading.Lock()
        self._joint_positions: dict[str, float] = {}
        self.create_subscription(
            JointState, "/joint_states", self._joint_callback, 10, callback_group=self._state_group
        )

        self._enable_pub = self.create_publisher(Bool, ACT_ENABLED_TOPIC, 10)

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
        self.get_logger().info("ACT orchestrator ready (/pick_and_place)")

    # ------------------------------------------------------------------
    # 关节状态
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Action 回调
    # ------------------------------------------------------------------
    def _goal_callback(self, goal_request):
        with self._goal_lock:
            if self._goal_active:
                self.get_logger().warning("Rejecting concurrent PickAndPlace goal")
                return GoalResponse.REJECT
            self._goal_active = True
            self._cancel_requested = False
        self.get_logger().info("ACT PickAndPlace goal accepted")
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().warning("ACT PickAndPlace cancel requested")
        self._cancel_requested = True
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        result = PickAndPlace.Result()
        try:
            self._publish_enabled(False)
            time.sleep(0.1)
            self._assert_not_cancelled()

            self._publish_feedback(goal_handle, 1, "Resetting...")
            if not self._call_trigger(self._reset_cli, "reset"):
                raise RuntimeError("reset failed")
            self._sleep(self._reset_settle)

            self._publish_feedback(goal_handle, 2, "Spawning cube...")
            if not self._call_trigger(self._spawn_cli, "spawn_cube"):
                raise RuntimeError("spawn_cube failed")
            self._sleep(self._spawn_settle)

            self._publish_feedback(goal_handle, 3, "Preparing policy...")
            if not self._call_trigger(self._prepare_cli, "prepare", timeout=self._prepare_timeout):
                raise RuntimeError("policy prepare failed")

            self._publish_feedback(goal_handle, 4, "Running policy...")
            self._publish_enabled(True)

            if not self._wait_leave_home():
                raise TimeoutError(f"arm did not leave home within {self._leave_timeout:.1f}s")
            if not self._wait_return_home():
                raise TimeoutError(f"arm did not return home within {self._cycle_timeout:.1f}s")

            self._publish_enabled(False)
            self._publish_feedback(goal_handle, 11, "Cycle complete")
            goal_handle.succeed()
            result.success = True
            result.message = "ACT cycle completed"
            self.get_logger().info("ACT cycle completed")
        except CycleCancelled:
            self._publish_enabled(False)
            goal_handle.canceled()
            result.success = False
            result.message = "Canceled"
            self.get_logger().info("ACT cycle canceled")
        except Exception as exc:
            self._publish_enabled(False)
            goal_handle.abort()
            result.success = False
            result.message = str(exc)
            self.get_logger().error(f"ACT cycle failed: {exc}")
        finally:
            with self._goal_lock:
                self._goal_active = False
        return result

    # ------------------------------------------------------------------
    # 完成判定
    # ------------------------------------------------------------------
    def _wait_leave_home(self) -> bool:
        end = time.monotonic() + self._leave_timeout
        while time.monotonic() < end:
            self._assert_not_cancelled()
            arm = self._arm_snapshot()
            if arm is not None and max(
                abs(value - home) for value, home in zip(arm, HOME)
            ) > self._home_leave_threshold:
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
            near_home = arm is not None and max(
                abs(value - home) for value, home in zip(arm, HOME)
            ) < self._home_return_tolerance
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

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _publish_enabled(self, enabled: bool) -> None:
        self._enable_pub.publish(Bool(data=enabled))

    def _publish_feedback(self, goal_handle, step: int, status: str) -> None:
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
            message = response.message if response else "no response"
            self.get_logger().error(f"{name} failed: {message}")
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
    parser.add_argument("--prepare-timeout", type=float, default=15.0)
    parser.add_argument("--service-timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = ActOrchestrator(args)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
