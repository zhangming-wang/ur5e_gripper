"""桥接节点 — 把 MoveIt action 转成 topic 发给 IsaacLab，等执行完成信号

运行在系统 ROS2 (Python 3.10)。
"""

import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Float64, Bool

# ---- 配置 ----
ARM_TOPIC = "/arm_controller/joint_trajectory"
GRIPPER_TOPIC = "/gripper_controller/command"
TRAJ_DONE_TOPIC = "/isaaclab/trajectory_done"
GRIPPER_DONE_TOPIC = "/isaaclab/gripper_done"
STOP_MOTION_TOPIC = "/isaaclab/stop_motion"
GENERIC_STOP_MOTION_TOPIC = "/stop_motion"
EXTRA_TIMEOUT = 6.0  # 超时兜底 (秒)


class BridgeNode(Node):
    def __init__(self):
        super().__init__("moveit_bridge")

        # IsaacLab 完成信号
        self._traj_done = threading.Event()
        self._gripper_done = threading.Event()
        self._traj_success = False
        self._gripper_success = False
        self._arm_cancel_requested = threading.Event()
        self._gripper_cancel_requested = threading.Event()
        self._busy_lock = threading.Lock()
        self._joint_state_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self.create_subscription(Bool, TRAJ_DONE_TOPIC, self._traj_done_cb, 10, callback_group=self._joint_state_group)
        self.create_subscription(
            Bool, GRIPPER_DONE_TOPIC, self._gripper_done_cb, 10, callback_group=self._joint_state_group
        )
        self.create_subscription(Bool, GENERIC_STOP_MOTION_TOPIC, self._external_stop_cb, 10)
        self.create_subscription(Bool, STOP_MOTION_TOPIC, self._stop_motion_cb, 10)

        self._arm_busy = False
        self._gripper_busy = False

        # ---- Arm ----
        self._arm_action = ActionServer(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
            goal_callback=self._arm_goal_cb,
            cancel_callback=self._arm_cancel_cb,
            execute_callback=self._arm_execute_cb,
        )
        self._arm_pub = self.create_publisher(JointTrajectory, ARM_TOPIC, 10)

        # ---- Gripper ----
        self._gripper_action = ActionServer(
            self,
            GripperCommand,
            "/robotiq_gripper_controller/gripper_cmd",
            goal_callback=self._gripper_goal_cb,
            cancel_callback=self._gripper_cancel_cb,
            execute_callback=self._gripper_execute_cb,
        )
        self._gripper_pub = self.create_publisher(Float64, GRIPPER_TOPIC, 10)
        self._stop_pub = self.create_publisher(Bool, STOP_MOTION_TOPIC, 10)

        self.get_logger().info("Bridge ready")

    # ------------------------------------------------------------------
    # Done signals from IsaacLab
    # ------------------------------------------------------------------

    def _traj_done_cb(self, msg):
        self._traj_success = bool(msg.data)
        self._traj_done.set()

    def _gripper_done_cb(self, msg):
        self._gripper_success = bool(msg.data)
        self._gripper_done.set()

    def _stop_motion_cb(self, msg):
        if msg is not None and not msg.data:
            return
        self._mark_stop()

    def _external_stop_cb(self, msg):
        if msg is not None and not msg.data:
            return
        self._request_stop()

    def _mark_stop(self):
        self._arm_cancel_requested.set()
        self._gripper_cancel_requested.set()
        self._traj_done.set()
        self._gripper_done.set()

    def _request_stop(self):
        self._mark_stop()
        self._stop_pub.publish(Bool(data=True))

    # ------------------------------------------------------------------
    # Arm
    # ------------------------------------------------------------------

    def _arm_goal_cb(self, _goal):
        with self._busy_lock:
            if self._arm_busy:
                self.get_logger().warn("Rejecting: still executing previous trajectory")
                return GoalResponse.REJECT
            self._arm_busy = True
            self._arm_cancel_requested.clear()
        return GoalResponse.ACCEPT

    def _arm_cancel_cb(self, _goal):
        self._request_stop()
        return CancelResponse.ACCEPT

    def _arm_execute_cb(self, goal_handle):
        if goal_handle.is_cancel_requested:
            with self._busy_lock:
                self._arm_busy = False
            goal_handle.canceled()
            return FollowJointTrajectory.Result(
                error_code=FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED, error_string="Canceled"
            )
        traj = goal_handle.request.trajectory
        self.get_logger().info(f"Trajectory: {len(traj.points)} pts")
        if not traj.points:
            with self._busy_lock:
                self._arm_busy = False
            goal_handle.abort()
            return FollowJointTrajectory.Result(
                error_code=FollowJointTrajectory.Result.INVALID_GOAL,
                error_string="Empty trajectory",
            )

        self._traj_done.clear()
        self._traj_success = False
        self._arm_pub.publish(traj)

        duration = 0.0
        if traj.points:
            last = traj.points[-1].time_from_start
            duration = last.sec + last.nanosec * 1e-9

        completed = self._traj_done.wait(timeout=duration + EXTRA_TIMEOUT)
        if self._arm_cancel_requested.is_set() or goal_handle.is_cancel_requested:
            with self._busy_lock:
                self._arm_busy = False
            goal_handle.canceled()
            return FollowJointTrajectory.Result(
                error_code=FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED, error_string="Canceled"
            )

        if (
            completed
            and self._traj_success
            and not self._arm_cancel_requested.is_set()
            and not goal_handle.is_cancel_requested
        ):
            self.get_logger().info("Trajectory completed")
            with self._busy_lock:
                self._arm_busy = False
            goal_handle.succeed()
            return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.SUCCESSFUL)

        self.get_logger().warn("Trajectory failed or timed out")
        with self._busy_lock:
            self._arm_busy = False
        goal_handle.abort()
        return FollowJointTrajectory.Result(
            error_code=FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED, error_string="Timeout"
        )

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    def _gripper_goal_cb(self, _goal):
        with self._busy_lock:
            if self._gripper_busy:
                self.get_logger().warn("Rejecting: still executing previous gripper command")
                return GoalResponse.REJECT
            self._gripper_busy = True
            self._gripper_cancel_requested.clear()
        return GoalResponse.ACCEPT

    def _gripper_cancel_cb(self, _goal):
        self._request_stop()
        return CancelResponse.ACCEPT

    def _gripper_execute_cb(self, goal_handle):
        target = goal_handle.request.command.position
        self.get_logger().info(f"Gripper: {target:.3f}")

        if goal_handle.is_cancel_requested:
            with self._busy_lock:
                self._gripper_busy = False
            goal_handle.canceled()
            return GripperCommand.Result(position=target, reached_goal=False)
        self._gripper_done.clear()
        self._gripper_success = False
        self._gripper_pub.publish(Float64(data=target))

        completed = self._gripper_done.wait(timeout=EXTRA_TIMEOUT)
        if completed and not self._gripper_cancel_requested.is_set() and not goal_handle.is_cancel_requested:
            if not self._gripper_success:
                with self._busy_lock:
                    self._gripper_busy = False
                goal_handle.abort()
                return GripperCommand.Result(position=target, reached_goal=False)
            with self._busy_lock:
                self._gripper_busy = False
            goal_handle.succeed()
            return GripperCommand.Result(position=target, reached_goal=True)

        if self._gripper_cancel_requested.is_set() or goal_handle.is_cancel_requested:
            with self._busy_lock:
                self._gripper_busy = False
            goal_handle.canceled()
            return GripperCommand.Result(position=target, reached_goal=False)

        with self._busy_lock:
            self._gripper_busy = False
        goal_handle.abort()
        return GripperCommand.Result(position=target, reached_goal=False)


def main():
    rclpy.init()
    node = BridgeNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
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
