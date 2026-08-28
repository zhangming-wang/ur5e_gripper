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
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Bool, UInt8

# ---- 配置 ----
ARM_TOPIC = "/arm_controller/joint_trajectory"
GRIPPER_TOPIC = "/gripper_controller/command"
TRAJ_DONE_TOPIC = "/isaaclab/trajectory_done"
GRIPPER_DONE_TOPIC = "/isaaclab/gripper_done"
STOP_MOTION_TOPIC = "/isaaclab/stop_motion"
GENERIC_STOP_MOTION_TOPIC = "/stop_motion"
SIM_TIME_TOPIC = "/isaaclab/sim_time"

# 夹爪完成结果码（与 control_node 保持一致）
GRIPPER_RESULT_FAILED = 0
GRIPPER_RESULT_REACHED = 1
GRIPPER_RESULT_STALLED = 2

GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"

# Bridge waits for controller results in simulation time. Wall clock is only a
# liveness safeguard so a paused/crashed simulator cannot block an action forever.
SIM_STALL_TIMEOUT = 10.0
ARM_SETTLE_BUDGET = 5.0
GRIPPER_SETTLE_BUDGET = 5.0
SIM_BUDGET_MARGIN = 2.0
WALL_SAFETY_TIMEOUT = 55.0
STOP_ACK_TIMEOUT = 3.0


class BridgeNode(Node):
    def __init__(self):
        super().__init__("moveit_bridge")

        # IsaacLab 完成信号
        self._traj_done = threading.Event()
        self._gripper_done = threading.Event()
        self._traj_success = False
        self._gripper_result = GRIPPER_RESULT_FAILED
        self._arm_cancel_requested = threading.Event()
        self._gripper_cancel_requested = threading.Event()
        self._busy_lock = threading.Lock()
        self._sim_time = 0.0
        self._sim_time_wall = time.monotonic()
        self._sim_time_lock = threading.Lock()
        self._latest_joint_state = {}
        self._joint_state_lock = threading.Lock()
        self._joint_state_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self.create_subscription(Bool, TRAJ_DONE_TOPIC, self._traj_done_cb, 10, callback_group=self._joint_state_group)
        self.create_subscription(
            UInt8, GRIPPER_DONE_TOPIC, self._gripper_done_cb, 10, callback_group=self._joint_state_group
        )
        self.create_subscription(Bool, GENERIC_STOP_MOTION_TOPIC, self._external_stop_cb, 10)
        self.create_subscription(Bool, STOP_MOTION_TOPIC, self._stop_motion_cb, 10)
        self.create_subscription(Float64, SIM_TIME_TOPIC, self._sim_time_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)

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
        self._gripper_result = int(msg.data)
        self._gripper_done.set()

    def _joint_state_cb(self, msg):
        with self._joint_state_lock:
            positions = list(msg.position)
            efforts = list(msg.effort) if msg.effort else [0.0] * len(positions)
            for name, pos, effort in zip(msg.name, positions, efforts):
                self._latest_joint_state[name] = (float(pos), float(effort))

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
        self._gripper_result = GRIPPER_RESULT_FAILED
        self._gripper_done.set()

    def _request_stop(self):
        self._mark_stop()
        self._stop_pub.publish(Bool(data=True))

    def _sim_time_cb(self, msg):
        with self._sim_time_lock:
            self._sim_time = float(msg.data)
            self._sim_time_wall = time.monotonic()

    def _get_sim_time(self):
        with self._sim_time_lock:
            return self._sim_time, self._sim_time_wall

    def _wait_for_completion(self, done_event, start_sim, sim_budget, cancel_event):
        """Wait for a controller result while simulation time is progressing."""
        deadline = time.monotonic() + WALL_SAFETY_TIMEOUT
        last_progress_sim = start_sim
        last_progress_wall = time.monotonic()
        while not done_event.is_set():
            if cancel_event.is_set():
                return "cancelled"
            now = time.monotonic()
            if now > deadline:
                return "wall_timeout"
            sim_time, sim_time_wall = self._get_sim_time()
            if sim_time > last_progress_sim + 1e-9:
                last_progress_sim = sim_time
                last_progress_wall = sim_time_wall
            if now - last_progress_wall > SIM_STALL_TIMEOUT:
                return "sim_stalled"
            if sim_time - start_sim > sim_budget:
                return "sim_budget"
            done_event.wait(timeout=0.05)
        return "done"

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
        start_sim, _ = self._get_sim_time()
        self._arm_pub.publish(traj)

        last = traj.points[-1].time_from_start
        duration = last.sec + last.nanosec * 1e-9
        reason = self._wait_for_completion(
            self._traj_done,
            start_sim,
            duration + ARM_SETTLE_BUDGET + SIM_BUDGET_MARGIN,
            self._arm_cancel_requested,
        )

        if self._arm_cancel_requested.is_set() or goal_handle.is_cancel_requested:
            with self._busy_lock:
                self._arm_busy = False
            goal_handle.canceled()
            return FollowJointTrajectory.Result(
                error_code=FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED, error_string="Canceled"
            )

        if reason == "done":
            if self._traj_success:
                self.get_logger().info("Trajectory completed")
                with self._busy_lock:
                    self._arm_busy = False
                goal_handle.succeed()
                return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.SUCCESSFUL)
            self.get_logger().warn("Trajectory failed (controller reported failure)")
            with self._busy_lock:
                self._arm_busy = False
            goal_handle.abort()
            return FollowJointTrajectory.Result(
                error_code=FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED, error_string="Execution failed"
            )

        self.get_logger().warn(f"Trajectory timed out ({reason}); stopping control")
        self._stop_pub.publish(Bool(data=True))
        self._traj_done.clear()
        self._traj_done.wait(timeout=STOP_ACK_TIMEOUT)
        with self._busy_lock:
            self._arm_busy = False
        goal_handle.abort()
        return FollowJointTrajectory.Result(
            error_code=FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED, error_string=f"Timeout ({reason})"
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
        self._gripper_result = GRIPPER_RESULT_FAILED
        start_sim, _ = self._get_sim_time()
        self._gripper_pub.publish(Float64(data=target))

        reason = self._wait_for_completion(
            self._gripper_done,
            start_sim,
            GRIPPER_SETTLE_BUDGET + SIM_BUDGET_MARGIN,
            self._gripper_cancel_requested,
        )

        def _measured_state():
            with self._joint_state_lock:
                info = self._latest_joint_state.get(GRIPPER_JOINT)
            if info is None:
                return target, 0.0
            return info[0], info[1]

        if self._gripper_cancel_requested.is_set() or goal_handle.is_cancel_requested:
            with self._busy_lock:
                self._gripper_busy = False
            goal_handle.canceled()
            position, effort = _measured_state()
            return GripperCommand.Result(position=position, effort=effort, reached_goal=False)

        if reason == "done":
            position, effort = _measured_state()
            if self._gripper_result == GRIPPER_RESULT_REACHED:
                self.get_logger().info("Gripper reached goal")
                with self._busy_lock:
                    self._gripper_busy = False
                goal_handle.succeed()
                return GripperCommand.Result(
                    position=position, effort=effort, stalled=False, reached_goal=True
                )
            if self._gripper_result == GRIPPER_RESULT_STALLED:
                self.get_logger().info("Gripper stalled (contact)")
                with self._busy_lock:
                    self._gripper_busy = False
                goal_handle.succeed()
                return GripperCommand.Result(
                    position=position, effort=effort, stalled=True, reached_goal=False
                )
            self.get_logger().warn("Gripper failed (controller reported failure)")
            with self._busy_lock:
                self._gripper_busy = False
            goal_handle.abort()
            return GripperCommand.Result(
                position=position, effort=effort, stalled=False, reached_goal=False
            )

        self.get_logger().warn(f"Gripper timed out ({reason}); stopping control")
        self._stop_pub.publish(Bool(data=True))
        self._gripper_done.clear()
        self._gripper_done.wait(timeout=STOP_ACK_TIMEOUT)
        with self._busy_lock:
            self._gripper_busy = False
        goal_handle.abort()
        position, effort = _measured_state()
        return GripperCommand.Result(
            position=position, effort=effort, stalled=False, reached_goal=False
        )


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
