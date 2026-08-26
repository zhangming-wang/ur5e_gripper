#!/usr/bin/env python3
"""规划节点 — 逆解/正解/夹爪
  xyz 单位米, roll/pitch/yaw 单位度, 关节角单位度

ros2 service call /plan_execute custom_msgs/srv/PlanExecute \
  "{command_type: ik_rel, data: '0.1 0 0 0 0 0'}"
"""

import math
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from custom_msgs.srv import PlanExecute
from control_msgs.action import FollowJointTrajectory, GripperCommand
from pymoveit2 import MoveIt2

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class PlanningNode(Node):
    CARTESIAN_FRACTION_THRESHOLD = 0.99

    def __init__(self):
        super().__init__("planning_node")

        self._arm = MoveIt2(
            self,
            joint_names=ARM_JOINTS,
            base_link_name="base_link",
            end_effector_name="tool0",
            group_name="ur_manipulator",
        )
        self._arm.cartesian_avoid_collisions = True

        self._traj_client = ActionClient(
            self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory"
        )
        self._gripper_client = ActionClient(self, GripperCommand, "/robotiq_gripper_controller/gripper_cmd")

        self._command_lock = threading.Lock()
        self._active_goal_lock = threading.Lock()
        self._active_arm_goal = None
        self._active_gripper_goal = None
        self._stop_requested = threading.Event()
        self.create_subscription(Bool, "/stop_motion", self._stop_motion_cb, 10)

        self._srv = self.create_service(
            PlanExecute, "/plan_execute", self._callback, callback_group=ReentrantCallbackGroup()
        )
        self.get_logger().info("Planning node ready")

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------

    def _joint_state_snapshot(self):
        """Return one consistently ordered arm state from MoveIt's latest sample."""
        source = self._arm.joint_state
        if source is None:
            return None

        positions = dict(zip(source.name, source.position))
        if any(name not in positions or not math.isfinite(positions[name]) for name in ARM_JOINTS):
            return None

        snapshot = JointState()
        snapshot.header = source.header
        snapshot.name = list(ARM_JOINTS)
        snapshot.position = [positions[name] for name in ARM_JOINTS]
        return snapshot

    async def _callback(self, request, response):
        if not self._command_lock.acquire(blocking=False):
            response.success = False
            response.message = "Another planning command is already executing"
            return response

        self.get_logger().info(f"Received: {request.command_type} '{request.data}'")
        try:
            cmd = request.command_type
            data = request.data
            if cmd in ("ik_abs", "ik_rel", "fk_abs", "fk_rel"):
                return await self._handle_arm(cmd, data, response)
            elif cmd == "gripper":
                return await self._handle_gripper(data, response)
            response.success = False
            response.message = f"Unknown command: {cmd}"
        except Exception as e:
            response.success = False
            response.message = str(e)
        finally:
            self._command_lock.release()
        return response

    # ------------------------------------------------------------------
    # Arm
    # ------------------------------------------------------------------

    def _parse_arm(self, data):
        """Parse 6 space-separated floats. Returns raw values (no unit conversion)."""
        return [float(x) for x in data.split()]

    async def _handle_arm(self, cmd, data, response):
        raw = self._parse_arm(data)
        if len(raw) != 6:
            response.success = False
            response.message = f"Need 6 values, got {len(raw)}"
            return response

        state = self._joint_state_snapshot()
        if state is None:
            response.success = False
            response.message = "Joint state is unavailable or incomplete"
            return response
        current = list(state.position)
        self._stop_requested.clear()

        if cmd in ("fk_abs", "fk_rel"):
            # all 6 are joint angles in degrees → radians
            parts = [math.radians(v) for v in raw]
            if cmd == "fk_abs":
                traj = self._arm.plan(joint_positions=parts, start_joint_state=state)
                return await self._finish_plan(traj, response)
            else:  # fk_rel
                target = [c + d for c, d in zip(current, parts)]
                fixed_names = [
                    name for name, delta in zip(ARM_JOINTS, parts) if abs(delta) < 1e-9
                ]
                fixed_positions = [
                    value for value, delta in zip(current, parts) if abs(delta) < 1e-9
                ]
                if fixed_names:
                    self._arm.set_path_joint_constraint(
                        joint_positions=fixed_positions,
                        joint_names=fixed_names,
                        tolerance=1e-4,
                    )
                traj = self._arm.plan(joint_positions=target, start_joint_state=state)
                return await self._finish_plan(traj, response)

        elif cmd in ("ik_abs", "ik_rel"):
            # first 3 are position in meters, last 3 are rpy in degrees → radians
            pos_raw = raw[:3]  # meters, no conversion
            rpy = [math.radians(v) for v in raw[3:]]
            if cmd == "ik_abs":
                traj = self._arm.plan(
                    position=pos_raw,
                    quat_xyzw=self._rpy_to_quat(*rpy),
                    start_joint_state=state,
                    cartesian=True,
                    max_step=0.01,
                    cartesian_fraction_threshold=self.CARTESIAN_FRACTION_THRESHOLD,
                )
                return await self._finish_plan(traj, response)
            else:  # ik_rel
                fk = self._arm.compute_fk(state, fk_link_names=["tool0"])
                if fk is None or not fk:
                    response.success = False
                    response.message = "FK failed"
                    return response
                cur = fk[0] if isinstance(fk, list) else fk
                new_pos = [
                    cur.pose.position.x + pos_raw[0],
                    cur.pose.position.y + pos_raw[1],
                    cur.pose.position.z + pos_raw[2],
                ]
                delta_quat = self._rpy_to_quat(*rpy)
                new_quat = self._quat_multiply(delta_quat, self._pose_to_quat(cur.pose))
                traj = self._arm.plan(
                    position=new_pos,
                    quat_xyzw=new_quat,
                    start_joint_state=state,
                    cartesian=True,
                    max_step=0.01,
                    cartesian_fraction_threshold=self.CARTESIAN_FRACTION_THRESHOLD,
                )
                return await self._finish_plan(traj, response)

    async def _finish_plan(self, traj, response):
        if traj is None or not traj.points:
            response.success = False
            response.message = "Planning failed or returned an empty trajectory"
            return response
        self.get_logger().info(f"Executing ({len(traj.points)} pts)...")
        ok = await self._call_bridge_arm(traj)
        response.success = ok
        response.message = "done" if ok else "execution failed"
        return response

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    async def _handle_gripper(self, data, response):
        pos = float(data.strip())
        self.get_logger().info(f"Gripper: {pos:.3f}")
        self._stop_requested.clear()
        goal = GripperCommand.Goal()
        goal.command.position = pos
        goal.command.max_effort = 50.0
        if not self._gripper_client.wait_for_server(timeout_sec=1.0):
            response.success = False
            response.message = "Gripper bridge not reachable"
            return response
        gh = await self._gripper_client.send_goal_async(goal)
        if gh is not None and gh.accepted:
            with self._active_goal_lock:
                self._active_gripper_goal = gh
                should_cancel = self._stop_requested.is_set()
            if should_cancel:
                gh.cancel_goal_async()
            try:
                action_result = await gh.get_result_async()
                command_result = action_result.result
                action_succeeded = action_result.status == GoalStatus.STATUS_SUCCEEDED
                reached_goal = bool(command_result.reached_goal)
                contact_stall = pos > 0.0 and bool(command_result.stalled)
                response.success = (
                    not self._stop_requested.is_set()
                    and action_succeeded
                    and (reached_goal or contact_stall)
                )
                if contact_stall and response.success and not reached_goal:
                    self.get_logger().info(
                        f"Gripper contact detected at position={command_result.position:.3f}; "
                        "accepting close as successful"
                    )
                elif not response.success:
                    self.get_logger().warn(
                        f"Gripper action failed: status={action_result.status}, "
                        f"position={command_result.position:.3f}, "
                        f"stalled={command_result.stalled}, "
                        f"reached_goal={command_result.reached_goal}"
                    )
                response.message = "contact detected" if contact_stall and response.success else (
                    "done" if response.success else "gripper execution failed"
                )
            finally:
                with self._active_goal_lock:
                    self._active_gripper_goal = None
        else:
            response.success = False
            response.message = "rejected"
        return response

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    async def _call_bridge_arm(self, traj):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        if not self._traj_client.wait_for_server(timeout_sec=1.0):
            return False
        gh = await self._traj_client.send_goal_async(goal)
        if gh is not None and gh.accepted:
            with self._active_goal_lock:
                self._active_arm_goal = gh
                should_cancel = self._stop_requested.is_set()
            if should_cancel:
                gh.cancel_goal_async()
            try:
                action_result = await gh.get_result_async()
                command_result = action_result.result
                return (
                    not self._stop_requested.is_set()
                    and action_result.status == GoalStatus.STATUS_SUCCEEDED
                    and command_result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
                )
            finally:
                with self._active_goal_lock:
                    self._active_arm_goal = None
        return False

    def _stop_motion_cb(self, msg):
        if msg is None or not msg.data:
            return
        self._stop_requested.set()
        with self._active_goal_lock:
            goals = [self._active_arm_goal, self._active_gripper_goal]
        for goal in goals:
            if goal is not None:
                goal.cancel_goal_async()

    @staticmethod
    def _rpy_to_quat(roll, pitch, yaw):
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    @staticmethod
    def _quat_to_rpy(q):
        x, y, z, w = q.x, q.y, q.z, q.w
        sinr, cosr = 2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny, cosy = 2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny, cosy)
        return (roll, pitch, yaw)

    @staticmethod
    def _pose_to_quat(pose):
        return (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)

    @staticmethod
    def _quat_multiply(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )


def main():
    rclpy.init()
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
