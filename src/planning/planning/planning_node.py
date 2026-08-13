#!/usr/bin/env python3
"""规划节点 — 逆解/正解/夹爪
  xyz 单位米, roll/pitch/yaw 单位度, 关节角单位度

ros2 service call /plan_execute custom_msgs/srv/PlanExecute \
  "{command_type: ik_rel, data: '0.1 0 0 0 0 0'}"
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
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
    def __init__(self):
        super().__init__("planning_node")

        self._arm = MoveIt2(
            self,
            joint_names=ARM_JOINTS,
            base_link_name="base_link",
            end_effector_name="tool0",
            group_name="ur_manipulator",
        )

        self._traj_client = ActionClient(
            self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory"
        )
        self._gripper_client = ActionClient(self, GripperCommand, "/robotiq_gripper_controller/gripper_cmd")

        self._current_joints = [0.0] * 6
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)

        self._srv = self.create_service(
            PlanExecute, "/plan_execute", self._callback, callback_group=ReentrantCallbackGroup()
        )
        self.get_logger().info("Planning node ready")

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------

    def _js_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            try:
                i = ARM_JOINTS.index(name)
                self._current_joints[i] = pos
            except ValueError:
                pass

    async def _callback(self, request, response):
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

        if cmd in ("fk_abs", "fk_rel"):
            # all 6 are joint angles in degrees → radians
            parts = [math.radians(v) for v in raw]
            if cmd == "fk_abs":
                traj = self._arm.plan(joint_positions=parts)
                return await self._finish_plan(traj, response)
            else:  # fk_rel
                target = [c + d for c, d in zip(self._current_joints, parts)]
                traj = self._arm.plan(joint_positions=target)
                return await self._finish_plan(traj, response)

        elif cmd in ("ik_abs", "ik_rel"):
            # first 3 are position in meters, last 3 are rpy in degrees → radians
            pos_raw = raw[:3]  # meters, no conversion
            rpy = [math.radians(v) for v in raw[3:]]
            if cmd == "ik_abs":
                traj = self._arm.plan(
                    position=pos_raw, quat_xyzw=self._rpy_to_quat(*rpy), cartesian=True, max_step=0.01
                )
                return await self._finish_plan(traj, response)
            else:  # ik_rel
                fk = self._arm.compute_fk(self._current_joints, fk_link_names=["tool0"])
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
                    position=new_pos, quat_xyzw=new_quat, cartesian=True, max_step=0.01
                )
                return await self._finish_plan(traj, response)

    async def _finish_plan(self, traj, response):
        if traj is None:
            response.success = False
            response.message = "Planning failed"
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
        goal = GripperCommand.Goal()
        goal.command.position = pos
        goal.command.max_effort = 50.0
        if not self._gripper_client.wait_for_server(timeout_sec=1.0):
            response.success = False
            response.message = "Gripper bridge not reachable"
            return response
        gh = await self._gripper_client.send_goal_async(goal)
        if gh is not None and gh.accepted:
            action_result = await gh.get_result_async()
            command_result = action_result.result
            response.success = action_result.status == GoalStatus.STATUS_SUCCEEDED and command_result.reached_goal
            response.message = "done" if response.success else "gripper execution failed"
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
            action_result = await gh.get_result_async()
            command_result = action_result.result
            return action_result.status == GoalStatus.STATUS_SUCCEEDED and command_result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        return False

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
