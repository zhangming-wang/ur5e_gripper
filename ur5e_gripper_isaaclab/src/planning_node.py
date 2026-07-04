#!/usr/bin/env python3
"""规划节点 — 逆解/正解/夹爪，绝对/相对，输入角度(度)

ros2 service call /plan_execute ur5e_gripper_msgs/srv/PlanExecute \
  "{command_type: ik_rel, data: '0.1 0 0 0 0 0'}"
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from ur5e_gripper_msgs.srv import PlanExecute
from control_msgs.action import FollowJointTrajectory, GripperCommand
from pymoveit2 import MoveIt2

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


class PlanningNode(Node):
    def __init__(self):
        super().__init__("planning_node")

        self._arm = MoveIt2(self, joint_names=ARM_JOINTS,
                            base_link_name="base_link", end_effector_name="tool0",
                            group_name="ur_manipulator")

        self._traj_client = ActionClient(self, FollowJointTrajectory,
                                         "/joint_trajectory_controller/follow_joint_trajectory")
        self._gripper_client = ActionClient(self, GripperCommand,
                                            "/robotiq_gripper_controller/gripper_cmd")

        self._current_joints = [0.0] * 6
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)

        self._srv = self.create_service(PlanExecute, "/plan_execute", self._callback,
                                        callback_group=ReentrantCallbackGroup())
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

    async def _handle_arm(self, cmd, data, response):
        parts = [math.radians(float(x)) for x in data.split()]

        if cmd == "fk_abs":
            if len(parts) != 6:
                response.success = False
                response.message = f"Need 6 joint angles, got {len(parts)}"
                return response
            traj = self._arm.plan(joint_positions=parts)
            return await self._finish_plan(traj, response)
        elif cmd == "fk_rel":
            if len(parts) != 6:
                response.success = False
                response.message = f"Need 6 joint deltas, got {len(parts)}"
                return response
            target = [c + d for c, d in zip(self._current_joints, parts)]
            traj = self._arm.plan(joint_positions=target)
            return await self._finish_plan(traj, response)
        elif cmd == "ik_abs":
            if len(parts) != 6:
                response.success = False
                response.message = f"Need x y z roll pitch yaw, got {len(parts)}"
                return response
            pos, rpy = parts[:3], parts[3:]
            traj = self._arm.plan(position=pos, quat_xyzw=self._rpy_to_quat(*rpy),
                                  cartesian=True, max_step=0.01)
            return await self._finish_plan(traj, response)
        elif cmd == "ik_rel":
            if len(parts) != 6:
                response.success = False
                response.message = f"Need dx dy dz droll dpitch dyaw, got {len(parts)}"
                return response
            fk = self._arm.compute_fk(self._current_joints, fk_link_names=["tool0"])
            if fk is None or not fk:
                response.success = False; response.message = "FK failed"; return response
            cur = fk[0] if isinstance(fk, list) else fk
            new_pos = [cur.pose.position.x + parts[0],
                       cur.pose.position.y + parts[1],
                       cur.pose.position.z + parts[2]]
            cur_rpy = self._quat_to_rpy(cur.pose.orientation)
            new_rpy = [cur_rpy[i] + parts[3 + i] for i in range(3)]
            traj = self._arm.plan(position=new_pos, quat_xyzw=self._rpy_to_quat(*new_rpy),
                                  cartesian=True, max_step=0.01)
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
        if gh.accepted:
            await gh.get_result_async()
            response.success = True
            response.message = "done"
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
        if gh.accepted:
            await gh.get_result_async()
            return True
        return False

    @staticmethod
    def _rpy_to_quat(roll, pitch, yaw):
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return (sr*cp*cy - cr*sp*sy,  cr*sp*cy + sr*cp*sy,
                cr*cp*sy - sr*sp*cy,  cr*cp*cy + sr*sp*sy)

    @staticmethod
    def _quat_to_rpy(q):
        x, y, z, w = q.x, q.y, q.z, q.w
        sinr, cosr = 2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0*(w*y - z*x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny, cosy = 2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)
        yaw = math.atan2(siny, cosy)
        return (roll, pitch, yaw)


def main():
    rclpy.init()
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
