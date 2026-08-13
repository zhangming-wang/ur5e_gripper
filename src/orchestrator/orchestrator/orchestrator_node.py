#!/usr/bin/env python3
"""调度节点 — pick-and-place 全流程 Action Server

调用方式:
  ros2 action send_goal /pick_and_place custom_msgs/action/PickAndPlace {}
"""

import math
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from custom_msgs.srv import DetectObject, PlanExecute
from custom_msgs.action import PickAndPlace


class OrchestratorNode(Node):
    def __init__(self):
        super().__init__("orchestrator_node")

        # 等待所有依赖服务就绪
        self.get_logger().info("Waiting for /spawn_cube ...")
        self._spawn_cli = self.create_client(Trigger, "/spawn_cube")
        self._spawn_cli.wait_for_service()

        self.get_logger().info("Waiting for /isaac_lab/reset ...")
        self._reset_cli = self.create_client(Trigger, "/isaac_lab/reset")
        self._reset_cli.wait_for_service()

        self.get_logger().info("Waiting for /detect_object ...")
        self._detect_cli = self.create_client(DetectObject, "/detect_object")
        self._detect_cli.wait_for_service()

        self.get_logger().info("Waiting for /plan_execute ...")
        self._plan_cli = self.create_client(PlanExecute, "/plan_execute")
        self._plan_cli.wait_for_service()

        # 标志位：是否取消当前任务
        self._cancel_requested = False
        self._goal_active = False
        self._goal_lock = threading.Lock()
        self._stop_motion_pub = self.create_publisher(Bool, "/isaaclab/stop_motion", 10)

        # Action Server
        self._action_server = ActionServer(
            self,
            PickAndPlace,
            "/pick_and_place",
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )

        self.get_logger().info("Orchestrator action server ready (/pick_and_place)")

    # ==================================================================
    # Action 回调
    # ==================================================================

    def _goal_callback(self, goal_request):
        with self._goal_lock:
            if self._goal_active:
                self.get_logger().warn("Rejecting concurrent PickAndPlace goal")
                return GoalResponse.REJECT
            self._goal_active = True
            self._cancel_requested = False
        self.get_logger().info("PickAndPlace goal received, accepting")
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().warn("PickAndPlace cancel requested")
        self._cancel_requested = True
        self._stop_motion_pub.publish(Bool(data=True))
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        result = PickAndPlace.Result()
        cycle = 0
        succeeded = False

        while not self._cancel_requested:
            cycle += 1
            try:
                # -------- 0. 清除旧方块 --------
                self._call(self._reset_cli, Trigger.Request(), timeout=5.0)
                self._sleep(1.0)
                self._assert_not_cancelled()

                # -------- 1. 生成方块 --------
                self.get_logger().info("━" * 40)
                self.get_logger().info("[1/11] spawn cube")
                self._publish_feedback(goal_handle, 1, "Spawning cube...")
                spawn_res = self._call(self._spawn_cli, Trigger.Request())
                if not spawn_res or not spawn_res.success:
                    raise RuntimeError(f"Spawn failed: {spawn_res.message if spawn_res else 'timeout'}")
                self.get_logger().info("  ✓ cube spawned")
                self._assert_not_cancelled()

                # 等仿真物理落定
                self._sleep(1.0)
                self._assert_not_cancelled()

                # -------- 2. 检测 --------
                self.get_logger().info("[2/11] Detecting cube")
                self._publish_feedback(goal_handle, 2, "Detecting cube...")
                det_res = self._call(self._detect_cli, DetectObject.Request())
                if not det_res or not det_res.detected:
                    raise RuntimeError("Detection failed — cube not found")
                obj_x, obj_y, obj_z, obj_yaw = det_res.x, det_res.y, det_res.z, det_res.yaw
                self.get_logger().info(
                    f"  ✓ detected → base({obj_x:.3f},{obj_y:.3f},{obj_z:.3f}, yaw={math.degrees(obj_yaw):.1f})"
                )
                self._assert_not_cancelled()

                # -------- 3. 移到预抓取位 --------
                place_x, place_y, place_z = obj_x, obj_y, obj_z + 0.25
                self.get_logger().info(
                    f"[3/11] ik_abs x={place_x:.4f} y={place_y:.4f} z={place_z:.4f} "
                    f"roll=180.0 pitch=0.0 yaw={math.degrees(obj_yaw):.1f}"
                )
                self._publish_feedback(goal_handle, 3, "Moving to pre-grasp...")
                self._plan_ik_abs(place_x, place_y, place_z, 180.0, 0.0, math.degrees(obj_yaw))
                self._assert_not_cancelled()

                # -------- 4. 垂直下降 --------
                move_rel_z = 0.075
                self.get_logger().info(f"[4/11] ik_rel dz={-move_rel_z:.3f}")
                self._publish_feedback(goal_handle, 4, "Descending to grasp...")
                self._plan_ik_rel(0, 0, -move_rel_z, 0, 0, 0)
                self._assert_not_cancelled()

                # -------- 5. 闭合夹爪 --------
                self.get_logger().info("[5/11] gripper pos=0.6")
                self._publish_feedback(goal_handle, 5, "Closing gripper...")
                self._plan_gripper(0.6)
                self._assert_not_cancelled()

                # -------- 6. 垂直抬升 --------
                self.get_logger().info(f"[6/11] ik_rel dz={move_rel_z:.3f}")
                self._publish_feedback(goal_handle, 6, "Lifting...")
                self._plan_ik_rel(0, 0, move_rel_z, 0, 0, 0)
                self._assert_not_cancelled()

                # -------- 7. 1轴转到放置侧对称位 --------
                place_angle = math.degrees(math.atan2(place_y, place_x))
                j1_place = -place_angle - 90
                self.get_logger().info(f"[7/11] fk_rel dj1={j1_place:.1f}")
                self._publish_feedback(goal_handle, 7, f"Rotating J1 to {j1_place:.1f}° for placement")
                self._plan_fk_rel(j1_place, 0, 0, 0, 0, 0)
                self._assert_not_cancelled()

                # raise RuntimeError("Cancelled by user")

                # -------- 8. 移到放置托盘上方 --------
                self.get_logger().info(f"[8/11] ik_rel dz={-move_rel_z:.3f}")
                self._publish_feedback(goal_handle, 8, "Moving above placement tray...")
                self._plan_ik_rel(0, 0, -move_rel_z, 0, 0, 0)
                self._assert_not_cancelled()

                # -------- 9. 垂直下降放置 --------
                self.get_logger().info("[9/11] gripper pos=0")
                self._publish_feedback(goal_handle, 9, "Descending to place...")
                self._plan_gripper(0)
                self._assert_not_cancelled()

                # -------- 10. 松开夹爪 --------
                self.get_logger().info(f"[10/11] ik_rel dz={move_rel_z:.3f}")
                self._publish_feedback(goal_handle, 10, "Opening gripper...")
                self._plan_ik_rel(0, 0, move_rel_z, 0, 0, 0)
                self._assert_not_cancelled()

                # -------- 11. 抬升并回原点 --------
                self.get_logger().info("[11/11] fk_abs home (0 -90 90 0 90 0)")
                self._publish_feedback(goal_handle, 11, "Going home...")
                self._go_home()
                self._assert_not_cancelled()

                self.get_logger().info(f"✅ Cycle {cycle} complete\n")
                succeeded = True
                break

            except Exception as e:
                if "Cancelled" in str(e):
                    break
                self.get_logger().error(f"Cycle {cycle} failed: {e}")
                break

        with self._goal_lock:
            self._goal_active = False
        if self._cancel_requested or goal_handle.is_cancel_requested:
            self.get_logger().info("Pick-and-place canceled")
            goal_handle.canceled()
            result.success = False
            result.message = "Canceled"
        elif succeeded:
            self.get_logger().info("Pick-and-place completed")
            goal_handle.succeed()
            result.success = True
            result.message = "Pick-and-place completed"
        else:
            self.get_logger().error("Pick-and-place failed")
            goal_handle.abort()
            result.success = False
            result.message = "Pick-and-place failed"
        return result

    # ==================================================================
    # 规划辅助
    # ==================================================================

    def _plan_ik_abs(self, x, y, z, roll, pitch, yaw):
        req = PlanExecute.Request()
        req.command_type = "ik_abs"
        req.data = f"{x:.5f} {y:.5f} {z:.5f} {roll:.3f} {pitch:.3f} {yaw:.3f}"
        return self._plan(req)

    def _plan_ik_rel(self, dx, dy, dz, droll, dpitch, dyaw):
        req = PlanExecute.Request()
        req.command_type = "ik_rel"
        req.data = f"{dx:.5f} {dy:.5f} {dz:.5f} {droll:.3f} {dpitch:.3f} {dyaw:.3f}"
        return self._plan(req)

    def _plan_gripper(self, pos):
        req = PlanExecute.Request()
        req.command_type = "gripper"
        req.data = f"{pos:.3f}"
        return self._plan(req)

    def _plan_fk_rel(self, dj1, dj2, dj3, dj4, dj5, dj6):
        req = PlanExecute.Request()
        req.command_type = "fk_rel"
        req.data = f"{dj1:.3f} {dj2:.3f} {dj3:.3f} {dj4:.3f} {dj5:.3f} {dj6:.3f}"
        return self._plan(req)

    def _plan_fk_abs(self, j1, j2, j3, j4, j5, j6):
        req = PlanExecute.Request()
        req.command_type = "fk_abs"
        req.data = f"{j1:.3f} {j2:.3f} {j3:.3f} {j4:.3f} {j5:.3f} {j6:.3f}"
        return self._plan(req)

    def _go_home(self):
        """机器人回原点"""
        self.get_logger().info("Going home...")
        self._plan_fk_abs(0.0, -90.0, 90.0, 0.0, 90.0, 0.0)

    def _plan(self, req):
        res = self._call(self._plan_cli, req, timeout=20.0)
        if not res or not res.success:
            raise RuntimeError(f"Plan/execute failed: {res.message if res else 'timeout'}")
        self._sleep(1.0)
        self.get_logger().info(f"  ✓ {res.message}")

    # ==================================================================
    # 通用 service 调用 / 工具
    # ==================================================================

    def _call(self, cli, req, timeout=30.0):
        future = cli.call_async(req)
        start = time.time()
        while not future.done() and time.time() - start < timeout:
            time.sleep(0.01)
        if future.done():
            return future.result()
        self.get_logger().error("Service call timeout")
        return None

    def _sleep(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self._assert_not_cancelled()
            time.sleep(0.05)

    def _assert_not_cancelled(self):
        if self._cancel_requested:
            raise RuntimeError("Cancelled by user")

    def _publish_feedback(self, goal_handle, step, status):
        feedback = PickAndPlace.Feedback()
        feedback.current_step = step
        feedback.status = status
        goal_handle.publish_feedback(feedback)
        self.get_logger().info(f"[{step}/11] {status}")


def main():
    rclpy.init()
    node = OrchestratorNode()
    executor = MultiThreadedExecutor()
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
