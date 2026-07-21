#!/usr/bin/env python3
"""调度节点 — 串行编排 pick-and-place 全流程

流程：
  1. /spawn_cube       → Isaac Lab 在左托盘生成方块，返回右托盘放置位姿
  2. /detect_object     → Perception 检测方块，返回基座坐标
  3. /plan_execute      → 移到方块正上方（预抓取）
  4. /plan_execute      → 垂直下降抓取
  5. /plan_execute      → 闭合夹爪
  6. /plan_execute      → 垂直抬升
  7. /plan_execute      → 移到放置托盘正上方
  8. /plan_execute      → 垂直下降放置
  9. /plan_execute      → 松开夹爪
 10. /plan_execute      → 垂直抬升，回到循环
"""

import json
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from custom_msgs.srv import DetectObject, PlanExecute


class OrchestratorNode(Node):
    def __init__(self):
        super().__init__("orchestrator_node")

        # 等待所有依赖服务就绪
        self.get_logger().info("Waiting for /spawn_cube ...")
        self._spawn_cli = self.create_client(Trigger, "/spawn_cube")
        self._spawn_cli.wait_for_service()

        self.get_logger().info("Waiting for /detect_object ...")
        self._detect_cli = self.create_client(DetectObject, "/detect_object")
        self._detect_cli.wait_for_service()

        self.get_logger().info("Waiting for /plan_execute ...")
        self._plan_cli = self.create_client(PlanExecute, "/plan_execute")
        self._plan_cli.wait_for_service()

        self.get_logger().info("Orchestrator ready")

    def _pick_and_place(self):
        # -------- 1. 生成方块 --------
        self.get_logger().info("─" * 40)
        self.get_logger().info("[1/10] 生成方块 ...")
        spawn_res = self._call(self._spawn_cli, Trigger.Request())
        if not spawn_res or not spawn_res.success:
            raise RuntimeError(f"Spawn failed: {spawn_res.message if spawn_res else 'timeout'}")
        place = json.loads(spawn_res.message)
        self.get_logger().info(f"  ✓ cube spawned, place→({place['place_x']:.3f},{place['place_y']:.3f},{place['place_z']:.3f})")

        # 等仿真物理落定
        time.sleep(1.0)

        # -------- 2. 检测 --------
        self.get_logger().info("[2/10] 检测方块 ...")
        det_res = self._call(self._detect_cli, DetectObject.Request())
        if not det_res or not det_res.detected:
            raise RuntimeError("Detection failed — cube not found")
        obj_x, obj_y, obj_z = det_res.x, det_res.y, det_res.z
        self.get_logger().info(f"  ✓ detected → base({obj_x:.3f},{obj_y:.3f},{obj_z:.3f})")

        # -------- 3. 移到预抓取位 --------
        self.get_logger().info("[3/10] 移到方块上方 15cm ...")
        self._plan_abs(obj_x, obj_y, obj_z + 0.15, 180.0, 0.0, 90.0)

        # -------- 4. 垂直下降 --------
        self.get_logger().info("[4/10] 垂直下降抓取 ...")
        self._plan_rel(0, 0, -0.15, 0, 0, 0)

        # -------- 5. 闭合夹爪 --------
        self.get_logger().info("[5/10] 闭合夹爪 ...")
        self._plan_gripper(0.0)

        # -------- 6. 垂直抬升 --------
        self.get_logger().info("[6/10] 抬升 ...")
        self._plan_rel(0, 0, 0.15, 0, 0, 0)

        # -------- 7. 移到放置位上方 --------
        self.get_logger().info("[7/10] 移到放置托盘上方 ...")
        self._plan_abs(place["place_x"], place["place_y"], place["place_z"] + 0.15,
                       place["place_roll"], place["place_pitch"], place["place_yaw"])

        # -------- 8. 垂直下降放置 --------
        self.get_logger().info("[8/10] 下降放置 ...")
        self._plan_rel(0, 0, -0.15, 0, 0, 0)

        # -------- 9. 松开夹爪 --------
        self.get_logger().info("[9/10] 松开夹爪 ...")
        self._plan_gripper(0.4)

        # -------- 10. 抬升，回到初始高度 --------
        self.get_logger().info("[10/10] 抬升 ...")
        self._plan_rel(0, 0, 0.15, 0, 0, 0)

        self.get_logger().info("✅ 一个 cycle 完成\n")

    # ==================================================================
    # 规划辅助
    # ==================================================================

    def _plan_abs(self, x, y, z, roll, pitch, yaw):
        req = PlanExecute.Request()
        req.command_type = "ik_abs"
        req.data = f"{x:.5f} {y:.5f} {z:.5f} {roll:.3f} {pitch:.3f} {yaw:.3f}"
        return self._plan(req)

    def _plan_rel(self, dx, dy, dz, droll, dpitch, dyaw):
        req = PlanExecute.Request()
        req.command_type = "ik_rel"
        req.data = f"{dx:.5f} {dy:.5f} {dz:.5f} {droll:.3f} {dpitch:.3f} {dyaw:.3f}"
        return self._plan(req)

    def _plan_gripper(self, pos):
        req = PlanExecute.Request()
        req.command_type = "gripper"
        req.data = f"{pos:.3f}"
        return self._plan(req)

    def _plan(self, req):
        res = self._call(self._plan_cli, req, timeout=20.0)
        if not res or not res.success:
            raise RuntimeError(f"Plan/execute failed: {res.message if res else 'timeout'}")
        self.get_logger().info(f"  ✓ {res.message}")

    # ==================================================================
    # 通用 service 调用
    # ==================================================================

    def _call(self, cli, req, timeout=10.0):
        future = cli.call_async(req)
        start = time.time()
        while not future.done() and time.time() - start < timeout:
            time.sleep(0.01)
        if future.done():
            return future.result()
        self.get_logger().error("Service call timeout")
        return None


def main():
    rclpy.init()
    node = OrchestratorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node._pick_and_place()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Pipeline failed: {e}")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
