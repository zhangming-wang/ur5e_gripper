#!/usr/bin/env python3
"""感知节点 — RGB + Depth，按需检测返回 3D 坐标

ros2 service call /detect_object custom_msgs/srv/DetectObject {}
"""

import cv2
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from custom_msgs.srv import DetectObject

# Gemini2 相机内参 (估计值，后续标定)
FX, FY = 686.3, 686.3
CX, CY = 640.0, 360.0


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self._bridge = CvBridge()
        self._latest_rgb = None   # (stamp, cv_image)
        self._latest_depth = None  # (stamp, cv_image)

        # 缓存最新帧
        self.create_subscription(Image, "/env_0/gemini2/rgb", self._rgb_cb, 10)
        self.create_subscription(Image, "/env_0/gemini2/depth", self._depth_cb, 10)

        self._srv = self.create_service(DetectObject, "/detect_object", self._detect_cb)
        self.get_logger().info("Perception node ready")

    def _rgb_cb(self, msg: Image):
        try:
            self._latest_rgb = (self.get_clock().now(),
                                self._bridge.imgmsg_to_cv2(msg, "bgr8"))
        except Exception as e:
            self.get_logger().warn(f"RGB error: {e}")

    def _depth_cb(self, msg: Image):
        try:
            self._latest_depth = (self.get_clock().now(),
                                  self._bridge.imgmsg_to_cv2(msg, "32FC1"))
        except Exception as e:
            self.get_logger().warn(f"Depth error: {e}")

    def _detect_cb(self, request, response):
        if self._latest_rgb is None:
            response.detected = False
            return response

        _, rgb = self._latest_rgb
        depth = self._latest_depth[1] if self._latest_depth else None
        result = self._detect(rgb, depth)

        response.x = result.get("x", 0.0)
        response.y = result.get("y", 0.0)
        response.z = result.get("z", 0.0)
        response.yaw = result.get("yaw", 0.0)
        response.detected = result.get("detected", False)
        return response

    def _detect(self, rgb, depth):
        """检测 + 深度转 3D 坐标"""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        # 检测红色物体
        mask1 = cv2.inRange(hsv, (0, 100, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 100, 50), (180, 255, 255))
        mask = mask1 | mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"detected": False}

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 200:
            return {"detected": False}

        M = cv2.moments(c)
        if M["m00"] == 0:
            return {"detected": False}

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 深度 → 真实 3D 坐标 (相机坐标系)
        h_rgb, w_rgb = rgb.shape[:2]
        if depth is not None:
            dh, dw = depth.shape[:2]
            # 缩放中心点到深度分辨率
            dcx = int(cx * dw / w_rgb)
            dcy = int(cy * dh / h_rgb)
            z = float(depth[dcy, dcx]) if 0 <= dcx < dw and 0 <= dcy < dh else 1.0
        else:
            z = 1.0

        x = (cx - CX) / FX * z
        y = (cy - CY) / FY * z

        self.get_logger().info(f"Detected: center=({cx},{cy}) z={z:.3f}m → ({x:.3f},{y:.3f},{z:.3f})")

        return {
            "detected": True,
            "x": x, "y": y, "z": z,
            "yaw": 1.5708,
        }


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
