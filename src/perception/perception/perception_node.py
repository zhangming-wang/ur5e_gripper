#!/usr/bin/env python3
"""感知节点 — RGB + Depth，按需检测返回 3D 坐标

ros2 service call /detect_object custom_msgs/srv/DetectObject {}
"""

import cv2
import numpy as np
import threading
import time
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from custom_msgs.srv import DetectObject

# Gemini2 相机内参 (从 IsaacLab 仿真传感器读取)
FX, FY = 686.3, 686.3
CX, CY = 640.0, 360.0

# 手眼标定: camera→base 变换 (从 IsaacLab USD Stage 地面真值计算)
# cam_world:  t=(0, 0.5, 1.5)  rpy=(180°,0,0)
# base_world: t=(0, 0,   0.7)  rpy=(0,0,0)
# → camera→base: t=(0, 0.5, 0.8)  rpy=(180°,0,0)
CAM_TO_BASE_T = np.array([0.0, 0.5, 0.8])
# 180° 绕 X 轴的旋转矩阵
CAM_TO_BASE_R = np.array([[1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0]])


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self._bridge = CvBridge()
        self._latest_rgb = None   # (stamp, cv_image)
        self._latest_depth = None  # (stamp, cv_image)

        # 缓存最新帧
        self.create_subscription(Image, "/env_0/gemini2/rgb", self._rgb_callback, 10)
        self.create_subscription(Image, "/env_0/gemini2/depth", self._depth_callback, 10)

        self._detected_pub = self.create_publisher(Image, "/detected_image", 10)
        self._detect_service = self.create_service(DetectObject, "/detect_object", self._detect_callback)

        # OpenCV 显示线程 — 分别存原图和检测结果，线程独立组合
        self._origin_image = None
        self._detected_image = None
        self._display_lock = threading.Lock()
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

        self.get_logger().info("Perception node ready")

    def _rgb_callback(self, msg: Image):
        try:
            self._latest_rgb = (self.get_clock().now(),
                                self._bridge.imgmsg_to_cv2(msg, "bgr8"))
            with self._display_lock:
                self._origin_image = self._latest_rgb[1]
        except Exception as e:
            self.get_logger().warn(f"RGB error: {e}")

    def _depth_callback(self, msg: Image):
        try:
            self._latest_depth = (self.get_clock().now(),
                                  self._bridge.imgmsg_to_cv2(msg, "32FC1"))
        except Exception as e:
            self.get_logger().warn(f"Depth error: {e}")

    def _detect_callback(self, request, response):
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

        # 发布带标注的检测图像
        if result.get("detected"):
            annotated = rgb.copy()
            cv2.drawContours(annotated, [result["contour"]], -1, (0, 255, 0), 2)
            cv2.circle(annotated, (result["cx"], result["cy"]), 5, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"({result['x']:.3f},{result['y']:.3f},{result['z']:.3f})",
                (result["cx"] + 10, result["cy"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            self._detected_pub.publish(self._bridge.cv2_to_imgmsg(annotated, "bgr8"))
            with self._display_lock:
                self._detected_image = annotated
        else:
            with self._display_lock:
                self._detected_image = None

        return response

    def _display_loop(self):
        """OpenCV 显示线程：有检测结果时上下拼接，否则显示原图"""
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        while True:
            with self._display_lock:
                det = self._detected_image
                org = self._origin_image
            if det is not None and org is not None:
                cv2.imshow("Detection", cv2.vconcat([org, det]))
            elif org is not None:
                cv2.imshow("Detection", org)
            else:
                time.sleep(0.1)
                continue
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        cv2.destroyAllWindows()

    def _detect(self, rgb, depth):
        """检测 + 深度转 3D 坐标"""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        # 检测红色物体
        mask1 = cv2.inRange(hsv, (0, 100, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 100, 50), (180, 255, 255))
        mask = mask1 | mask2

        # 只检测图像高度中间 1/3 区域（裁掉上 1/3 和下 1/3）
        h, w = mask.shape[:2]
        mask[:h // 3, :] = 0
        mask[2 * h // 3:, :] = 0

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

        x_cam = (cx - CX) / FX * z
        y_cam = (cy - CY) / FY * z

        # 相机坐标系 → 机器人基座坐标系 (手眼标定)
        p_cam = np.array([x_cam, y_cam, z])
        p_base = CAM_TO_BASE_R @ p_cam + CAM_TO_BASE_T

        self.get_logger().info(
            f"Detected: center=({cx},{cy}) z={z:.3f}m "
            f"→ cam({x_cam:.3f},{y_cam:.3f},{z:.3f}) "
            f"→ base({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f})"
        )

        return {
            "detected": True,
            "x": float(p_base[0]),
            "y": float(p_base[1]),
            "z": float(p_base[2]),
            "yaw": 1.5708,
            "cx": cx, "cy": cy, "contour": c,
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
