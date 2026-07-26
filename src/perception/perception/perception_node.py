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
# 180° 绕 X 轴的旋转矩阵 + x/y 镜像矫正
CAM_TO_BASE_R = np.array([[-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, -1.0]])


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        # 环境前缀参数，默认 "env_0"，支持 launch 传参
        self.declare_parameter("env_prefix", "env_0")
        self._env = self.get_parameter("env_prefix").get_parameter_value().string_value

        # 检测区域参数 (占图像高度的比例，范围 0.0~1.0)
        self._roi_top = 1.0 / 3.0
        self._roi_bottom = 2.0 / 3.0

        self._bridge = CvBridge()
        self._latest_rgb = None   # (stamp, cv_image)
        self._latest_depth = None  # (stamp, cv_image)

        # 缓存最新帧
        self.create_subscription(Image, f"/{self._env}/gemini2/rgb", self._rgb_callback, 10)
        self.create_subscription(Image, f"/{self._env}/gemini2/depth", self._depth_callback, 10)

        self._detected_pub = self.create_publisher(Image, "/detected_image", 10)
        self._detect_service = self.create_service(DetectObject, "/detect_object", self._detect_callback)

        # OpenCV 显示线程 — 分别存原图和检测结果，线程独立组合
        self._origin_image = None
        self._detected_image = None
        self._display_lock = threading.Lock()
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

        self.get_logger().info(f"Perception node ready (env_prefix={self._env})")

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
            cv2.drawContours(annotated, [result["box"]], 0, (0, 255, 0), 2)
            cv2.circle(annotated, (result["cx"], result["cy"]), 5, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"({result['x']:.3f},{result['y']:.3f},{result['z']:.3f},{np.rad2deg(result['yaw']):.1f})",
                (result["cx"] + 10, result["cy"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
            # 画基座坐标系: X红 Y绿
            bx, by, bz = result["x"], result["y"], result["z"] + 0.005
            for axis_color, dbx, dby in [((0, 0, 255), 0.03, 0.0), ((0, 255, 0), 0.0, 0.03)]:
                px, py = self._base_to_pixel(bx + dbx, by + dby, bz, rgb.shape)
                cv2.arrowedLine(annotated, self._base_to_pixel(bx, by, bz, rgb.shape),
                                (px, py), axis_color, 2)
            disp = annotated
        else:
            disp = rgb.copy()
            cv2.putText(disp, "NO DETECT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        self._detected_pub.publish(self._bridge.cv2_to_imgmsg(disp, "bgr8"))
        with self._display_lock:
            self._detected_image = disp

        return response

    def _display_loop(self):
        """OpenCV 显示线程：有检测结果时上下拼接，否则显示原图"""
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        while True:
            with self._display_lock:
                det = self._detected_image
                org = self._origin_image
            if org is not None and det is not None:
                h = det.shape[0]
                y1 = int(h * self._roi_top)
                y2 = int(h * self._roi_bottom)
                det = det.copy()
                cv2.line(det, (0, y1), (det.shape[1], y1), (0, 0, 255), 2)
                cv2.line(det, (0, y2), (det.shape[1], y2), (0, 0, 255), 2)
                org = org.copy()
                sep = np.full((5, org.shape[1], 3), 255, dtype=np.uint8)
                cv2.imshow("Detection", cv2.vconcat([org, sep, det]))
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

        # 只检测 ROI 区域 (按 self._roi_top / self._roi_bottom 裁剪)
        h, w = mask.shape[:2]
        mask[: int(h * self._roi_top), :] = 0
        mask[int(h * self._roi_bottom) :, :] = 0

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

        # 最小包围矩形获取方向 — 用长边向量算角度
        box = cv2.boxPoints(cv2.minAreaRect(c))
        # 找最长边
        edges = [(box[(i + 1) % 4] - box[i]) for i in range(4)]
        longest = max(edges, key=lambda v: np.linalg.norm(v))
        yaw = np.arctan2(longest[0], longest[1])  # arctan2(dx, dy)
        yaw_rad = yaw % np.pi  # 归一化到 [0, π)

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
            f"→ base({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f}, yaw={np.rad2deg(yaw_rad):.1f})"
        )

        return {
            "detected": True,
            "x": float(p_base[0]),
            "y": float(p_base[1]),
            "z": float(p_base[2]),
            "yaw": yaw_rad,
            "cx": cx, "cy": cy, "contour": c, "box": box.astype(np.int32),
        }
    @staticmethod
    def _base_to_pixel(bx, by, bz, rgb_shape):
        x_cam = -bx
        y_cam = by - 0.5
        z_cam = 0.8 - bz
        if abs(z_cam) < 1e-6:
            return (-1, -1)
        px = int(x_cam / z_cam * FX + CX)
        py = int(y_cam / z_cam * FY + CY)
        return (min(max(px, 0), rgb_shape[1] - 1), min(max(py, 0), rgb_shape[0] - 1))


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
