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


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self.declare_parameter("env_prefix", "env_0")
        self.declare_parameter("rgb_topic", "/{env_prefix}/gemini2/rgb")
        self.declare_parameter("depth_topic", "/{env_prefix}/gemini2/depth")
        self.declare_parameter("detected_image_topic", "/detected_image")
        self.declare_parameter("fx", 686.3)
        self.declare_parameter("fy", 686.3)
        self.declare_parameter("cx", 640.0)
        self.declare_parameter("cy", 360.0)
        self.declare_parameter("camera_to_base_translation", [0.0, 0.5, 0.8])
        self.declare_parameter(
            "camera_to_base_rotation",
            [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
        )
        self.declare_parameter("roi_top", 1.0 / 3.0)
        self.declare_parameter("roi_bottom", 2.0 / 3.0)
        self.declare_parameter("hsv_lower_1", [0, 100, 50])
        self.declare_parameter("hsv_upper_1", [10, 255, 255])
        self.declare_parameter("hsv_lower_2", [170, 100, 50])
        self.declare_parameter("hsv_upper_2", [180, 255, 255])
        self.declare_parameter("min_contour_area", 200.0)
        self.declare_parameter("default_depth", 1.0)

        self._env = str(self.get_parameter("env_prefix").value)
        self._fx = float(self.get_parameter("fx").value)
        self._fy = float(self.get_parameter("fy").value)
        self._cx = float(self.get_parameter("cx").value)
        self._cy = float(self.get_parameter("cy").value)
        self._roi_top = float(self.get_parameter("roi_top").value)
        self._roi_bottom = float(self.get_parameter("roi_bottom").value)
        self._min_contour_area = float(self.get_parameter("min_contour_area").value)
        self._default_depth = float(self.get_parameter("default_depth").value)

        translation = np.asarray(self.get_parameter("camera_to_base_translation").value, dtype=float).reshape(-1)
        rotation = np.asarray(self.get_parameter("camera_to_base_rotation").value, dtype=float).reshape(-1)
        if translation.size != 3:
            raise ValueError("camera_to_base_translation must contain 3 values")
        if rotation.size != 9:
            raise ValueError("camera_to_base_rotation must contain 9 values")
        self._cam_to_base_t = translation
        self._cam_to_base_r = rotation.reshape(3, 3)

        if self._fx <= 0 or self._fy <= 0:
            raise ValueError("fx and fy must be positive")
        if not 0.0 <= self._roi_top < self._roi_bottom <= 1.0:
            raise ValueError("roi_top and roi_bottom must satisfy 0 <= top < bottom <= 1")
        if self._min_contour_area < 0 or self._default_depth <= 0:
            raise ValueError("min_contour_area must be non-negative and default_depth positive")

        def hsv_parameter(name):
            values = np.asarray(self.get_parameter(name).value, dtype=float).reshape(-1)
            limits = np.array([180, 255, 255], dtype=float)
            if values.size != 3 or np.any(values < 0) or np.any(values > limits):
                raise ValueError(f"{name} must contain H/S/V values within 0..180/255/255")
            return tuple(int(value) for value in values)

        self._hsv_lower_1 = hsv_parameter("hsv_lower_1")
        self._hsv_upper_1 = hsv_parameter("hsv_upper_1")
        self._hsv_lower_2 = hsv_parameter("hsv_lower_2")
        self._hsv_upper_2 = hsv_parameter("hsv_upper_2")
        for lower, upper in (
            (self._hsv_lower_1, self._hsv_upper_1),
            (self._hsv_lower_2, self._hsv_upper_2),
        ):
            if any(low > high for low, high in zip(lower, upper)) or upper[0] > 180:
                raise ValueError("HSV lower/upper thresholds are invalid")

        rgb_topic = self._resolve_topic(str(self.get_parameter("rgb_topic").value))
        depth_topic = self._resolve_topic(str(self.get_parameter("depth_topic").value))
        detected_image_topic = str(self.get_parameter("detected_image_topic").value)

        self._bridge = CvBridge()
        self._latest_rgb = None   # (stamp, cv_image)
        self._latest_depth = None  # (stamp, cv_image)

        # 缓存最新帧
        self.create_subscription(Image, rgb_topic, self._rgb_callback, 10)
        self.create_subscription(Image, depth_topic, self._depth_callback, 10)

        self._detected_pub = self.create_publisher(Image, detected_image_topic, 10)
        self._detect_service = self.create_service(DetectObject, "/detect_object", self._detect_callback)

        # OpenCV 显示线程 — 分别存原图和检测结果，线程独立组合
        self._origin_image = None
        self._detected_image = None
        self._display_lock = threading.Lock()
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

        self.get_logger().info(f"Perception node ready (env_prefix={self._env})")

    def _resolve_topic(self, topic):
        return topic.replace("{env_prefix}", self._env)

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
        mask1 = cv2.inRange(hsv, self._hsv_lower_1, self._hsv_upper_1)
        mask2 = cv2.inRange(hsv, self._hsv_lower_2, self._hsv_upper_2)
        mask = mask1 | mask2

        # 只检测 ROI 区域 (按 self._roi_top / self._roi_bottom 裁剪)
        h, w = mask.shape[:2]
        mask[: int(h * self._roi_top), :] = 0
        mask[int(h * self._roi_bottom) :, :] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"detected": False}

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < self._min_contour_area:
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
            z = float(depth[dcy, dcx]) if 0 <= dcx < dw and 0 <= dcy < dh else self._default_depth
        else:
            z = self._default_depth

        if not np.isfinite(z) or z <= 0:
            z = self._default_depth

        x_cam = (cx - self._cx) / self._fx * z
        y_cam = (cy - self._cy) / self._fy * z

        # 相机坐标系 → 机器人基座坐标系 (手眼标定)
        p_cam = np.array([x_cam, y_cam, z])
        p_base = self._cam_to_base_r @ p_cam + self._cam_to_base_t

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
    def _base_to_pixel(self, bx, by, bz, rgb_shape):
        p_base = np.array([bx, by, bz], dtype=float)
        p_cam = self._cam_to_base_r.T @ (p_base - self._cam_to_base_t)
        x_cam, y_cam, z_cam = p_cam
        if abs(z_cam) < 1e-6:
            return (-1, -1)
        px = int(x_cam / z_cam * self._fx + self._cx)
        py = int(y_cam / z_cam * self._fy + self._cy)
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
