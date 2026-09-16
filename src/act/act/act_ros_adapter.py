#!/usr/bin/env python3
"""ACT 策略执行器 — 把 IsaacSim 观测转发给 ACT 推理服务并发布关节目标。

由 ``act_orchestrator`` 通过 ``/isaaclab/act/enabled`` 与 ``~/prepare`` 驱动：
  * ``~/prepare``（Trigger）用最新观测预取一个 action chunk 填满缓冲
  * ``/isaaclab/act/enabled`` 为 True 时按 ``fps`` 发布缓冲中的动作，并后台预取
  * 为 False 时静默（``control_node`` 的 watchdog 会让机械臂保持位置）

运行在系统 ROS Python，使用单线程执行器；控制状态只由 ROS 回调线程访问，
线程池仅用于 HTTP 请求且只通过 ``queue.Queue`` 回传，因此无需加锁。
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import queue
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
POLICY_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]
ACT_ENABLED_TOPIC = "/isaaclab/act/enabled"


class ActPolicyExecutor(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("act_ros_adapter")
        self._args = args
        self._bridge = CvBridge()
        self._server_url = args.server_url.rstrip("/")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-infer")
        self._fetch_future: Future | None = None
        self._result_queue: queue.Queue[tuple[list[list[float]] | None, str | None]] = queue.Queue(
            maxsize=2
        )

        self._latest_image: np.ndarray | None = None
        self._latest_image_time = 0.0
        self._latest_state: np.ndarray | None = None
        self._latest_state_time = 0.0
        self._buffer: collections.deque = collections.deque()
        self._enabled = False
        self._last_gripper_command = 0.0
        self._published_count = 0
        self._last_stale_log = 0.0
        self._last_fetch_error_log = 0.0

        qos = 10
        self.create_subscription(
            Image,
            f"/{args.env_prefix}/gemini2/rgb",
            self._image_callback,
            qos,
        )
        self.create_subscription(JointState, "/joint_states", self._joint_callback, qos)
        self.create_subscription(Bool, ACT_ENABLED_TOPIC, self._enabled_callback, qos)
        self._target_pub = self.create_publisher(
            JointTrajectory, "/isaaclab/act/joint_target", qos
        )
        self._prepare_service = self.create_service(Trigger, "~/prepare", self._prepare_callback)
        self._timer = self.create_timer(1.0 / args.fps, self._tick)

        self.get_logger().info(
            f"ACT policy executor ready: server={self._server_url}, "
            f"camera=/{args.env_prefix}/gemini2/rgb, fps={args.fps}, "
            f"prefetch_threshold={args.prefetch_threshold}"
        )

    # ------------------------------------------------------------------
    # 观测订阅
    # ------------------------------------------------------------------
    def _image_callback(self, msg: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"expected RGB image, got {image.shape}")
            self._latest_image = np.ascontiguousarray(image)
            self._latest_image_time = time.monotonic()
        except Exception as exc:
            self.get_logger().warning(f"RGB conversion failed: {exc}")

    def _joint_callback(self, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position))
        if any(name not in positions for name in POLICY_JOINTS):
            return
        state = np.asarray([positions[name] for name in POLICY_JOINTS], dtype=np.float32)
        if not np.isfinite(state).all():
            return
        self._latest_state = state
        self._latest_state_time = time.monotonic()

    def _enabled_callback(self, msg: Bool) -> None:
        enabled = bool(msg.data)
        if enabled and not self._enabled:
            self._enabled = True
            self._last_gripper_command = 0.0
            self._published_count = 0
            self.get_logger().info(f"ACT enabled (buffer={len(self._buffer)})")
        elif not enabled and self._enabled:
            self._enabled = False
            self._buffer.clear()
            self.get_logger().info("ACT disabled")

    # ------------------------------------------------------------------
    # 推理服务协议
    # ------------------------------------------------------------------
    def _infer_chunk_request(self, image: np.ndarray, state: np.ndarray) -> list[list[float]]:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._args.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("failed to encode RGB image")
        payload = {
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "state": state.tolist(),
        }
        request = urllib.request.Request(
            f"{self._server_url}/infer_chunk",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._args.request_timeout) as response:
            result = json.loads(response.read())
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error", "policy server rejected request")))
        chunk = np.asarray(result.get("chunk"), dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != len(POLICY_JOINTS) or chunk.shape[0] < 1:
            raise RuntimeError(f"invalid chunk from server: shape={chunk.shape}")
        if not np.isfinite(chunk).all():
            raise RuntimeError("chunk contains non-finite values")
        return chunk.tolist()

    def _prepare_callback(self, request, response):
        """编排节点在 enable 前调用，用最新观测填满动作缓冲。"""
        if self._latest_image is None or self._latest_state is None:
            response.success = False
            response.message = "no observation received yet"
            return response
        image = self._latest_image.copy()
        state = self._latest_state.copy()
        try:
            chunk = self._infer_chunk_request(image, state)
        except Exception as exc:
            response.success = False
            response.message = f"prepare failed: {exc}"
            self.get_logger().error(f"prepare failed: {exc}")
            return response
        self._buffer = collections.deque(chunk)
        self._fetch_future = None
        response.success = True
        response.message = f"prepared {len(chunk)} actions"
        self.get_logger().info(f"prepared {len(chunk)} actions")
        return response

    # ------------------------------------------------------------------
    # 预取与控制发布
    # ------------------------------------------------------------------
    def _submit_prefetch(self, image: np.ndarray, state: np.ndarray) -> None:
        future = self._executor.submit(self._infer_chunk_request, image, state)
        self._fetch_future = future

        def collect(done: Future) -> None:
            try:
                chunk = done.result()
                item = (chunk, None)
            except Exception as exc:  # 经由队列回到 ROS 线程
                item = (None, str(exc))
            try:
                self._result_queue.put_nowait(item)
            except queue.Full:
                pass

        future.add_done_callback(collect)

    def _consume_result(self) -> None:
        try:
            chunk, error = self._result_queue.get_nowait()
        except queue.Empty:
            return
        self._fetch_future = None
        if error is not None:
            now = time.monotonic()
            if now - self._last_fetch_error_log > 1.0:
                self.get_logger().warning(f"policy chunk failed: {error}")
                self._last_fetch_error_log = now
            return
        if chunk is None or not self._enabled:
            return
        self._buffer = collections.deque(chunk)
        self.get_logger().info(f"chunk buffered ({len(chunk)} actions)")

    @staticmethod
    def _map_gripper(predicted: float, previous: float) -> tuple[float, float]:
        # 数据集存的是实测 knuckle 位置（接触时约 0.63），而 IsaacSim 接受 0~0.8 的逻辑命令。
        if predicted >= 0.20:
            return 0.8, 0.8
        if predicted <= 0.10:
            return 0.0, 0.0
        return previous, previous

    def _publish_action(self, action: list[float]) -> None:
        predicted_gripper = float(action[6])
        gripper, self._last_gripper_command = self._map_gripper(
            predicted_gripper, self._last_gripper_command
        )

        msg = JointTrajectory()
        msg.joint_names = POLICY_JOINTS
        point = JointTrajectoryPoint()
        point.positions = [*action[:6], gripper]
        point.time_from_start.sec = int(self._args.command_period)
        point.time_from_start.nanosec = int(
            (self._args.command_period - int(self._args.command_period)) * 1_000_000_000
        )
        msg.points = [point]
        self._target_pub.publish(msg)

        self._published_count += 1
        if self._published_count % self._args.diag_interval == 0:
            self.get_logger().info(
                f"diag: buffer={len(self._buffer)} pred_gripper={predicted_gripper:.3f} "
                f"cmd_gripper={gripper:.2f}"
            )

    def _tick(self) -> None:
        self._consume_result()
        if not self._enabled:
            return

        image = self._latest_image
        state = self._latest_state
        if image is None or state is None:
            return
        image_age = time.monotonic() - self._latest_image_time
        state_age = time.monotonic() - self._latest_state_time
        if image_age > self._args.max_observation_age or state_age > self._args.max_observation_age:
            # 不夺权：只停止发布，让 control_node 的 watchdog 保持位置，
            # 由编排节点判定超时并结束本轮。
            now = time.monotonic()
            if now - self._last_stale_log > 1.0:
                self.get_logger().warning(
                    f"stale observation (image={image_age:.3f}s, state={state_age:.3f}s); holding"
                )
                self._last_stale_log = now
            return

        if self._buffer:
            self._publish_action(self._buffer.popleft())

        if self._fetch_future is None and len(self._buffer) <= self._args.prefetch_threshold:
            self._submit_prefetch(image.copy(), state.copy())

    def destroy_node(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:27655")
    parser.add_argument("--env-prefix", default="env_0")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--request-timeout", type=float, default=2.0)
    parser.add_argument("--max-observation-age", type=float, default=0.25)
    parser.add_argument("--command-period", type=float, default=1.0 / 15.0)
    parser.add_argument("--prefetch-threshold", type=int, default=10)
    parser.add_argument("--diag-interval", type=int, default=30)
    args = parser.parse_args()
    if args.fps <= 0 or args.request_timeout <= 0 or args.command_period <= 0:
        parser.error("fps, request-timeout, and command-period must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("jpeg-quality must be between 1 and 100")
    if args.prefetch_threshold < 0:
        parser.error("prefetch-threshold must be non-negative")
    if args.diag_interval <= 0:
        parser.error("diag-interval must be positive")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = ActPolicyExecutor(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
