#!/usr/bin/env python3
"""Execute Diffusion Policy action chunks through isolated IsaacLab topics."""

from __future__ import annotations

import argparse
import base64
import collections
import json
import queue
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
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
DIFFUSION_ENABLED_TOPIC = "/isaaclab/diffusion/enabled"
DIFFUSION_TARGET_TOPIC = "/isaaclab/diffusion/joint_target"


class DiffusionPolicyExecutor(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("diffusion_ros_adapter")
        self._args = args
        self._bridge = CvBridge()
        self._server_url = args.server_url.rstrip("/")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diffusion-infer")
        self._fetch_future: Future | None = None
        self._result_queue: queue.Queue[
            tuple[int, int, list[list[float]] | None, str | None]
        ] = queue.Queue(maxsize=2)
        self._generation = 0
        self._latest_image: np.ndarray | None = None
        self._latest_image_time = 0.0
        self._latest_state: np.ndarray | None = None
        self._latest_state_time = 0.0
        self._observations: collections.deque[tuple[np.ndarray, np.ndarray]] = collections.deque(maxlen=2)
        self._buffer: collections.deque[list[float]] = collections.deque()
        self._enabled = False
        self._gripper_phase = "open"
        self._gripper_candidate_count = 0
        self._last_action: np.ndarray | None = None
        self._published_count = 0
        self._last_stale_log = 0.0
        self._last_fetch_error_log = 0.0

        qos = 10
        self.create_subscription(Image, f"/{args.env_prefix}/gemini2/rgb", self._image_callback, qos)
        self.create_subscription(JointState, "/joint_states", self._joint_callback, qos)
        self.create_subscription(Bool, DIFFUSION_ENABLED_TOPIC, self._enabled_callback, qos)
        self._target_pub = self.create_publisher(JointTrajectory, DIFFUSION_TARGET_TOPIC, qos)
        self._prepare_service = self.create_service(Trigger, "~/prepare", self._prepare_callback)
        self._timer = self.create_timer(1.0 / args.fps, self._tick)
        self.get_logger().info(
            f"Diffusion policy executor ready: server={self._server_url}, "
            f"camera=/{args.env_prefix}/gemini2/rgb, fps={args.fps}, "
            f"prefetch_threshold={args.prefetch_threshold}"
        )

    def _image_callback(self, msg: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"expected RGB image, got {image.shape}")
            self._latest_image = np.ascontiguousarray(image)
            self._latest_image_time = time.monotonic()
            if self._latest_state is not None:
                self._observations.append((self._latest_image.copy(), self._latest_state.copy()))
        except Exception as exc:
            self.get_logger().warning(f"RGB conversion failed: {exc}")

    def _joint_callback(self, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position))
        if any(name not in positions for name in POLICY_JOINTS):
            return
        state = np.asarray([positions[name] for name in POLICY_JOINTS], dtype=np.float32)
        if np.isfinite(state).all():
            self._latest_state = state
            self._latest_state_time = time.monotonic()

    def _enabled_callback(self, msg: Bool) -> None:
        enabled = bool(msg.data)
        if enabled and not self._enabled:
            self._enabled = True
            self._gripper_phase = "open"
            self._gripper_candidate_count = 0
            self._last_action = None
            self._published_count = 0
            self.get_logger().info(f"Diffusion enabled (buffer={len(self._buffer)})")
            # The chunk was prepared before enable. Publish its first action now
            # instead of waiting for the next wall-clock timer tick.
            if self._buffer:
                self._publish_action(self._buffer.popleft())
        elif not enabled and self._enabled:
            self._enabled = False
            self._generation += 1
            self._buffer.clear()
            self.get_logger().info("Diffusion disabled")

    def _request(self, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self._server_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._args.request_timeout) as response:
            result = json.loads(response.read())
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error", "policy server rejected request")))
        return result

    def _history_snapshot(self) -> list[tuple[np.ndarray, np.ndarray]] | None:
        if len(self._observations) == 2:
            return [(image.copy(), state.copy()) for image, state in self._observations]
        if self._latest_image is None or self._latest_state is None:
            return None
        return [(self._latest_image.copy(), self._latest_state.copy())] * 2

    def _infer_chunk_request(self, history: list[tuple[np.ndarray, np.ndarray]]) -> list[list[float]]:
        images = []
        states = []
        for image, state in history:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._args.jpeg_quality])
            if not ok:
                raise RuntimeError("failed to encode RGB image")
            images.append(base64.b64encode(encoded.tobytes()).decode("ascii"))
            states.append(state.tolist())
        result = self._request(
            "/infer_chunk",
            {"images": images, "states": states},
        )
        chunk = np.asarray(result.get("chunk"), dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != len(POLICY_JOINTS) or chunk.shape[0] != self._args.chunk_size:
            raise RuntimeError(f"invalid chunk from server: shape={chunk.shape}")
        if not np.isfinite(chunk).all():
            raise RuntimeError("chunk contains non-finite values")
        return chunk.tolist()

    def _prepare_callback(self, request, response):
        history = self._history_snapshot()
        if history is None:
            response.success = False
            response.message = "no observation received yet"
            return response
        try:
            self._request("/reset", {})
            chunk = self._infer_chunk_request(history)
        except Exception as exc:
            response.success = False
            response.message = f"prepare failed: {exc}"
            self.get_logger().error(response.message)
            return response
        self._generation += 1
        self._buffer = collections.deque(chunk)
        response.success = True
        response.message = f"prepared {len(chunk)} actions"
        self.get_logger().info(response.message)
        return response

    def _submit_prefetch(self, history: list[tuple[np.ndarray, np.ndarray]]) -> None:
        generation = self._generation
        request_step = self._published_count
        self._fetch_future = self._executor.submit(self._infer_chunk_request, history)

        def collect(done: Future) -> None:
            try:
                item = (generation, request_step, done.result(), None)
            except Exception as exc:
                item = (generation, request_step, None, str(exc))
            try:
                self._result_queue.put_nowait(item)
            except queue.Full:
                pass

        self._fetch_future.add_done_callback(collect)

    def _consume_result(self) -> None:
        try:
            generation, request_step, chunk, error = self._result_queue.get_nowait()
        except queue.Empty:
            return
        self._fetch_future = None
        if generation != self._generation:
            return
        if error is not None:
            now = time.monotonic()
            if now - self._last_fetch_error_log > 1.0:
                self.get_logger().warning(f"policy chunk failed: {error}")
                self._last_fetch_error_log = now
            return
        if chunk is not None and self._enabled:
            elapsed_steps = max(0, self._published_count - request_step)
            if elapsed_steps >= len(chunk):
                self.get_logger().warning(
                    f"discarding expired chunk ({elapsed_steps} actions elapsed during inference)"
                )
                return

            aligned = np.asarray(chunk[elapsed_steps:], dtype=np.float64)
            blend_steps = min(self._args.blend_steps, len(self._buffer), len(aligned))
            if blend_steps:
                current = np.asarray(list(self._buffer), dtype=np.float64)
                for index in range(blend_steps):
                    weight = (index + 1) / blend_steps
                    aligned[index, :6] = (
                        current[index, :6] * (1.0 - weight) + aligned[index, :6] * weight
                    )
            elif self._last_action is not None:
                blend_steps = min(self._args.blend_steps, len(aligned))
                for index in range(blend_steps):
                    weight = (index + 1) / blend_steps
                    aligned[index, :6] = (
                        self._last_action[:6] * (1.0 - weight) + aligned[index, :6] * weight
                    )
            self._buffer = collections.deque(aligned.tolist())
            self.get_logger().info(
                f"aligned chunk buffered ({len(aligned)} actions, "
                f"dropped={elapsed_steps}, blended={blend_steps})"
            )

    def _map_gripper(self, predicted: float) -> float:
        if self._gripper_phase == "open":
            self._gripper_candidate_count = (
                self._gripper_candidate_count + 1 if predicted >= 0.20 else 0
            )
            if self._gripper_candidate_count >= self._args.gripper_confirm_steps:
                self._gripper_phase = "closed"
                self._gripper_candidate_count = 0
        elif self._gripper_phase == "closed":
            self._gripper_candidate_count = (
                self._gripper_candidate_count + 1 if predicted <= 0.10 else 0
            )
            if self._gripper_candidate_count >= self._args.gripper_confirm_steps:
                self._gripper_phase = "released"
                self._gripper_candidate_count = 0

        return 0.8 if self._gripper_phase == "closed" else 0.0

    def _publish_action(self, action: list[float]) -> None:
        gripper = self._map_gripper(float(action[6]))
        msg = JointTrajectory()
        msg.joint_names = POLICY_JOINTS
        point = JointTrajectoryPoint()
        point.positions = [*action[:6], gripper]
        point.time_from_start.sec = int(self._args.command_period)
        point.time_from_start.nanosec = int((self._args.command_period % 1.0) * 1_000_000_000)
        msg.points = [point]
        self._target_pub.publish(msg)
        self._last_action = np.asarray(action, dtype=np.float64)
        self._published_count += 1
        if self._published_count % self._args.diag_interval == 0:
            arm_delta = float("nan")
            if self._latest_state is not None:
                arm_delta = float(np.max(np.abs(np.asarray(action[:6]) - self._latest_state[:6])))
            self.get_logger().info(
                f"diag: buffer={len(self._buffer)} pred_gripper={action[6]:.3f} "
                f"cmd_gripper={gripper:.2f} gripper_phase={self._gripper_phase} "
                f"max_arm_target_delta={arm_delta:.3f} rad"
            )

    def _tick(self) -> None:
        self._consume_result()
        if not self._enabled:
            return
        history = self._history_snapshot()
        if history is None:
            return
        image_age = time.monotonic() - self._latest_image_time
        state_age = time.monotonic() - self._latest_state_time
        if image_age > self._args.max_observation_age or state_age > self._args.max_observation_age:
            now = time.monotonic()
            if now - self._last_stale_log > 1.0:
                self.get_logger().warning(f"stale observation (image={image_age:.3f}s, state={state_age:.3f}s); holding")
                self._last_stale_log = now
            return
        if self._buffer:
            self._publish_action(self._buffer.popleft())
        if (
            self._fetch_future is None
            and len(self._buffer) <= self._args.prefetch_threshold
        ):
            self._submit_prefetch(history)

    def destroy_node(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:27658")
    parser.add_argument("--env-prefix", default="env_0")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--request-timeout", type=float, default=3.0)
    parser.add_argument("--max-observation-age", type=float, default=0.25)
    parser.add_argument("--command-period", type=float, default=1.0 / 15.0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--prefetch-threshold", type=int, default=0)
    parser.add_argument("--blend-steps", type=int, default=4)
    parser.add_argument("--gripper-confirm-steps", type=int, default=3)
    parser.add_argument("--diag-interval", type=int, default=30)
    args = parser.parse_args()
    if args.fps <= 0 or args.request_timeout <= 0 or args.command_period <= 0:
        parser.error("fps, request-timeout, and command-period must be positive")
    if not 1 <= args.jpeg_quality <= 100 or args.chunk_size <= 0:
        parser.error("jpeg-quality must be 1..100 and chunk-size must be positive")
    if not 0 <= args.prefetch_threshold < args.chunk_size or args.diag_interval <= 0:
        parser.error("prefetch-threshold must be in [0, chunk-size) and diag-interval must be positive")
    if args.blend_steps < 0 or args.gripper_confirm_steps <= 0:
        parser.error("blend-steps must be non-negative and gripper-confirm-steps must be positive")
    return args


def main() -> None:
    rclpy.init()
    node = DiffusionPolicyExecutor(parse_args())
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
