#!/usr/bin/env python3
"""录制节点 — 订阅仿真数据，按 pick-and-place 循环切分 LeRobot 原始数据

订阅:
  /{env_prefix}/gemini2/rgb           1280x720 图像
  /{env_prefix}/gemini2/depth         深度 (米)
  /joint_states                       全关节状态
  /pick_and_place/_action/feedback    episode 边界 (current_step 2/12)

落盘 (每集一个目录):
  <output_dir>/episode_XXXXXX/
    frames_rgb/frame_000000.jpg ...
    frames_depth/frame_000000.png ...
    states.csv       7维状态 (6臂关节 + 左knuckle)
    meta.json        fps/prompt/帧数
"""

import csv
import json
import os
import shutil

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from custom_msgs.action import PickAndPlace

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
STATE_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]
PROMPT = "pick the red cube from the left tray and place it on the right tray"


class RecorderNode(Node):
    def __init__(self):
        super().__init__("recorder_node")

        self.declare_parameter("fps", 15.0)
        self.declare_parameter("output_dir", "/home/dev/work/ur5e_gripper/raw_data")
        self.declare_parameter("env_prefix", "env_0")
        self.declare_parameter("min_frames", 30)

        self._fps = float(self.get_parameter("fps").value)
        self._output_dir = str(self.get_parameter("output_dir").value)
        self._env = str(self.get_parameter("env_prefix").value)
        self._min_frames = int(self.get_parameter("min_frames").value)

        self._bridge = CvBridge()
        self._latest_rgb = None  # BGR ndarray
        self._latest_depth = None  # float32 ndarray (米)
        self._joint_pos = {}  # name -> pos

        # 录制状态
        self._recording = False
        self._episode_idx = self._scan_existing_episodes()
        self._frame_idx = 0
        self._episode_dir = None
        self._state_rows = []

        self.create_subscription(Image, f"/{self._env}/gemini2/rgb", self._rgb_cb, 5)
        self.create_subscription(Image, f"/{self._env}/gemini2/depth", self._depth_cb, 5)
        self.create_subscription(JointState, "/joint_states", self._joints_cb, 10)
        self.create_subscription(
            PickAndPlace.Impl.FeedbackMessage,
            "/pick_and_place/_action/feedback",
            self._feedback_cb,
            10,
        )

        self._timer = self.create_timer(1.0 / self._fps, self._tick)
        os.makedirs(self._output_dir, exist_ok=True)
        self.get_logger().info(
            f"Recorder ready: fps={self._fps}, output={self._output_dir}, "
            f"min_frames={self._min_frames}, next_episode={self._episode_idx:06d}"
        )
        self.get_logger().info("Waiting for /pick_and_place feedback (step 2) to start episodes...")

    def _scan_existing_episodes(self):
        """扫描已存在的 episode 目录，返回下一个可用编号（防重启覆盖）"""
        if not os.path.isdir(self._output_dir):
            return 0
        idxs = []
        for name in os.listdir(self._output_dir):
            if name.startswith("episode_"):
                try:
                    idxs.append(int(name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        return max(idxs) + 1 if idxs else 0

    # ------------------------------------------------------------------
    # 订阅回调
    # ------------------------------------------------------------------

    def _rgb_cb(self, msg: Image):
        try:
            self._latest_rgb = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"RGB error: {e}")

    def _depth_cb(self, msg: Image):
        try:
            self._latest_depth = self._bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            self.get_logger().warn(f"Depth error: {e}")

    def _joints_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in STATE_JOINTS:
                self._joint_pos[name] = float(pos)

    def _feedback_cb(self, msg):
        step = msg.feedback.current_step
        if step == 2:
            self._start_episode()
        elif step == 12:
            self._finish_episode()

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------

    def _tick(self):
        if not self._recording:
            return
        if self._latest_rgb is None or self._latest_depth is None:
            return
        missing = [j for j in STATE_JOINTS if j not in self._joint_pos]
        if missing:
            self.get_logger().warn(f"Joint state missing: {missing}, skipping frame")
            return

        rgb_path = os.path.join(self._episode_dir, "frames_rgb", f"frame_{self._frame_idx:06d}.jpg")
        depth_path = os.path.join(self._episode_dir, "frames_depth", f"frame_{self._frame_idx:06d}.png")

        cv2.imwrite(rgb_path, self._latest_rgb)
        depth_mm = np.nan_to_num(self._latest_depth, nan=0.0) * 1000.0
        cv2.imwrite(depth_path, np.clip(depth_mm, 0, 65535).astype(np.uint16))

        self._state_rows.append([self._joint_pos[j] for j in STATE_JOINTS])
        self._frame_idx += 1

    # ------------------------------------------------------------------
    # episode 管理
    # ------------------------------------------------------------------

    def _start_episode(self):
        if self._recording:
            self.get_logger().warn("step==1 while recording, closing previous episode")
            self._finish_episode()

        self._episode_dir = os.path.join(self._output_dir, f"episode_{self._episode_idx:06d}")
        os.makedirs(os.path.join(self._episode_dir, "frames_rgb"), exist_ok=True)
        os.makedirs(os.path.join(self._episode_dir, "frames_depth"), exist_ok=True)
        self._frame_idx = 0
        self._state_rows = []
        self._recording = True
        self.get_logger().info(f"▶ episode_{self._episode_idx:06d} started")

    def _finish_episode(self):
        if not self._recording:
            return
        self._recording = False

        if self._frame_idx < self._min_frames:
            shutil.rmtree(self._episode_dir, ignore_errors=True)
            self.get_logger().warn(
                f"✗ episode_{self._episode_idx:06d} too short ({self._frame_idx} < {self._min_frames}), dropped"
            )
        else:
            self._write_metadata()
            self.get_logger().info(f"■ episode_{self._episode_idx:06d} saved ({self._frame_idx} frames)")
        self._episode_idx += 1
        self._episode_dir = None

    def _write_metadata(self):
        with open(os.path.join(self._episode_dir, "states.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "timestamp_s"] + STATE_JOINTS)
            for i, row in enumerate(self._state_rows):
                writer.writerow([i, f"{i / self._fps:.4f}"] + [f"{v:.8f}" for v in row])

        meta = {
            "fps": self._fps,
            "prompt": PROMPT,
            "num_frames": self._frame_idx,
            "state_joints": STATE_JOINTS,
        }
        with open(os.path.join(self._episode_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def main():
    rclpy.init()
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._finish_episode()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
