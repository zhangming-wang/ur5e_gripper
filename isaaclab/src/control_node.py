"""UR5e Gripper 控制节点 — 订阅桥接节点发的轨迹，在 IsaacLab 中执行

IsaacLab 执行完成后 pub 完成信号，bridge 收到后完成 action。
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Bool
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory
import carb
import omni.appwindow


class ControlNode(Node):
    def __init__(self, isaaclab, joint_name_list):
        super().__init__("ur5e_control_node")

        self.isaaclab = isaaclab
        self.joint_name_list = list(joint_name_list)
        self.num_joints = len(self.joint_name_list)
        self.name_to_idx = {name: i for i, name in enumerate(self.joint_name_list)}

        # 初始关节角度 (rad)，run_isaaclab 启动后会覆盖为仿真实际值
        self.current_pos = np.zeros(self.num_joints, dtype=np.float64)
        self.target_pos = np.zeros(self.num_joints, dtype=np.float64)
        self.init_pos = np.zeros(self.num_joints, dtype=np.float64)
        # 安全默认值（通过关节名设置，不依赖索引）
        for name, val in [
            ("shoulder_lift_joint", -1.57),
            ("elbow_joint", 1.57),
            ("wrist_2_joint", 1.57),
        ]:
            if name in self.name_to_idx:
                self.init_pos[self.name_to_idx[name]] = val
                self.target_pos[self.name_to_idx[name]] = val

        # 轨迹执行
        self._traj_points = []
        self._traj_start_time = 0.0
        self._traj_active = False
        self._traj_settling = False  # 轨迹时间到，正在等稳定
        self._traj_settle_start = 0.0
        self._traj_stable_cnt = 0
        self._traj_prev_pos = None
        self._last_traj_joint_names = []

        # 夹爪
        self._gripper_target = 0.0
        self._gripper_done_sent = True
        self._gripper_stable_cnt = 0
        self._gripper_prev_pos = None
        self._motion_stopped = False

        qos = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE)

        # Pub: 物理状态
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", qos)
        # Pub: 执行完成信号
        self._traj_done_pub = self.create_publisher(Bool, "/isaaclab/trajectory_done", qos)
        self._gripper_done_pub = self.create_publisher(Bool, "/isaaclab/gripper_done", qos)

        # Sub: 桥接节点发的指令
        self.create_subscription(JointTrajectory, "/arm_controller/joint_trajectory", self._arm_traj_callback, qos)
        self.create_subscription(Float64, "/gripper_controller/command", self._gripper_callback, qos)
        self.create_subscription(Bool, "/isaaclab/stop_motion", self._stop_motion_callback, qos)

        # Service
        self.reset_service = self.create_service(Trigger, "/isaac_lab/reset", self.reset_callback)
        self.spawn_service = self.create_service(Trigger, "/spawn_cube", self._spawn_cube_callback)

        # 键盘
        self.appwindow = omni.appwindow.get_default_app_window()
        self.keyboard = self.appwindow.get_keyboard()
        self.inp = carb.input.acquire_input_interface()
        self.inp.subscribe_to_keyboard_events(self.keyboard, self.on_keyboard_event)

    # ------------------------------------------------------------------
    # 关节状态
    # ------------------------------------------------------------------

    def publish_joint_state(self):
        t = self.get_clock().now().to_msg()
        msg = JointState()
        msg.header.stamp = t
        msg.name = self.joint_name_list
        msg.position = self.current_pos.tolist()
        self.joint_state_pub.publish(msg)

    # ------------------------------------------------------------------
    # 轨迹
    # ------------------------------------------------------------------

    def _arm_traj_callback(self, msg: JointTrajectory):
        self._traj_points = []
        self._motion_stopped = False
        if not msg.points:
            self._traj_active = False
            return
        self._traj_settling = False
        self._last_traj_joint_names = msg.joint_names
        for p in msg.points:
            pos = np.array(self.target_pos)
            for jname, pval in zip(msg.joint_names, p.positions):
                if jname in self.name_to_idx:
                    pos[self.name_to_idx[jname]] = pval
            t = p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
            self._traj_points.append((pos, t))
        self._traj_start_time = time.time()
        self._traj_active = True
        self.get_logger().info(f"Trajectory: {len(msg.points)} pts, " f"duration={self._traj_points[-1][1]:.1f}s")

    def step_traj(self):
        """每物理帧调用，线性插值当前目标位置，结束后等稳定"""
        if self._motion_stopped:
            return
        if not self._traj_active:
            return

        if not self._traj_settling:
            elapsed = time.time() - self._traj_start_time
            prev_pos, prev_t = self._traj_points[0]

            if elapsed <= prev_t:
                self._apply_arm_positions(prev_pos)
                return

            for i in range(1, len(self._traj_points)):
                cur_pos, cur_t = self._traj_points[i]
                if elapsed <= cur_t:
                    alpha = (elapsed - prev_t) / (cur_t - prev_t)
                    self._apply_arm_positions(prev_pos + (cur_pos - prev_pos) * alpha)
                    return
                prev_pos, prev_t = cur_pos, cur_t

            self._apply_arm_positions(self._traj_points[-1][0])
            self._traj_points.clear()
            self._traj_settling = True
            self._traj_settle_start = time.time()
            self._traj_stable_cnt = 0
            self._traj_prev_pos = None
            return

        # 稳定检测：连续3帧位置不变 或 超时3秒
        if time.time() - self._traj_settle_start > 5.0:
            self.get_logger().warn("Arm settle timeout, force done")
            self._finish_traj()
            return

        cur = self.current_pos[self.name_to_idx[self._last_traj_joint_names[0]]]
        if self._traj_prev_pos is not None and abs(cur - self._traj_prev_pos) < 1e-4:
            self._traj_stable_cnt += 1
            if self._traj_stable_cnt >= 3:
                self._finish_traj()
                return
        else:
            self._traj_stable_cnt = 0
        self._traj_prev_pos = cur

    def _finish_traj(self):
        self._traj_active = False
        self._traj_settling = False
        errs = []
        for name in self._last_traj_joint_names:
            if name in self.name_to_idx:
                i = self.name_to_idx[name]
                e = abs(self.current_pos[i] - self.target_pos[i])
                errs.append(f"{name}={e:.4f}")
        self.get_logger().info(f"Trajectory done. Errors: {', '.join(errs)}")
        self._traj_done_pub.publish(Bool(data=True))

    def _apply_arm_positions(self, src):
        for name in self._last_traj_joint_names:
            if name in self.name_to_idx:
                self.target_pos[self.name_to_idx[name]] = src[self.name_to_idx[name]]

    # ------------------------------------------------------------------
    # 夹爪
    # ------------------------------------------------------------------

    def step_gripper(self):
        """每物理帧调用，设夹爪目标 + 不动了发完成信号"""
        if self._motion_stopped:
            return
        if self._gripper_done_sent:
            return
        all_gripper = [
            "robotiq_85_left_knuckle_joint",
            "robotiq_85_right_knuckle_joint",
            "robotiq_85_left_inner_knuckle_joint",
            "robotiq_85_right_inner_knuckle_joint",
            "robotiq_85_left_finger_tip_joint",
            "robotiq_85_right_finger_tip_joint",
        ]
        # 左右镜像：knuckle+inner 左正右负，finger_tip 反过来
        for jname in all_gripper:
            if jname in self.name_to_idx:
                if "tip" in jname:
                    v = self._gripper_target if "right" in jname else -self._gripper_target
                else:
                    v = -self._gripper_target if "right" in jname else self._gripper_target
                self.target_pos[self.name_to_idx[jname]] = v

        J = "robotiq_85_left_knuckle_joint"
        idx = self.name_to_idx.get(J)
        if idx is None:
            return
        cur = self.current_pos[idx]
        if self._gripper_prev_pos is not None and abs(cur - self._gripper_prev_pos) < 1e-5:
            self._gripper_stable_cnt += 1
            if self._gripper_stable_cnt >= 3 and not self._gripper_done_sent:
                self._gripper_done_pub.publish(Bool(data=True))
                self._gripper_done_sent = True
                self.get_logger().info("Gripper done")
        else:
            self._gripper_stable_cnt = 0
        self._gripper_prev_pos = cur

    def _gripper_callback(self, msg: Float64):
        self._motion_stopped = False
        self._gripper_target = msg.data
        self._gripper_done_sent = False
        self._gripper_stable_cnt = 0
        self._gripper_prev_pos = None

    def _stop_motion_callback(self, msg: Bool):
        if not msg.data:
            return
        self._motion_stopped = True
        self._traj_points.clear()
        self._traj_active = False
        self._traj_settling = False
        self._traj_prev_pos = None
        self._last_traj_joint_names = []
        self.target_pos = self.current_pos.copy()
        self._gripper_done_sent = True
        self._gripper_stable_cnt = 0
        self._gripper_prev_pos = None
        self._gripper_target = float(
            self.current_pos[self.name_to_idx["robotiq_85_left_knuckle_joint"]]
        ) if "robotiq_85_left_knuckle_joint" in self.name_to_idx else 0.0
        self.get_logger().warn("Motion stopped")

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def reset_callback(self, request, response):
        self._motion_stopped = False
        self._traj_active = False
        self._traj_settling = False
        self._traj_points.clear()
        self.isaaclab.reset_env()
        response.success = True
        response.message = "Reset done"
        return response

    def _spawn_cube_callback(self, request, response):
        result = self.isaaclab.spawn_cube()
        response.success = result["success"]
        response.message = result["message"]
        return response

    # ------------------------------------------------------------------
    # 键盘
    # ------------------------------------------------------------------

    def on_keyboard_event(self, event, *_):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                self._motion_stopped = False
                self._traj_active = False
                self._traj_points.clear()
                self.isaaclab.reset_env()
                print("[INFO]: 键盘 'R' — 重置仿真环境")
