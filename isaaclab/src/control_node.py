"""UR5e Gripper 控制节点 — 订阅桥接节点发的轨迹，在 IsaacLab 中执行

IsaacLab 执行完成后 pub 完成信号，bridge 收到后完成 action。
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Bool, UInt8
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory
import carb
import omni.appwindow

# 夹爪完成结果码（与 bridge_node 保持一致）
GRIPPER_FAILED = 0
GRIPPER_REACHED = 1
GRIPPER_STALLED = 2
GRIPPER_MASTER_JOINT = "robotiq_85_left_knuckle_joint"
GRIPPER_JOINTS = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]
ACT_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
ACT_COMMAND_TOPIC = "/isaaclab/act/joint_target"
ACT_ENABLED_TOPIC = "/isaaclab/act/enabled"


class ControlNode(Node):
    def __init__(self, isaaclab, joint_name_list, act_mode=False):
        super().__init__("ur5e_control_node")

        self.isaaclab = isaaclab
        self._act_mode = bool(act_mode)
        self.joint_name_list = list(joint_name_list)
        self.num_joints = len(self.joint_name_list)
        self.name_to_idx = {name: i for i, name in enumerate(self.joint_name_list)}

        # 初始关节角度 (rad)，run_isaaclab 启动后会覆盖为仿真实际值
        self.current_pos = np.zeros(self.num_joints, dtype=np.float64)
        self.current_vel = np.zeros(self.num_joints, dtype=np.float64)
        self.applied_torque = np.zeros(self.num_joints, dtype=np.float64)
        self.effort_limits = np.ones(self.num_joints, dtype=np.float64)
        self.target_pos = np.zeros(self.num_joints, dtype=np.float64)
        self.init_pos = np.zeros(self.num_joints, dtype=np.float64)
        # 指尖接触力 (N)，由 run_isaaclab 每物理帧通过 ContactSensor 更新
        self.gripper_contact_left = 0.0
        self.gripper_contact_right = 0.0
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
        self._sim_time = 0.0

        # 夹爪
        self._gripper_target = 0.0
        self._gripper_done_sent = True
        self._gripper_start_time = 0.0
        self._gripper_contact_cnt = 0
        self._motion_stopped = False

        # ACT direct-control state.  This path is intentionally separate from
        # the MoveIt trajectory state machine above.
        self._act_enabled = False
        self._act_target = self.target_pos.copy()
        self._act_start_target = self.target_pos.copy()
        self._act_command_start_time = 0.0
        self._act_last_command_time = float("-inf")
        self._act_gripper_start = 0.0
        self._act_gripper_target = 0.0
        self._act_stale_reported = False
        self._act_clamp_count = 0
        self._act_command_count = 0

        qos = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE)

        # 到位判定参数（可被 ros2 run --ros-args -p 覆盖）
        self.declare_parameter("arm_goal_tolerance", 0.005)
        self.declare_parameter("arm_stable_threshold", 1e-4)
        self.declare_parameter("arm_stable_samples", 3)
        self.declare_parameter("arm_settle_timeout", 5.0)
        self.declare_parameter("gripper_goal_tolerance", 0.01)
        self.declare_parameter("gripper_contact_force_threshold_n", 1.0)
        self.declare_parameter("gripper_contact_samples", 5)
        self.declare_parameter("gripper_settle_timeout", 5.0)
        self._arm_goal_tolerance = float(self.get_parameter("arm_goal_tolerance").value)
        self._arm_stable_threshold = float(self.get_parameter("arm_stable_threshold").value)
        self._arm_stable_samples = int(self.get_parameter("arm_stable_samples").value)
        self._arm_settle_timeout = float(self.get_parameter("arm_settle_timeout").value)
        self._gripper_goal_tolerance = float(self.get_parameter("gripper_goal_tolerance").value)
        self._gripper_contact_force_threshold_n = float(self.get_parameter("gripper_contact_force_threshold_n").value)
        self._gripper_contact_samples = int(self.get_parameter("gripper_contact_samples").value)
        self._gripper_settle_timeout = float(self.get_parameter("gripper_settle_timeout").value)

        self.declare_parameter("act_watchdog_sec", 0.5)
        self.declare_parameter("act_command_period", 1.0 / 15.0)
        self.declare_parameter("act_max_joint_step", 0.15)
        self._act_watchdog_sec = float(self.get_parameter("act_watchdog_sec").value)
        self._act_command_period = float(self.get_parameter("act_command_period").value)
        self._act_max_joint_step = float(self.get_parameter("act_max_joint_step").value)
        if self._act_watchdog_sec <= 0 or self._act_command_period <= 0 or self._act_max_joint_step <= 0:
            raise ValueError("ACT watchdog, command period, and max joint step must be positive")

        # Pub: 物理状态
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", qos)
        self._sim_time_pub = self.create_publisher(Float64, "/isaaclab/sim_time", qos)
        # Pub: 执行完成信号
        self._traj_done_pub = self.create_publisher(Bool, "/isaaclab/trajectory_done", qos)
        self._gripper_done_pub = self.create_publisher(UInt8, "/isaaclab/gripper_done", qos)

        # Sub: ordinary bridge commands, or the isolated ACT command path.
        if self._act_mode:
            self.create_subscription(JointTrajectory, ACT_COMMAND_TOPIC, self._act_callback, qos)
            self.create_subscription(Bool, ACT_ENABLED_TOPIC, self._act_enabled_callback, qos)
        else:
            self.create_subscription(
                JointTrajectory, "/arm_controller/joint_trajectory", self._arm_traj_callback, qos
            )
            self.create_subscription(Float64, "/gripper_controller/command", self._gripper_callback, qos)
        self.create_subscription(Bool, "/stop_motion", self._stop_motion_callback, qos)
        self.create_subscription(Bool, "/isaaclab/stop_motion", self._stop_motion_callback, qos)

        # Service
        self.reset_service = self.create_service(Trigger, "/reset", self.reset_callback)
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
        if self.current_vel is not None:
            msg.velocity = self.current_vel.tolist()
        if self.applied_torque is not None:
            msg.effort = self.applied_torque.tolist()
        self.joint_state_pub.publish(msg)

    # ------------------------------------------------------------------
    # 轨迹
    # ------------------------------------------------------------------

    def _act_enabled_callback(self, msg: Bool):
        enabled = bool(msg.data)
        if enabled:
            if not self._act_enabled:
                self._motion_stopped = False
                self._act_enabled = True
                self._act_target = self.current_pos.copy()
                self._act_start_target = self.current_pos.copy()
                self._act_gripper_start = float(
                    self.current_pos[self.name_to_idx[GRIPPER_MASTER_JOINT]]
                ) if GRIPPER_MASTER_JOINT in self.name_to_idx else 0.0
                self._act_gripper_target = self._act_gripper_start
                self._act_command_start_time = self._sim_time
                self._act_last_command_time = float("-inf")
                self._act_stale_reported = False
        else:
            self._act_enabled = False
            self._act_last_command_time = float("-inf")
            self.target_pos = self.current_pos.copy()

    def _act_callback(self, msg: JointTrajectory):
        if not self._act_mode:
            return
        if not msg.points or len(msg.points) != 1:
            self.get_logger().error("ACT command must contain exactly one point")
            return
        if (
            len(msg.joint_names) != len(set(msg.joint_names))
            or set(msg.joint_names) != set(ACT_ARM_JOINTS + [GRIPPER_MASTER_JOINT])
        ):
            self.get_logger().error("ACT command has an unexpected joint-name set")
            return

        point = msg.points[0]
        if len(point.positions) != len(msg.joint_names):
            self.get_logger().error("ACT command positions do not match joint names")
            return
        values = dict(zip(msg.joint_names, point.positions))
        if any(not np.isfinite(float(value)) for value in values.values()):
            self.get_logger().error("Rejected non-finite ACT command")
            return

        target = self._act_target.copy()
        limits = {
            "shoulder_pan_joint": (-2.0 * np.pi, 2.0 * np.pi),
            "shoulder_lift_joint": (-2.0 * np.pi, 2.0 * np.pi),
            "elbow_joint": (-np.pi, np.pi),
            "wrist_1_joint": (-2.0 * np.pi, 2.0 * np.pi),
            "wrist_2_joint": (-2.0 * np.pi, 2.0 * np.pi),
            "wrist_3_joint": (-2.0 * np.pi, 2.0 * np.pi),
        }
        clamped = False
        for name in ACT_ARM_JOINTS:
            idx = self.name_to_idx[name]
            requested = float(values[name])
            lower, upper = limits[name]
            bounded = float(np.clip(requested, lower, upper))
            reference = float(self.current_pos[idx])
            clipped_step = float(
                np.clip(bounded - reference, -self._act_max_joint_step, self._act_max_joint_step)
            )
            if abs(bounded - requested) > 1e-9 or abs(clipped_step - (bounded - reference)) > 1e-9:
                clamped = True
            target[idx] = reference + clipped_step

        gripper_idx = self.name_to_idx.get(GRIPPER_MASTER_JOINT)
        if gripper_idx is None:
            self.get_logger().error("ACT command cannot find the gripper master joint")
            return
        target[gripper_idx] = float(np.clip(float(values[GRIPPER_MASTER_JOINT]), 0.0, 0.8))

        self._act_start_target = self.target_pos.copy()
        self._act_target = target
        self._act_gripper_start = float(self.target_pos[gripper_idx])
        self._act_gripper_target = float(target[gripper_idx])
        self._act_command_start_time = self._sim_time
        self._act_last_command_time = self._sim_time
        self._act_stale_reported = False

        self._act_command_count += 1
        if clamped:
            self._act_clamp_count += 1
        if self._act_command_count % 150 == 0:
            self.get_logger().info(
                f"ACT clamp observation: {self._act_clamp_count}/{self._act_command_count} "
                f"commands limited by act_max_joint_step={self._act_max_joint_step:.3f} rad"
            )

    def _apply_gripper_drive_targets(self, master_target):
        """Mirror the logical gripper command into the independent USD drives."""
        for name in GRIPPER_JOINTS:
            if name not in self.name_to_idx:
                continue
            if "tip" in name:
                value = master_target if "right" in name else -master_target
            else:
                value = -master_target if "right" in name else master_target
            self.target_pos[self.name_to_idx[name]] = value

    def step_act(self):
        """Apply the latest ACT target directly, with simulation-time safety guards.

        空闲/超时时必须保留 ``target_pos``（上一帧插值目标），不能写成
        ``current_pos``：把位置目标设成实测位置会让 PD 误差归零、失去抗重力力矩，
        而每帧重设会持续跟随下沉，表现为机械臂缓慢下垂。
        """
        if not self._act_mode:
            return
        if not self._act_enabled:
            return
        if self._sim_time - self._act_last_command_time > self._act_watchdog_sec:
            if not self._act_stale_reported:
                self.get_logger().warning(
                    f"ACT command watchdog expired after {self._act_watchdog_sec:.3f}s; holding position"
                )
                self._act_stale_reported = True
            return

        alpha = np.clip(
            (self._sim_time - self._act_command_start_time) / self._act_command_period,
            0.0,
            1.0,
        )
        for name in ACT_ARM_JOINTS:
            idx = self.name_to_idx[name]
            self.target_pos[idx] = self._act_start_target[idx] + (
                self._act_target[idx] - self._act_start_target[idx]
            ) * alpha

        gripper_idx = self.name_to_idx[GRIPPER_MASTER_JOINT]
        gripper_target = self._act_gripper_start + (
            self._act_gripper_target - self._act_gripper_start
        ) * alpha
        self.target_pos[gripper_idx] = gripper_target
        self._apply_gripper_drive_targets(gripper_target)

    def _arm_traj_callback(self, msg: JointTrajectory):
        if self._act_mode:
            self.get_logger().warning("Ignoring ordinary arm trajectory while ACT mode is active")
            return
        self._traj_points = []
        self._motion_stopped = False
        self._last_traj_joint_names = list(msg.joint_names)
        if (
            not msg.points
            or len(set(msg.joint_names)) != len(msg.joint_names)
            or any(name not in self.name_to_idx for name in msg.joint_names)
        ):
            self.get_logger().error("Rejected invalid or empty arm trajectory")
            self._traj_active = False
            self._traj_done_pub.publish(Bool(data=False))
            return
        self._traj_settling = False
        previous_t = -1.0
        for p in msg.points:
            if len(p.positions) != len(msg.joint_names):
                self.get_logger().error("Rejected arm trajectory with mismatched positions")
                self._traj_active = False
                self._traj_done_pub.publish(Bool(data=False))
                return
            pos = np.array(self.target_pos)
            for jname, pval in zip(msg.joint_names, p.positions):
                if jname in self.name_to_idx:
                    pos[self.name_to_idx[jname]] = pval
            t = p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
            if t <= previous_t:
                self.get_logger().error("Rejected arm trajectory with non-increasing times")
                self._traj_active = False
                self._traj_done_pub.publish(Bool(data=False))
                return
            previous_t = t
            self._traj_points.append((pos, t))
        if self._traj_points[0][1] > 0.0:
            self._traj_points.insert(0, (np.array(self.current_pos), 0.0))
        self._traj_start_time = self._sim_time
        self._traj_active = True
        self.get_logger().info(f"Trajectory: {len(msg.points)} pts, " f"duration={self._traj_points[-1][1]:.1f}s")

    def step_traj(self):
        """每物理帧调用，按仿真时间插值当前目标位置，结束后等稳定。"""
        if self._motion_stopped:
            return
        if not self._traj_active:
            return

        if not self._traj_settling:
            elapsed = self._sim_time - self._traj_start_time
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
            self._traj_settle_start = self._sim_time
            self._traj_stable_cnt = 0
            self._traj_prev_pos = None
            return

        # 稳定检测：所有被命令关节连续 N 帧稳定，或超时。
        if self._sim_time - self._traj_settle_start > self._arm_settle_timeout:
            self.get_logger().error("Arm settle timeout")
            self._finish_traj(False)
            return

        current = np.array([
            self.current_pos[self.name_to_idx[name]]
            for name in self._last_traj_joint_names
            if name in self.name_to_idx
        ])
        target = np.array([
            self.target_pos[self.name_to_idx[name]]
            for name in self._last_traj_joint_names
            if name in self.name_to_idx
        ])
        if not current.size:
            self._finish_traj(False)
            return
        if self._traj_prev_pos is not None and np.max(np.abs(current - self._traj_prev_pos)) < self._arm_stable_threshold:
            self._traj_stable_cnt += 1
            if self._traj_stable_cnt >= self._arm_stable_samples and np.max(np.abs(current - target)) < self._arm_goal_tolerance:
                self._finish_traj(True)
                return
        else:
            self._traj_stable_cnt = 0
        self._traj_prev_pos = current

    def _finish_traj(self, success):
        self._traj_active = False
        self._traj_settling = False
        errs = []
        for name in self._last_traj_joint_names:
            if name in self.name_to_idx:
                i = self.name_to_idx[name]
                e = abs(self.current_pos[i] - self.target_pos[i])
                errs.append(f"{name}={e:.4f}")
        level = self.get_logger().info if success else self.get_logger().error
        level(f"Trajectory {'done' if success else 'failed'}. Errors: {', '.join(errs)}")
        self._traj_done_pub.publish(Bool(data=success))

    def _apply_arm_positions(self, src):
        for name in self._last_traj_joint_names:
            if name in self.name_to_idx:
                self.target_pos[self.name_to_idx[name]] = src[self.name_to_idx[name]]

    # ------------------------------------------------------------------
    # 夹爪
    # ------------------------------------------------------------------

    def step_gripper(self):
        """每物理帧调用，设夹爪目标，按到位/接触判定完成。

        仅以 robotiq_85_left_knuckle_joint 作为逻辑夹爪关节：
        - 到达目标容差内 → REACHED
        - 闭合时左右指尖接触力都超过阈值并持续数个物理帧 → STALLED（夹住物体）
        - 超时/停止/遥测无效 → FAILED
        """
        if self._motion_stopped:
            return
        if self._gripper_done_sent:
            return
        if GRIPPER_MASTER_JOINT not in self.name_to_idx:
            self.get_logger().error("No gripper joints are available")
            self._finish_gripper(GRIPPER_FAILED, "no gripper joint")
            return
        idx = self.name_to_idx[GRIPPER_MASTER_JOINT]
        # The fallback USD has no physical mimic constraints, so mirror the
        # master target across the six independent joint drives.
        self._apply_gripper_drive_targets(self._gripper_target)

        current = float(self.current_pos[idx])
        target = float(self.target_pos[idx])

        error = abs(current - target)

        # 双指接触力检测：接触优先于位置到达，这样小物体也会被报告为 grasp。
        closing = self._gripper_target > 0.0
        contact_ok = (
            closing
            and self.gripper_contact_left >= self._gripper_contact_force_threshold_n
            and self.gripper_contact_right >= self._gripper_contact_force_threshold_n
        )
        if contact_ok:
            self._gripper_contact_cnt += 1
            if self._gripper_contact_cnt >= self._gripper_contact_samples:
                self._finish_gripper(
                    GRIPPER_STALLED,
                    f"contact pos={current:.4f} "
                    f"F_left={self.gripper_contact_left:.2f}N "
                    f"F_right={self.gripper_contact_right:.2f}N",
                )
                return
        else:
            self._gripper_contact_cnt = 0

        # 空载闭合或打开时仍按位置报告成功。
        if error <= self._gripper_goal_tolerance:
            self._finish_gripper(GRIPPER_REACHED, f"reached pos={current:.4f}")
            return

        # 超时（带诊断）
        if self._sim_time - self._gripper_start_time > self._gripper_settle_timeout:
            self._finish_gripper(
                GRIPPER_FAILED,
                f"settle timeout pos={current:.4f} "
                f"F_left={self.gripper_contact_left:.2f}N "
                f"F_right={self.gripper_contact_right:.2f}N "
                f"contact_cnt={self._gripper_contact_cnt}/{self._gripper_contact_samples}",
            )
            return

    def _finish_gripper(self, code, note=""):
        if self._gripper_done_sent:
            return
        self._gripper_done_sent = True
        label = {GRIPPER_FAILED: "failed", GRIPPER_REACHED: "reached", GRIPPER_STALLED: "stalled"}[code]
        self.get_logger().info(f"Gripper {label} ({note})")
        self._gripper_done_pub.publish(UInt8(data=code))

    def _gripper_callback(self, msg: Float64):
        if self._act_mode:
            self.get_logger().warning("Ignoring ordinary gripper command while ACT mode is active")
            return
        self._motion_stopped = False
        self._gripper_target = msg.data
        self._gripper_done_sent = False
        self._gripper_contact_cnt = 0
        self._gripper_start_time = self._sim_time

    def _stop_motion_callback(self, msg: Bool):
        if not msg.data:
            return
        if self._act_mode:
            self._motion_stopped = True
            self._act_enabled = False
            self._act_last_command_time = float("-inf")
            self.target_pos = self.current_pos.copy()
            self.get_logger().warn("ACT motion stopped")
            return
        arm_was_active = self._traj_active or self._traj_settling
        gripper_was_active = not self._gripper_done_sent
        self._motion_stopped = True
        self._traj_points.clear()
        self._traj_active = False
        self._traj_settling = False
        self._traj_prev_pos = None
        self._last_traj_joint_names = []
        self.target_pos = self.current_pos.copy()
        self._gripper_done_sent = True
        self._gripper_contact_cnt = 0
        self._gripper_target = float(
            self.current_pos[self.name_to_idx[GRIPPER_MASTER_JOINT]]
        ) if GRIPPER_MASTER_JOINT in self.name_to_idx else 0.0
        if arm_was_active:
            self._traj_done_pub.publish(Bool(data=False))
        if gripper_was_active:
            self._gripper_done_pub.publish(UInt8(data=GRIPPER_FAILED))
        self.get_logger().warn("Motion stopped")

    def advance_sim_time(self, dt):
        self._sim_time += float(dt)
        self._sim_time_pub.publish(Float64(data=self._sim_time))

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def reset_callback(self, request, response):
        try:
            self._motion_stopped = False
            self._traj_active = False
            self._traj_settling = False
            self._traj_points.clear()
            self._act_enabled = False
            self._act_last_command_time = float("-inf")
            self._gripper_done_sent = True
            self._gripper_start_time = self._sim_time
            self.isaaclab.reset_env()
            response.success = True
            response.message = "Reset done"
        except Exception as exc:
            self.get_logger().error(f"Reset failed: {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _spawn_cube_callback(self, request, response):
        try:
            result = self.isaaclab.spawn_cube()
            response.success = bool(result["success"])
            response.message = result["message"]
        except Exception as exc:
            self.get_logger().error(f"Cube spawn failed: {exc}")
            response.success = False
            response.message = str(exc)
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
