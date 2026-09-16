#!/usr/bin/env python3
"""UR5e + Gripper 调试面板 — 关节滑块 + 夹爪滑块 + 点位管理"""

import sys
import signal
import math
from datetime import datetime

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from custom_msgs.action import PickAndPlace
from custom_msgs.srv import DetectObject, PlanExecute
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from pymoveit2 import MoveIt2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QGroupBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QSizePolicy, QPushButton, QInputDialog, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor

ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


class RosClient(Node):
    def __init__(self):
        super().__init__("gui_debug_panel")
        self.cli = self.create_client(PlanExecute, "/plan_execute")
        self.spawn_cli = self.create_client(Trigger, "/spawn_cube")
        self.reset_cli = self.create_client(Trigger, "/reset")
        self.detect_cli = self.create_client(DetectObject, "/detect_object")
        self.pick_client = ActionClient(self, PickAndPlace, "/pick_and_place")
        self._pick_active = False
        self._pick_goal_handle = None
        self._pick_goal_future = None
        self._pick_cancel_requested = False
        self._pick_done_callback = None
        self._pick_feedback_callback = None

        # 实时关节状态
        self.joint_positions = {}     # name → rad
        self.gripper_position = 0.0   # 夹爪开度
        self.js_received = False      # 首次收到 /joint_states
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)

        # FK 计算器
        self._fk = MoveIt2(
            self, joint_names=ARM_JOINT_NAMES,
            base_link_name="base_link", end_effector_name="tool0",
            group_name="ur_manipulator",
        )

    def _js_cb(self, msg):
        self.js_received = True
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos
            # 夹爪用 left_knuckle 近似
            if name == "robotiq_85_left_knuckle_joint":
                self.gripper_position = abs(pos)

    def _call(self, cmd, data):
        if not self.cli.wait_for_service(timeout_sec=0.5):
            return False, "service not ready"
        req = PlanExecute.Request()
        req.command_type = cmd
        req.data = data
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        res = future.result()
        return (res.success, res.message) if res else (False, "")

    def spawn_cube(self):
        if not self.spawn_cli.wait_for_service(timeout_sec=0.5):
            return False, "/spawn_cube service not ready"
        future = self.spawn_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        res = future.result()
        return (res.success, res.message) if res else (False, "spawn service timeout")

    def reset_sim(self):
        """仿真复位：机械臂回 home 并删除方块（各后端通用）。"""
        if not self.reset_cli.wait_for_service(timeout_sec=0.5):
            return False, "/reset service not ready"
        future = self.reset_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        res = future.result()
        return (res.success, res.message) if res else (False, "reset service timeout")

    def detect_cube(self):
        if not self.detect_cli.wait_for_service(timeout_sec=0.5):
            return False, "/detect_object service not ready"
        future = self.detect_cli.call_async(DetectObject.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        res = future.result()
        if res is None:
            return False, "detect service timeout"
        if not res.detected:
            return False, "cube not detected"
        return True, (
            f"cube: x={res.x:.3f}, y={res.y:.3f}, z={res.z:.3f}, "
            f"yaw={math.degrees(res.yaw):.1f} deg"
        )

    def start_pick_and_place(self, on_done, on_feedback):
        if self._pick_active:
            return False, "pick-and-place already running"
        if not self.pick_client.wait_for_server(timeout_sec=0.5):
            return False, "/pick_and_place action not ready"

        self._pick_active = True
        self._pick_cancel_requested = False
        self._pick_done_callback = on_done
        self._pick_feedback_callback = on_feedback
        try:
            future = self.pick_client.send_goal_async(
                PickAndPlace.Goal(), feedback_callback=self._pick_feedback)
            self._pick_goal_future = future
            future.add_done_callback(self._pick_goal_response)
        except Exception as exc:
            self._finish_pick(False, str(exc))
            return False, str(exc)
        return True, "pick-and-place started"

    def cancel_pick_and_place(self):
        if not self._pick_active:
            return False, "no pick-and-place action is running"
        self._pick_cancel_requested = True
        if self._pick_goal_handle is not None:
            try:
                self._pick_goal_handle.cancel_goal_async()
            except Exception as exc:
                return False, str(exc)
        return True, "pick-and-place cancellation requested"

    def _pick_feedback(self, feedback_message):
        if self._pick_feedback_callback is not None:
            feedback = feedback_message.feedback
            self._pick_feedback_callback(feedback.current_step, feedback.status)

    def _pick_goal_response(self, future):
        self._pick_goal_future = None
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._finish_pick(False, str(exc))
            return
        if not goal_handle.accepted:
            self._finish_pick(False, "pick-and-place goal rejected")
            return
        self._pick_goal_handle = goal_handle
        if self._pick_cancel_requested:
            try:
                goal_handle.cancel_goal_async()
            except Exception as exc:
                self._finish_pick(False, str(exc))
                return
        goal_handle.get_result_async().add_done_callback(self._pick_result)

    def _pick_result(self, future):
        try:
            result = future.result().result
            self._finish_pick(result.success, result.message)
        except Exception as exc:
            self._finish_pick(False, str(exc))

    def _finish_pick(self, success, message):
        callback = self._pick_done_callback
        self._pick_active = False
        self._pick_goal_handle = None
        self._pick_goal_future = None
        self._pick_cancel_requested = False
        self._pick_done_callback = None
        self._pick_feedback_callback = None
        if callback is not None:
            callback(success, message)

    def fk_abs(self, joints_deg):
        """joints_deg: 6 个关节角(度)"""
        data = " ".join(f"{j:.3f}" for j in joints_deg)
        return self._call("fk_abs", data)

    def ik_abs(self, x, y, z, roll, pitch, yaw):
        """绝对位姿: pos(米), rpy(度)"""
        data = f"{x:.5f} {y:.5f} {z:.5f} {roll:.3f} {pitch:.3f} {yaw:.3f}"
        return self._call("ik_abs", data)

    def gripper(self, pos):
        return self._call("gripper", f"{pos:.3f}")

    def get_current_arm_deg(self):
        """返回当前 6 个臂关节角(度)，读不到则返回 None"""
        try:
            return [math.degrees(self.joint_positions[n]) for n in ARM_JOINT_NAMES]
        except KeyError:
            return None

    def compute_fk(self, joints_deg):
        """返回当前 tool0 位姿: (x,y,z, roll,pitch,yaw) 或 None"""
        try:
            joints_rad = [math.radians(j) for j in joints_deg]
            fk = self._fk.compute_fk(joints_rad, fk_link_names=["tool0"])
            if fk is None:
                return None
            cur = fk[0] if isinstance(fk, list) else fk
            p = cur.pose.position
            q = cur.pose.orientation
            rpy = self._quat_to_rpy(q.x, q.y, q.z, q.w)
            return (p.x, p.y, p.z, math.degrees(rpy[0]), math.degrees(rpy[1]), math.degrees(rpy[2]))
        except Exception:
            return None

    @staticmethod
    def _quat_to_rpy(x, y, z, w):
        sinr, cosr = 2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0*(w*y - z*x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny, cosy = 2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)
        yaw = math.atan2(siny, cosy)
        return (roll, pitch, yaw)


class MainWindow(QMainWindow):
    LOOP_INTERVAL_MS = 500
    ARM_JOINTS = [
        ("joint1", -360, 360, 0),
        ("joint2", -360, 360, -90),
        ("joint3", -360, 360, 90),
        ("joint4", -360, 360, 0),
        ("joint5", -360, 360, 90),
        ("joint6", -360, 360, 0),
    ]

    def __init__(self, ros_node):
        super().__init__()
        self.ros = ros_node
        self.setWindowTitle("UR5e + Gripper 调试面板")
        self._shutting_down = False
        self._loop_active = False
        self._loop_cycle = 0
        self._loop_timer = QTimer(self)
        self._loop_timer.setSingleShot(True)
        self._loop_timer.timeout.connect(self._start_loop_cycle)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ==================== 6 轴 ====================
        arm = QGroupBox("机械臂 — 关节角")
        arm_layout = QVBoxLayout(arm)

        self.joint_sliders = []
        for name, lo, hi, default in self.ARM_JOINTS:
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(lbl)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(default)
            slider.setTickInterval(30)
            val = QLabel(f"{default:+.0f}°")
            val.setFixedWidth(40)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            slider.valueChanged.connect(lambda v, l=val: l.setText(f"{v:+.0f}°"))
            slider.sliderReleased.connect(self._on_arm_change)
            row.addWidget(slider)
            row.addWidget(val)
            self.joint_sliders.append(slider)
            arm_layout.addLayout(row)

        root.addWidget(arm)

        # ==================== 夹爪 ====================
        grip = QGroupBox("夹爪 (0.0=张开  0.8=闭合)")
        grip_layout = QVBoxLayout(grip)

        row = QHBoxLayout()
        self.grip_label = QLabel("0.40")
        self.grip_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.grip_label.setFixedWidth(50)
        row.addWidget(self.grip_label)
        row.addWidget(QLabel("(0.0 ~ 0.8)"))

        self.grip_slider = QSlider(Qt.Horizontal)
        self.grip_slider.setRange(0, 800)
        self.grip_slider.setValue(400)
        self.grip_slider.valueChanged.connect(lambda v: self.grip_label.setText(f"{v/1000:.2f}"))
        self.grip_slider.sliderReleased.connect(self._on_gripper_change)
        self.grip_slider.setTickPosition(QSlider.TicksBelow)
        self.grip_slider.setTickInterval(100)

        grip_layout.addWidget(self.grip_slider)
        grip_layout.addLayout(row)
        root.addWidget(grip)

        # ==================== 点位 ====================
        wp_group = QGroupBox("点位")
        wp_outer = QHBoxLayout(wp_group)

        self.wp_list = QListWidget()
        self.wp_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.wp_list.setEditTriggers(QAbstractItemView.DoubleClicked)
        wp_outer.addWidget(self.wp_list, stretch=1)

        btn_layout = QVBoxLayout()
        btn_layout.setAlignment(Qt.AlignTop)
        for text, slot in [
            ("添加", self._add_waypoint),
            ("删除", self._del_waypoint),
            ("移动", self._move_waypoint),
        ]:
            btn = QPushButton(text)
            btn.setFixedWidth(60)
            btn.setMinimumHeight(30)
            btn.clicked.connect(slot)
            btn_layout.addWidget(btn)

        btn_layout.addStretch()

        for text, slot in [
            ("复位", self._on_reset),
            ("刷新", self._on_refresh),
        ]:
            btn = QPushButton(text)
            btn.setFixedWidth(60)
            btn.setMinimumHeight(30)
            btn.clicked.connect(slot)
            btn_layout.addWidget(btn)
        wp_outer.addLayout(btn_layout)

        root.addWidget(wp_group, stretch=1)

        # ==================== 检测抓取 ====================
        debug_group = QGroupBox("检测抓取")
        debug_layout = QHBoxLayout(debug_group)
        self.debug_buttons = []
        buttons = [
            ("生成 Cube", self._on_spawn_cube),
            ("检测", self._on_detect_cube),
        ]
        for text, slot in buttons:
            button = QPushButton(text)
            button.setMinimumHeight(32)
            button.clicked.connect(slot)
            debug_layout.addWidget(button, stretch=1)
            self.debug_buttons.append(button)
        self.single_pick_button = QPushButton("单次抓取")
        self.single_pick_button.setMinimumHeight(32)
        self.single_pick_button.clicked.connect(self._on_pick_cube)
        debug_layout.addWidget(self.single_pick_button, stretch=1)
        self.debug_buttons.append(self.single_pick_button)

        self.loop_pick_button = QPushButton("循环抓取")
        self.loop_pick_button.setMinimumHeight(32)
        self.loop_pick_button.clicked.connect(self._on_loop_pick_cube)
        debug_layout.addWidget(self.loop_pick_button, stretch=1)
        self.debug_buttons.append(self.loop_pick_button)
        root.addWidget(debug_group)

        # ==================== 日志 ====================
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log = QListWidget()
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log.setSizePolicy(sp)
        self.log.setAlternatingRowColors(False)
        self.log.setSelectionMode(QAbstractItemView.NoSelection)
        log_layout.addWidget(self.log)
        root.addWidget(log_group)

    # ==================================================================
    # 点位操作
    # ==================================================================

    def _get_current_pose_str(self):
        """FK 算当前关节角对应的位姿，返回 'x,y,z,roll,pitch,yaw,gripper' 字符串"""
        joints = [s.value() for s in self.joint_sliders]
        grip = self.grip_slider.value() / 1000.0
        pose = self.ros.compute_fk(joints)
        if pose is None:
            self._log("FK 计算失败，使用关节角代替", "warn")
            return f"{joints[0]},{joints[1]},{joints[2]},{joints[3]},{joints[4]},{joints[5]},{grip:.3f} (joints)"
        x, y, z, r, p, yw = pose
        return f"{x:.4f},{y:.4f},{z:.4f},{r:.1f},{p:.1f},{yw:.1f},{grip:.3f}"

    def _add_waypoint(self):
        text = self._get_current_pose_str()
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.wp_list.addItem(item)
        self._log(f"点位 {self.wp_list.count()}: {text}", "info")

    def _del_waypoint(self):
        for item in self.wp_list.selectedItems():
            row = self.wp_list.row(item)
            self.wp_list.takeItem(row)
            self._log(f"删除点位 #{row+1}", "info")

    def _move_waypoint(self):
        sel = self.wp_list.selectedItems()
        if not sel:
            self._log("未选中点位", "warn")
            return
        text = sel[0].text()
        parts = text.replace(" (joints)", "").split(",")
        if len(parts) < 7:
            self._log(f"格式错误: {text}", "error")
            return
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            self._log(f"解析失败: {text}", "error")
            return

        x, y, z, roll, pitch, yaw, grip_pos = vals[:7]
        # 移动机械臂
        self._send(f"IK 绝对 → {x:.3f},{y:.3f},{z:.3f} rpy:{roll:.1f},{pitch:.1f},{yaw:.1f}",
                   self.ros.ik_abs, x, y, z, roll, pitch, yaw)
        # 移动夹爪
        self._send(f"夹爪 → {grip_pos:.3f}", self.ros.gripper, grip_pos)
        # 同步滑块
        self.grip_slider.setValue(int(grip_pos * 1000))

    def _stop(self):
        self._log("停止: 暂不支持轨迹中断", "warn")

    # ==================================================================
    # 通用
    # ==================================================================

    def closeEvent(self, event):
        self._shutting_down = True
        self._loop_active = False
        self._loop_timer.stop()
        if self.ros._pick_active:
            self.ros.cancel_pick_and_place()
        super().closeEvent(event)

    def _log(self, text, level="info"):
        ts = datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"{ts}  {text}")
        colors = {"info": "#000", "warn": "#b8860b", "error": "#c00"}
        item.setForeground(QColor(colors.get(level, "#000")))
        self.log.insertItem(0, item)
        if self.log.count() > 200:
            self.log.takeItem(self.log.count() - 1)

    def _send(self, label, fn, *args):
        if self._shutting_down:
            return
        self._log(label, "info")
        QApplication.processEvents()
        try:
            ok, msg = fn(*args)
        except Exception as e:
            ok, msg = False, str(e)
            self._log(f"[ERROR] {msg}", "error")
            return
        if ok:
            self._log(f"[OK] {msg}", "info")
        else:
            self._log(f"[FAIL] {msg}", "error")

    def _on_spawn_cube(self):
        self._send("生成 Cube", self.ros.spawn_cube)

    def _on_detect_cube(self):
        self._send("检测方块", self.ros.detect_cube)

    def _set_debug_buttons_enabled(self, enabled):
        for button in self.debug_buttons:
            button.setEnabled(enabled)

    def _reset_pick_ui(self):
        self._loop_active = False
        self._loop_timer.stop()
        self._set_debug_buttons_enabled(True)
        self.loop_pick_button.setText("循环抓取")

    def _start_pick_action(self):
        if self._shutting_down or self.ros._pick_active:
            return
        if self._loop_active:
            self._loop_cycle += 1
            self._log(f"开始循环抓取第 {self._loop_cycle} 轮...", "info")
        else:
            self._log("开始单次抓取...", "info")

        started, message = self.ros.start_pick_and_place(
            self._on_pick_done, self._on_pick_feedback
        )
        if started:
            self._log(f"[OK] {message}", "info")
            return

        self._log(f"[FAIL] {message}", "error")
        self._reset_pick_ui()

    def _on_pick_cube(self):
        if self._shutting_down or self.ros._pick_active or self._loop_active:
            return
        self._set_debug_buttons_enabled(False)
        self._start_pick_action()

    def _on_loop_pick_cube(self):
        if self._shutting_down:
            return

        if self._loop_active:
            self._loop_active = False
            self._loop_timer.stop()
            self._set_debug_buttons_enabled(False)
            self.loop_pick_button.setText("停止中...")
            canceled, message = self.ros.cancel_pick_and_place()
            self._log(message, "info" if canceled else "warn")
            if not self.ros._pick_active:
                self._reset_pick_ui()
            return

        if self.ros._pick_active:
            return
        self._loop_active = True
        self._loop_cycle = 0
        self.loop_pick_button.setText("停止循环")
        self._set_debug_buttons_enabled(False)
        self.loop_pick_button.setEnabled(True)
        self._start_pick_action()

    def _start_loop_cycle(self):
        if not self._loop_active or self._shutting_down:
            return
        if self.ros._pick_active:
            return
        self._set_debug_buttons_enabled(False)
        self.loop_pick_button.setEnabled(True)
        self.loop_pick_button.setText("停止循环")
        self._start_pick_action()

    def _on_pick_feedback(self, step, status):
        self._log(f"抓取 [{step}/11] {status}", "info")

    def _on_pick_done(self, success, message):
        status = "OK" if success else "FAIL"
        self._log(f"[{status}] {message}", "info" if success else "error")
        if self._loop_active and success and not self._shutting_down:
            self._log(
                f"第 {self._loop_cycle} 轮完成，{self.LOOP_INTERVAL_MS / 1000:.1f}s 后开始下一轮",
                "info",
            )
            self._set_debug_buttons_enabled(False)
            self.loop_pick_button.setEnabled(True)
            self.loop_pick_button.setText("停止循环")
            self._loop_timer.start(self.LOOP_INTERVAL_MS)
            return

        if self._loop_active and not success:
            self._log("循环抓取因本轮失败而停止", "error")
        self._reset_pick_ui()

    def _on_arm_change(self):
        joints = [s.value() for s in self.joint_sliders]
        self._send(f"发送关节: {joints}", self.ros.fk_abs, joints)

    def _on_reset(self):
        """复位仿真（回 home + 删除方块），稍后同步滑块到实际状态。"""
        self._send("复位", self.ros.reset_sim)
        QTimer.singleShot(300, self._on_refresh)

    def _on_refresh(self):
        """从 /joint_states 同步当前关节角和夹爪到滑块"""
        joints = self.ros.get_current_arm_deg()
        if joints is not None:
            for slider, j in zip(self.joint_sliders, joints):
                slider.blockSignals(True)
                slider.setValue(int(round(j)))
                slider.blockSignals(False)
            self._log(f"刷新关节: {[f'{j:.1f}°' for j in joints]}", "info")
        else:
            self._log("刷新失败: 未收到关节状态", "warn")

        grip_val = self.ros.gripper_position
        self.grip_slider.blockSignals(True)
        self.grip_slider.setValue(int(grip_val * 1000))
        self.grip_slider.blockSignals(False)
        self._log(f"刷新夹爪: {grip_val:.3f}", "info")

    def _on_gripper_change(self):
        pos = self.grip_slider.value() / 1000.0
        self._send(f"夹爪: {pos:.3f}", self.ros.gripper, pos)


def main():
    rclpy.init()
    ros = RosClient()

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    win = MainWindow(ros)
    win.show()

    alive = True
    js_synced = False

    def spin():
        nonlocal alive, js_synced
        if not alive:
            return
        try:
            rclpy.spin_once(ros, timeout_sec=0.01)
        except Exception:
            pass
        if not js_synced and ros.js_received:
            js_synced = True
            win._on_refresh()

    timer = QTimer()
    timer.timeout.connect(spin)
    timer.start(50)

    code = app.exec()
    alive = False
    timer.stop()

    try:
        ros.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass
    sys.exit(code)


if __name__ == "__main__":
    main()
