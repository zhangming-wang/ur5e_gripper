"""
Launch Isaac Lab with the UR5e + Robotiq 2F-85 USD stage.
IsaacLab hosts FollowJointTrajectory action server — MoveIt plans, IsaacLab executes.
"""

import argparse
from isaaclab.app import AppLauncher
import set_isaaclab_env

# ----------------------------------------------------------------------
# 1. Args
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="UR5e + Robotiq 2F-85 in Isaac Lab")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ----------------------------------------------------------------------
# 2. Launch App — MUST come before other imports
# ----------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------
# 3. Imports after App
# ----------------------------------------------------------------------
import torch
import rclpy
from rclpy.executors import SingleThreadedExecutor

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from pathlib import Path

from control_node import ControlNode

# ----------------------------------------------------------------------
# 4. Config
# ----------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
_robot_usd_path = project_root / "urdf/ur5e_with_gripper/ur5e_with_gripper.usd"

if not _robot_usd_path.exists():
    raise FileNotFoundError(f"Robot USD not found: {_robot_usd_path}")


@configclass
class SceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    ur5e = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ur5e",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_robot_usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "ur5e_arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
                effort_limit_sim={
                    "shoulder_pan_joint": 150.0,
                    "shoulder_lift_joint": 150.0,
                    "elbow_joint": 150.0,
                    "wrist_1_joint": 28.0,
                    "wrist_2_joint": 28.0,
                    "wrist_3_joint": 28.0,
                },
                velocity_limit_sim={
                    "shoulder_pan_joint": 3.14,
                    "shoulder_lift_joint": 3.14,
                    "elbow_joint": 3.14,
                    "wrist_1_joint": 3.14,
                    "wrist_2_joint": 3.14,
                    "wrist_3_joint": 3.14,
                },
                stiffness=10000.0,
                damping=400.0,
            ),
            "ur5e_gripper": ImplicitActuatorCfg(
                joint_names_expr=[
                    "robotiq_85_left_knuckle_joint",
                    "robotiq_85_right_knuckle_joint",
                    "robotiq_85_left_inner_knuckle_joint",
                    "robotiq_85_right_inner_knuckle_joint",
                    "robotiq_85_left_finger_tip_joint",
                    "robotiq_85_right_finger_tip_joint",
                ],
                effort_limit_sim=50.0,
                velocity_limit_sim=0.5,
                stiffness=200.0,
                damping=10.0,
            ),
        },
    )


# ----------------------------------------------------------------------
# 5. MainLoop
# ----------------------------------------------------------------------
class MainLoop:
    def __init__(self):
        self.control_node = None
        try:
            self.init()
            self.exec()
        except Exception as e:
            print(f"[INFO] Simulation interrupted: {e}")
        finally:
            self.shutdown()
            simulation_app.close()
            print("-------------------exit-------------------")

    def init(self):
        rclpy.init(args=None)

        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view((2.0, 2.0, 1.5), (0.0, 0.0, 0.5))
        self.sim_dt = self.sim.get_physics_dt()

        scene_cfg = SceneCfg(args_cli.num_envs, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.ur5e = self.scene["ur5e"]
        joint_names = list(self.ur5e.data.joint_names)
        num_joints = self.ur5e.num_joints

        print(f"\n===== UR5e + Robotiq 2F-85 关节信息 =====")
        for i, name in enumerate(joint_names):
            print(f"  [{i:2d}] {name}")
        print(f"总关节数: {num_joints}\n")

        self.ros_executor = SingleThreadedExecutor()
        self.control_node = ControlNode(self, joint_names)
        self.ros_executor.add_node(self.control_node)

        self._init_robot()

    def exec(self):
        print("[INFO] Simulation running. Press Ctrl+C to stop.")

        while simulation_app.is_running():
            # 1. ROS2 事件处理（接收桥接节点发的轨迹/夹爪指令）
            self.ros_executor.spin_once(timeout_sec=0.0)

            # 2. 轨迹插值 + 夹爪
            self.control_node.step_traj()
            self.control_node.step_gripper()

            # 3. 读取物理状态，发布 /joint_states
            self.control_node.current_pos = self.ur5e.data.joint_pos[0].cpu().numpy()
            self.control_node.publish_joint_state()

            # 4. 应用目标位置到仿真
            target = torch.from_numpy(self.control_node.target_pos).to(self.sim.device).unsqueeze(0)
            self.ur5e.set_joint_position_target(target)

            # 5. 物理步进
            self.scene.write_data_to_sim()
            self.scene.update(self.sim_dt)
            self.sim.step()

    def shutdown(self):
        if self.control_node is not None:
            self.control_node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass  # SIGTERM 时 rclpy 可能已经 shutdown 了

    def _init_robot(self):
        """传送到初始位姿并设目标"""
        p = self.control_node.init_pos.copy()
        self.ur5e.write_joint_state_to_sim(
            torch.from_numpy(p).to(self.sim.device).unsqueeze(0),
            torch.zeros(1, self.ur5e.num_joints, device=self.sim.device))
        self.ur5e.set_joint_position_target(
            torch.from_numpy(p).to(self.sim.device).unsqueeze(0))
        self.control_node.current_pos = p.copy()
        self.control_node.target_pos = p.copy()

    def reset_env(self):
        self._init_robot()
        print("[INFO]: 仿真环境已重置")


if __name__ == "__main__":
    MainLoop()
