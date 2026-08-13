"""
Launch Isaac Lab with the UR5e + Robotiq 2F-85 USD stage.
IsaacLab hosts FollowJointTrajectory action server — MoveIt plans, IsaacLab executes.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher
import set_isaaclab_env

# ----------------------------------------------------------------------
# 1. Args
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="UR5e + Robotiq 2F-85 in Isaac Lab")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # 必须启用，否则相机不渲染

# ----------------------------------------------------------------------
# 2. Launch App — MUST come before other imports
# ----------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------
# 3. Imports after App
# ----------------------------------------------------------------------
import math
import random
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.executors import SingleThreadedExecutor

import omni
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdPhysics
from isaacsim.core.utils import extensions  # type: ignore

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass

from control_node import ControlNode

extensions.enable_extension("isaacsim.ros2.bridge")

# ----------------------------------------------------------------------
# 4. Config
# ----------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
_robot_usd_path = project_root / "urdf/ur5e_with_gripper/ur5e_with_gripper.usd"
_desk_usd_path = project_root / "usd/desk/model_desk.usd"
_tray_usd_path = project_root / "usd/tray/model_redtray.usd"
_gemini2_usd_path = project_root / "usd/Gemini2/World0.usd"

for _p, _n in [
    (_robot_usd_path, "Robot"),
    (_desk_usd_path, "Desk"),
    (_tray_usd_path, "Tray"),
    (_gemini2_usd_path, "Gemini2"),
]:
    if not _p.exists():
        raise FileNotFoundError(f"{_n} USD not found: {_p}")


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

    # ---- 静态场景资产 ----
    desk = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/desk",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_desk_usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )

    left_tray = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/left_tray",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_tray_usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.5, 0.7),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    right_tray = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/right_tray",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_tray_usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -0.5, 0.7),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    gemini2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/gemini2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_gemini2_usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.5, 1.5),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
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
            pos=(0.0, 0.0, 0.7),
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
                    "shoulder_pan_joint": 300.0,
                    "shoulder_lift_joint": 300.0,
                    "elbow_joint": 300.0,
                    "wrist_1_joint": 300.0,
                    "wrist_2_joint": 300.0,
                    "wrist_3_joint": 300.0,
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
                velocity_limit_sim=2.0,
                stiffness=500.0,
                damping=20.0,
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
            if self.control_node is not None:
                self.control_node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            simulation_app.close()
            print("-------------------exit-------------------")

    def init(self):
        if not rclpy.ok():
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

        # 隐藏 Gemini2 的 camera_ldm
        stage = omni.usd.get_context().get_stage()
        for env_idx in range(args_cli.num_envs):
            cam_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/gemini2/Orbbec_Gemini2/camera_ldm")
            if cam_prim.IsValid():
                cam_prim.GetAttribute("visibility").Set("invisible")

        print(f"\n===== UR5e + Robotiq 2F-85 关节信息 =====")
        for i, name in enumerate(joint_names):
            print(f"  [{i:2d}] {name}")
        print(f"总关节数: {num_joints}\n")

        self.ros_executor = SingleThreadedExecutor()
        self.control_node = ControlNode(self, joint_names)
        self.ros_executor.add_node(self.control_node)

        self._init_robot()
        self._setup_gemini2_cameras()

    def exec(self):
        print("[INFO] Simulation running. Press Ctrl+C to stop.")

        while simulation_app.is_running():
            self.control_node.current_pos = self.ur5e.data.joint_pos[0].cpu().numpy()
            self.ros_executor.spin_once(timeout_sec=0.001)

            self.control_node.step_traj()
            self.control_node.step_gripper()

            self.control_node.publish_joint_state()

            target = torch.from_numpy(self.control_node.target_pos).to(self.sim.device).unsqueeze(0)
            self.ur5e.set_joint_position_target(target)

            self.scene.write_data_to_sim()
            self.scene.update(self.sim_dt)
            for cam in self.gemini2_cameras:
                cam.update(self.sim_dt)

            self.sim.step()

    def _init_robot(self):
        p = self.control_node.init_pos.copy()
        self.ur5e.write_joint_state_to_sim(
            torch.from_numpy(p).to(self.sim.device).unsqueeze(0),
            torch.zeros(1, self.ur5e.num_joints, device=self.sim.device),
        )
        self.ur5e.set_joint_position_target(torch.from_numpy(p).to(self.sim.device).unsqueeze(0))
        self.control_node.current_pos = p.copy()
        self.control_node.target_pos = p.copy()

    def reset_env(self):
        self._init_robot()
        # 删除旧方块
        stage = omni.usd.get_context().get_stage()
        cube_prim = stage.GetPrimAtPath("/World/cube_0")
        if cube_prim.IsValid():
            stage.RemovePrim("/World/cube_0")
        print("[INFO]: 仿真环境已重置")

    def spawn_cube(self):
        """在左托盘生成红色方块，返回对侧托盘的放置位姿"""

        stage = omni.usd.get_context().get_stage()
        cube_path = "/World/cube_0"
        cube_size = 0.03  # 3cm

        # 左托盘表面，矩形区域内随机位置
        # spawn_x, spawn_y, spawn_z = 0.0, 0.5, 0.8
        spawn_x = random.uniform(-0.1, 0.1)
        spawn_y = random.uniform(0.45, 0.55)
        spawn_z = 0.8

        # 随机绕 Z 轴旋转
        rand_angle = random.uniform(0, 360)
        rand_quat = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), rand_angle)

        # 创建方块
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.AddTranslateOp().Set(Gf.Vec3d(spawn_x, spawn_y, spawn_z))
        cube.AddOrientOp().Set(
            Gf.Quatf(
                rand_quat.GetQuaternion().GetReal(),
                *rand_quat.GetQuaternion().GetImaginary(),
            )
        )
        cube.AddScaleOp().Set(Gf.Vec3d(cube_size / 2, cube_size / 2, cube_size / 2))

        # 红色材质
        mat_path = f"{cube_path}/material"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)

        # 物理刚体
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        print(f"[INFO] Spawned red cube at ({spawn_x:.2f}, {spawn_y:.2f}, {spawn_z:.3f}) angle={rand_angle:.0f}°")

        return {
            "success": True,
            "message": "cube spawned",
        }

    # ==================================================================
    # Gemini2 相机 ROS2 发布
    # ==================================================================

    def _print_camera_info(self):
        """打印相机内参 + 手眼标定信息（numpy 手算，不依赖 Gf 矩阵运算）"""

        def _gf_to_pos_quat(mat):
            """Gf.Matrix4d → (pos_xyz, quat_xyzw) as numpy arrays"""
            t = mat.ExtractTranslation()
            q = mat.ExtractRotation().GetQuaternion()
            return (
                np.array([t[0], t[1], t[2]]),
                np.array([q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2], q.GetReal()]),
            )

        def _print_pose(pos, quat_xyzw, label):
            rpy = R.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
            print(
                f"[INFO] {label}: "
                f"t=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) "
                f"rpy=({rpy[0]:.2f},{rpy[1]:.2f},{rpy[2]:.2f})°"
            )

        if self.gemini2_cameras:
            intr = self.gemini2_cameras[0].data.intrinsic_matrices[0].cpu().numpy()
            res = self.gemini2_cameras[0].data.image_shape
            print(
                f"[INFO] Gemini2 intrinsics: fx={intr[0][0]:.1f} fy={intr[1][1]:.1f} "
                f"cx={intr[0][2]:.1f} cy={intr[1][2]:.1f} res={res[1]}×{res[0]}"
            )
        stage = omni.usd.get_context().get_stage()
        for env_path in sim_utils.find_matching_prim_paths("/World/envs/env_.*"):
            env = env_path.split("/")[-1]
            cam_mat = UsdGeom.Xformable(stage.GetPrimAtPath(f"{env_path}/gemini2")).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            base_mat = UsdGeom.Xformable(stage.GetPrimAtPath(f"{env_path}/ur5e")).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )

            cam_pos, cam_quat = _gf_to_pos_quat(cam_mat)
            base_pos, base_quat = _gf_to_pos_quat(base_mat)

            _print_pose(cam_pos, cam_quat, f"cam_world[{env}]")
            _print_pose(base_pos, base_quat, f"base_world[{env}]")

            # ---- numpy 手算 camera→base ----
            R_base = R.from_quat(base_quat)  # scipy 也用 xyzw
            R_cam = R.from_quat(cam_quat)

            # 平移: 世界下偏移 → 转到基座坐标系
            diff_world = cam_pos - base_pos
            t_c2b = R_base.inv().apply(diff_world)

            # 旋转: base_rot⁻¹ * cam_rot
            R_c2b = R_base.inv() * R_cam
            q_c2b = R_c2b.as_quat()  # xyzw

            _print_pose(t_c2b, q_c2b, f"camera→base[{env}]")

    def _setup_gemini2_cameras(self):
        """用 isaaclab.sensors.Camera + rep.writers 发布 Gemini2 的 4 路流"""
        num_envs = self.scene.num_envs

        # (Camera prim 名称,     数据类型,   topic,                   分辨率)
        streams = [
            ("camera_rgb/camera_rgb/Stream_rgb", "rgb", "gemini2/rgb", 1280, 720),
            (
                "camera_ir_left/camera_left/Stream_depth",
                "depth",
                "gemini2/depth",
                640,
                400,
            ),
            (
                "camera_ir_left/camera_left/Stream_ir_left",
                "rgb",
                "gemini2/ir_left",
                640,
                400,
            ),
            (
                "camera_ir_right/camera_right/Stream_ir_right",
                "rgb",
                "gemini2/ir_right",
                640,
                400,
            ),
        ]

        self.gemini2_cameras = []

        for env_i in range(num_envs):
            base = f"/World/envs/env_{env_i}/gemini2/Orbbec_Gemini2"

            for cam_rel, data_type, topic_base, w, h in streams:
                cam = Camera(
                    CameraCfg(
                        prim_path=f"{base}/{cam_rel}",
                        data_types=[data_type],
                        spawn=None,
                        width=w,
                        height=h,
                    )
                )
                cam._initialize_impl()
                self.gemini2_cameras.append(cam)

                # ROS2 publisher
                sensor_type = sd.SensorType.Rgb if data_type == "rgb" else sd.SensorType.DistanceToImagePlane
                rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sensor_type.name)
                writer = rep.writers.get(rv + "ROS2PublishImage")
                topic = f"/env_{env_i}/{topic_base}"
                writer.initialize(topicName=topic, frameId=f"gemini2_e{env_i}_{data_type}")
                writer.attach([cam._render_product_paths[0]])
                print(f"[INFO] {data_type:5s} → {topic}")

        print(f"[INFO] Gemini2 cameras: {num_envs} env × {len(streams)} streams")
        self._print_camera_info()


if __name__ == "__main__":
    MainLoop()
