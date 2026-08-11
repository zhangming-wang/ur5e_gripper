#!/usr/bin/env python3
"""MuJoCo 模式 — ros2_control 启动文件。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue, ParameterFile
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    headless = LaunchConfiguration("headless").perform(context)
    pkgs_mujoco = FindPackageShare("mujoco")

    urdf_xacro = PathJoinSubstitution([pkgs_mujoco, "urdf", "ur5e_mujoco.urdf.xacro"])
    mujoco_scene = PathJoinSubstitution([pkgs_mujoco, "xml", "combined", "scene.xml"])
    controllers_file = PathJoinSubstitution([pkgs_mujoco, "config", "mujoco_controllers.yaml"])

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            urdf_xacro,
            " ",
            "mujoco_model_path:=",
            mujoco_scene,
            " ",
            "headless:=",
            headless,
            " ",
            "name:=ur",
        ]
    )
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[robot_description, {"use_sim_time": True}],
        ),
        Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            emulate_tty=True,
            output="both",
            parameters=[
                {"use_sim_time": True},
                ParameterFile(controllers_file),
            ],
            remappings=[("~/robot_description", "/robot_description")],
            on_exit=Shutdown(),
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", "--param-file", controllers_file],
            output="both",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_trajectory_controller", "--param-file", controllers_file],
            output="both",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["robotiq_gripper_controller", "--param-file", controllers_file],
            output="both",
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="false"),
            OpaqueFunction(function=launch_setup),
        ]
    )
