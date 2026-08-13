from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_params = PathJoinSubstitution(
        [FindPackageShare("perception"), "config", "perception.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("env_prefix", default_value="env_0"),
            Node(
                package="perception",
                executable="perception_node",
                name="perception_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {"env_prefix": LaunchConfiguration("env_prefix")},
                ],
            ),
        ]
    )
