"""剔除系统 ROS2，使用 Isaac Sim 自带的 ROS2 bridge。"""

import os
import sys
from pathlib import Path

bridge = Path(
    sys.prefix,
    "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
    "isaacsim",
    "exts",
    "isaacsim.ros2.bridge",
)
distro = os.environ.get("ISAACLAB_ROS_DISTRO", "humble")
if not (bridge / distro).is_dir():
    raise RuntimeError(
        f"Isaac Sim ROS 2 bridge for '{distro}' was not found under {bridge}; "
        "set ISAACLAB_ROS_DISTRO only when the matching bridge is installed"
    )
root = bridge / distro
rclpy_dir = str(root / "rclpy")
lib_dir = str(root / "lib")

# 从环境变量和 sys.path 中剔除 /opt/ros/，注入 isaacsim 内部路径
to_inject = {"LD_LIBRARY_PATH": lib_dir, "PYTHONPATH": rclpy_dir, "AMENT_PREFIX_PATH": str(root)}
for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "CMAKE_PREFIX_PATH", "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH"):
    val = ":".join(p for p in os.environ.get(key, "").split(":") if p and "/opt/ros/" not in p)
    if key in to_inject:
        val = f"{to_inject[key]}:{val}" if val else to_inject[key]
    os.environ[key] = val

os.environ["ROS_DISTRO"] = distro

sys.path = [p for p in sys.path if "/opt/ros/" not in p]
if rclpy_dir not in sys.path:
    sys.path.insert(0, rclpy_dir)
