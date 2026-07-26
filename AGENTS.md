# AGENTS.md — ur5e_gripper

UR5e arm + Robotiq 2F-85 gripper simulation with ROS 2 Humble + Isaac Sim.

## Two-env architecture (critical)

- **System ROS 2 (Python 3.10)**: all ROS packages under `src/`. Uses `/opt/ros/humble`.
- **Isaac Sim (Python 3.11, conda env `isaaclab`)**: simulation in `isaaclab/`. Has its own bundled ROS 2 rclpy path.
- Both communicate over ROS 2 middleware with `ROS_DOMAIN_ID=46`.
- `isaaclab/src/set_isaaclab_env.py` strips system ROS paths from `PYTHONPATH`/`LD_LIBRARY_PATH` and injects isaacsim's own ROS 2 paths before importing rclpy. This isolation is required — do not remove it.

## Build

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

Build output goes to `build/`, `install/`, `log/` — colcon standard. Repo has no root `CMakeLists.txt`.

`COLCON_IGNORE` markers exist in: `build/`, `install/`, `log/`, `third_party/`, `src/trac_ik/trac_ik_examples/`, `src/trac_ik/trac_ik_python/`.

Git submodules (`third_party/`) are registered but not cloned. URDF/xacro descriptions live self-contained in `src/description/` (adapted from UR robots repos).

## Run

```bash
./run.sh
```

Launches 7 processes: MoveIt → Bridge → Planning → Perception → IsaacLab → Orchestrator → GUI panel. Ctrl+C stops all. All processes share `ROS_DOMAIN_ID=46`.

## Test

Only the `description` package has tests (xacro→URDF validation + launch validation):

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
colcon test --packages-select description --event-handlers console_direct+
```

## Service type name (README is outdated)

The package is `src/custom_msgs/` with package name `custom_msgs`. The README incorrectly uses `ur5e_gripper_msgs` in example commands. Correct service types:

- `custom_msgs/srv/PlanExecute` — arm IK/FK/gripper
- `custom_msgs/srv/DetectObject` — perception
- `custom_msgs/srv/SpawnCube` — IsaacLab cube spawn

## Key packages

| Directory | Package name | Type | Role |
|---|---|---|---|
| `src/description/` | `description` | ament_cmake | URDF/xacro/meshes, has tests |
| `src/moveit_config/` | `moveit_config` | ament_cmake | MoveIt 2 launch + config, uses TRAC-IK kinematics |
| `src/custom_msgs/` | `custom_msgs` | ament_cmake | ROS 2 service definitions |
| `src/bridge/` | `bridge` | ament_python | `bridge_node` (MoveIt actions→IsaacLab topics) + `planning_node` (IK/FK/gripper service) |
| `src/orchestrator/` | `orchestrator` | ament_python | 10-step pick-and-place pipeline |
| `src/perception/` | `perception` | ament_python | Red-object detection via HSV + Depth, hardcoded Gemini2 camera params |
| `src/pymoveit2/` | `pymoveit2` | ament_cmake | Fork of MoveIt2 Python client, v4.2.0, not a pip package |
| `src/trac_ik/trac_ik_lib/` | `trac_ik_lib` | ament_cmake | TRAC-IK solver core |
| `src/trac_ik/trac_ik_kinematics_plugin/` | `trac_ik_kinematics_plugin` | ament_cmake | MoveIt IK plugin (replaces KDL) |
| `isaaclab/src/` | — | standalone | `run_isaaclab.py` (sim entrypoint), `control_node.py` (ROS node in sim) |

## Code style

- VS Code configured for `black` formatter (`--line-length 120`) and `flake8` linter
- No CI/CD (`.github/` is empty)
