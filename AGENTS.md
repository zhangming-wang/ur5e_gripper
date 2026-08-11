# AGENTS.md — ur5e_gripper

UR5e + Robotiq 2F-85 simulation (ROS 2 Humble + Isaac Lab + MuJoCo).

## Runtime split

- **`src/`** — colcon ROS workspace using `/opt/ros/humble` (system Python 3.10).
- **`isaaclab/`** — standalone Isaac Lab app (not a colcon package), runs in conda env `isaaclab` (Python 3.11) with its own bundled ROS 2 libraries.
- **`mujoco/`** — standalone colcon package at repo root (discovered alongside `src/`). Provides the ros2_control backend: MJCF scene, URDF, launch, and controller config.
- All processes share `ROS_DOMAIN_ID=46`. `run.sh` exports it; direct commands must export it too.
- `isaaclab/src/set_isaaclab_env.py` strips `/opt/ros` from `PYTHONPATH`/`LD_LIBRARY_PATH` and injects Isaac Sim's ROS 2. **Keep this import before `rclpy`** — mixing ROS installations breaks imports and library loading.
- `run.sh` hardcodes conda init as `source /home/dev/miniconda3/etc/profile.d/conda.sh`. Update this path if conda is installed elsewhere.

## Build, test, run

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

- Active colcon packages are discovered under `src/` and `mujoco/`; `build/`, `install/`, `log/` are generated and ignored.
- `COLCON_IGNORE` markers: `third_party/`, `src/trac_ik/trac_ik_examples/`, `src/trac_ik/trac_ik_python/`.
- Only the `description` package has registered tests: `ur5e_urdf_xacro` and `view_ur5e_launch`.
- Run a single test with: `colcon test --packages-select description --event-handlers console_direct+ --ctest-args " -R ur5e_urdf_xacro"`.
- `./run.sh` starts 7 processes: MoveIt → Bridge → Planning → Perception → IsaacLab → Orchestrator → `script/panel.py`. Ctrl+C stops all.
- `./run.sh --isaacsim` starts the Isaac Lab backend. `./run.sh --mujoco` uses `mujoco_ros2_control` with 4 processes: MoveIt → MuJoCo → Planning → `script/panel.py`.
- `script/panel.py` requires PySide6 and must run under system ROS Python (not `isaaclab` conda env).

## Ownership & flow

| Directory | Package | Role |
|---|---|---|
| `src/description/` | `description` | URDF/XACRO, meshes, tests |
| `src/moveit_config/` | `moveit_config` | MoveIt 2 launch + config (TRAC-IK kinematics) |
| `src/custom_msgs/` | `custom_msgs` | Service + action definitions |
| `src/planning/` | `planning` | `planning_node` (IK/FK/gripper) |
| `src/orchestrator/` | `orchestrator` | 11-step pick-and-place pipeline via `/pick_and_place` action |
| `src/perception/` | `perception` | Red-object HSV+Depth detection, hardcoded Gemini2 intrinsics |
| `src/pymoveit2/` | `pymoveit2` | Local fork of MoveIt2 Python client (v4.2.0, not a pip package) |
| `src/trac_ik/` | `trac_ik_lib`, `trac_ik_kinematics_plugin`, `trac_ik` | TRAC-IK IK solver (replaces KDL) |
| `isaaclab/src/` | — | `run_isaaclab.py` (sim entrypoint), `control_node.py` (ROS node in sim) |
| `mujoco/` | `mujoco` | MJCF scene, URDF, launch, and controller config. `run_mujoco.py` — standalone MuJoCo viewer (conda env `mujoco`). |
| `third_party/` | — | Git submodules (not cloned) + untracked `mujoco_menagerie/` (source of MuJoCo assets) |

## Live ROS contracts

- **`/plan_execute`** — `custom_msgs/srv/PlanExecute`. Valid types: `ik_abs`, `ik_rel`, `fk_abs`, `fk_rel`, `gripper`. Positions are meters, angles/RPY are degrees, all arm commands take 6 space-separated values.
- **`/detect_object`** — `custom_msgs/srv/DetectObject`.
- **`/pick_and_place`** — `custom_msgs/action/PickAndPlace` (11-step pipeline, driven by `orchestrator_node`).
- **`/spawn_cube`** and **`/isaac_lab/reset`** — actually use `std_srvs/srv/Trigger` in the running implementation. `custom_msgs/srv/SpawnCube` exists but is **not** the live endpoint.
- **Bridge ↔ Control topic pairs** (must stay in sync):
  - Arm: `/arm_controller/joint_trajectory` ↔ `/isaaclab/trajectory_done`
  - Gripper: `/gripper_controller/command` ↔ `/isaaclab/gripper_done`
- **Perception** defaults to `env_0` and subscribes to `/{env_prefix}/gemini2/rgb` and `/{env_prefix}/gemini2/depth`. Gemini2 intrinsics (`FX/FY=686.3`, `CX=640`, `CY=360`) and camera→base transform are hardcoded in `perception_node.py` — changing the scene layout breaks detection.
- The root `README.md` uses stale package names (`ur5e_gripper_description` etc.), a wrong panel filename, and a wrong process count. **Trust `run.sh` and package manifests**, not the README.

## Backend control paths

- **IsaacLab**: `planning_node` → `bridge_node` (action server, at `isaaclab/src/bridge_node.py`) → topics → `control_node` (sim). `bridge_node` provides `/joint_trajectory_controller/follow_joint_trajectory` and `/robotiq_gripper_controller/gripper_cmd` actions.
- **MuJoCo**: `planning_node` → `mujoco_ros2_control/ros2_control_node` (plugin). Same action endpoints, served directly by `controller_manager`. No bridge needed.
- MuJoCo mode depends on system ROS apt packages: `ros-humble-mujoco-ros2-control`, `ros-humble-mujoco-vendor`, `ros-humble-mujoco-ros2-control-msgs`, `ros-humble-mujoco-ros2-control-plugins`.
- The MuJoCo MJCF must keep actuator names synced with the ros2_control joint list in `mujoco/urdf/`. The gripper uses a tendon-based position actuator named `robotiq_85_left_knuckle_joint` matching the URDF/MoveIt joint.
