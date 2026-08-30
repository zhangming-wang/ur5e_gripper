# AGENTS.md — ur5e_gripper

UR5e + Robotiq 2F-85 simulation (ROS 2 Humble + Isaac Lab + MuJoCo), with an openpi VLA subproject.

## Runtime split

- **`src/`** — colcon ROS workspace using `/opt/ros/humble` (system Python 3.10).
- **`isaaclab/`** — standalone Isaac Lab app (not a colcon package), runs in conda env `isaaclab` (Python 3.11) with its own bundled ROS 2 libraries.
- **`mujoco/`** — standalone colcon package at repo root (discovered alongside `src/`). Provides the ros2_control backend: MJCF scene, URDF, launch, and controller config.
- **`third_party/openpi/`** — openpi VLA repo (Physical Intelligence), managed by `uv` (Python 3.11, no ROS). Not a colcon package.
- All ROS processes share `ROS_DOMAIN_ID=46`. `run.sh` exports it; direct commands must export it too.
- `isaaclab/src/set_isaaclab_env.py` strips `/opt/ros` from `PYTHONPATH`/`LD_LIBRARY_PATH` and injects Isaac Sim's ROS 2. **Keep this import before `rclpy`** — mixing ROS installations breaks imports and library loading.
- `run.sh` defaults `CONDA_ROOT` to `/home/dev/miniconda3` and `ISAACLAB_ENV` to `isaaclab`; override these variables when conda is installed elsewhere.

## Build, test, run

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

- Active colcon packages are discovered under `src/` and `mujoco/`; `build/`, `install/`, `log/` are generated and ignored.
- `COLCON_IGNORE` markers: `third_party/`, `src/trac_ik/trac_ik_examples/`, `src/trac_ik/trac_ik_python/`.
- No colcon tests are registered anywhere in the repo (`colcon test` has nothing to run).
- `./run.sh` defaults to **MuJoCo** (6 processes: MoveIt → MuJoCo → Planning → Perception → Orchestrator → `script/panel.py`). `./run.sh --isaacsim` uses Isaac Lab (7 processes: MoveIt → Bridge → Planning → Perception → IsaacLab → Orchestrator → `script/panel.py`). Ctrl+C stops all.
- `script/panel.py` requires PySide6 and must run under system ROS Python (not `isaaclab` conda env).

## Ownership & flow

| Directory | Package | Role |
|---|---|---|
| `src/description/` | `description` | URDF/XACRO, meshes |
| `src/moveit_config/` | `moveit_config` | MoveIt 2 launch + config (TRAC-IK kinematics) |
| `src/custom_msgs/` | `custom_msgs` | Service + action definitions |
| `src/planning/` | `planning` | `planning_node` (IK/FK/gripper) |
| `src/orchestrator/` | `orchestrator` | pick-and-place pipeline via `/pick_and_place` action |
| `src/perception/` | `perception` | Red-object HSV+Depth detection, Gemini2 intrinsics |
| `src/pymoveit2/` | `pymoveit2` | Local fork of MoveIt2 Python client (v4.2.0, not a pip package) |
| `src/trac_ik/` | `trac_ik_lib`, `trac_ik_kinematics_plugin`, `trac_ik` | TRAC-IK IK solver (replaces KDL) |
| `isaaclab/src/` | — | `run_isaaclab.py` (sim entrypoint), `control_node.py` (ROS node in sim), `bridge_node.py` (action server) |
| `mujoco/` | `mujoco` | MJCF scene, URDF, launch, and controller config; builds C++ `mujoco_scene_plugin` (serves `/spawn_cube` and `/reset`). `run_mujoco.py` — standalone MuJoCo viewer (conda env `mujoco`). |
| `third_party/` | — | Git submodules + `openpi/` (see below) |

## Live ROS contracts

- **`/plan_execute`** — `custom_msgs/srv/PlanExecute`. Valid types: `ik_abs`, `ik_rel`, `fk_abs`, `fk_rel`, `gripper`. Positions are meters, **angles/RPY are degrees**, all arm commands take 6 space-separated values.
- **`/detect_object`** — `custom_msgs/srv/DetectObject`. Returns x/y/z (meters) and **yaw in radians** — convert with `math.degrees()` before passing to `/plan_execute`.
- **`/pick_and_place`** — `custom_msgs/action/PickAndPlace`. One goal runs the pipeline **in an infinite loop** until cancelled (`ros2 action cancel` or Ctrl+C); each cycle starts by calling `/reset` to clear the old cube. Orchestrator helpers: `_plan_ik_abs/_plan_ik_rel/_plan_fk_abs/_plan_fk_rel/_plan_gripper/_go_home`.
- **`/spawn_cube`** and **`/reset`** — use `std_srvs/srv/Trigger` (not `custom_msgs/srv/SpawnCube`, which exists but is unused). Served by IsaacLab `control_node` in sim mode and by the MuJoCo `mujoco_scene_plugin` in MuJoCo mode.
- **Bridge ↔ Control topic pairs** (must stay in sync):
  - Arm: `/arm_controller/joint_trajectory` ↔ `/isaaclab/trajectory_done` (`Bool`)
  - Gripper: `/gripper_controller/command` (`Float64`) ↔ `/isaaclab/gripper_done` (`UInt8`)
  - Isaac execution uses **simulation time**; `bridge_node` uses wall time only for no-progress watchdogs. After a trajectory's time elapses, `control_node` waits for the arm to settle (3 consecutive frames within `arm_stable_threshold`, or `arm_settle_timeout` seconds — both ROS params, defaults `1e-4` / `5.0`) before publishing done.
- **Stop topic**: `/stop_motion` is the shared stop signal; `/isaaclab/stop_motion` remains a legacy IsaacLab-compatible topic.
- **Perception** subscribes to `/{env_prefix}/gemini2/rgb` and `/{env_prefix}/gemini2/depth` (`env_prefix` ROS param, default `env_0`). Gemini2 intrinsics (`FX/FY=686.3`, `CX=640`, `CY=360`), camera→base transform (`camera_to_base_rotation` is `diag(-1,1,-1)`, translation `[0, 0.5, 0.8]`), and ROI crop (`roi_top`/`roi_bottom`) are all ROS params — shared `src/perception/config/perception.yaml` for both modes; changing the scene layout breaks detection.
- `run.sh` is the source of truth for process ordering and backend selection; keep it synchronized with package manifests.

## Backend control paths

- **IsaacLab**: `planning_node` → `bridge_node` (action server, at `isaaclab/src/bridge_node.py`) → topics → `control_node` (sim).
- **`ik_rel`** in `planning_node` composes orientations via **quaternion multiplication** (`_quat_multiply(delta, cur)`), not RPY addition — adding Euler angles directly is wrong and skews Cartesian moves.
- **MuJoCo**: `planning_node` → `mujoco_ros2_control/ros2_control_node` (plugin). Same action endpoints, served directly by `controller_manager`. No bridge needed.
- MuJoCo mode depends on system ROS apt packages: `ros-humble-mujoco-ros2-control`, `ros-humble-mujoco-vendor`, `ros-humble-mujoco-ros2-control-plugins`.
- The MuJoCo MJCF must keep actuator names synced with the ros2_control joint list in `mujoco/urdf/`. The gripper uses a tendon-based position actuator named `robotiq_85_left_knuckle_joint` (matching the URDF/MoveIt joint), driven by `robotiq_gripper_controller` (`GripperActionController`).

## Known gotchas

- **README.md is outdated**: it uses service type `ur5e_gripper_msgs` — the real package is `custom_msgs`. Its panel script name is also stale (`script/panel.py`).
- **Sim tuning knobs** in `isaaclab/src/run_isaaclab.py`: arm `effort_limit_sim=300.0` Nm (all 6 joints), gripper `stiffness=500`/`damping=20`/`velocity_limit_sim=2.0`. `spawn_cube` places the cube at a random position (x∈[-0.1,0.1], y∈[0.45,0.55]) and random Z rotation (0-360°); `reset_env` deletes `/World/cube_0`.
- `moveit_config/config/joint_limits.yaml` caps `max_acceleration=5.0 rad/s²` for time parametrization — large joint rotations get short trajectories that the sim PD may not track; the settle watchdog absorbs the residual.

## openpi subproject (`third_party/openpi/`)

- Clone of Physical Intelligence's openpi (VLA models π₀ / π₀₋₅ / π₀-FAST). **Not a registered submodule** — currently an untracked clone on branch `kevin/pi05-support` (π₀₋₅ support lives there; `main` lacks it). π₀₋₇ exists but is not open-sourced in any branch.
- Managed by `uv` (`uv sync` inside the dir creates `.venv/`, Python 3.11). Fully isolated from ROS and the `isaaclab` conda env — never `source /opt/ros` here.
- Model weights are NOT in the repo; `serve_policy.py` downloads checkpoints from `gs://openpi-assets*` into `~/.cache/openpi` on first run.
- Robot-side integration uses the lightweight `packages/openpi-client` (WebSocket) — inference server runs where the GPU is; `examples/ur5/` has UR-specific config templates.
