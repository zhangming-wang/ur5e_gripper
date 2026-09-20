# AGENTS.md — ur5e_gripper

UR5e + Robotiq 2F-85 simulation (ROS 2 Humble + Isaac Lab + MuJoCo) with three interchangeable
control modes: MoveIt planning (`src/moveit/`), ACT (`src/act/`), and Diffusion Policy (`src/diffusion/`).

## Runtime split

- **`src/`** — colcon ROS workspace using `/opt/ros/humble` (system Python 3.10).
- **`src/act/`** — ACT control mode: ROS nodes only (`act_orchestrator`, `act_ros_adapter`).
  Policy inference runs in conda env `lerobot` (Py3.12) via `script/act_policy_server.py`;
  LeRobot requires Python ≥3.12, so it can never run in the same process as system ROS.
- **`src/diffusion/`** — Diffusion control mode: ROS nodes only (`diffusion_orchestrator`,
  `diffusion_ros_adapter`). Policy inference runs in the same `lerobot` environment via
  `script/diffusion_policy_server.py`, but uses independent topics and port `27658`.
- **`isaaclab/`** — standalone Isaac Lab app (not a colcon package), runs in conda env `isaaclab` (Python 3.11) with its own bundled ROS 2 libraries.
- **`mujoco/`** — standalone colcon package at repo root (discovered alongside `src/`). Provides the ros2_control backend: MJCF scene, URDF, launch, and controller config.
- All ROS processes share `ROS_DOMAIN_ID`; `run.sh` exports it (default `46`).
- `isaaclab/src/set_isaaclab_env.py` strips `/opt/ros` from `PYTHONPATH`/`LD_LIBRARY_PATH` and injects Isaac Sim's ROS 2. **Keep this import before `rclpy`** — mixing ROS installations breaks imports and library loading.
- `run.sh` defaults `CONDA_ROOT` to `/home/dev/miniconda3`, `ISAACLAB_ENV` to `isaaclab`, `LEROBOT_ENV` to `lerobot`; override these variables when conda is installed elsewhere.

## Build, test, run

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

- Active colcon packages are discovered under `src/` and `mujoco/`; `build/`, `install/`, `log/` are generated and ignored.
- `COLCON_IGNORE` markers: `third_party/`, `src/moveit/trac_ik/trac_ik_examples/`, `src/moveit/trac_ik/trac_ik_python/`.
- No colcon tests are registered anywhere in the repo (`colcon test` has nothing to run).
- `./run.sh` defaults to **Isaac Sim** (7 processes: MoveIt → Bridge → Planning → Perception → IsaacLab → Orchestrator → `script/panel.py`). `./run.sh --mujoco` uses MuJoCo (6 processes). `./run.sh --act` runs the ACT mode (5 processes: policy server → IsaacLab `--act` → `act_orchestrator` → `act_ros_adapter` → `script/panel.py`). `./run.sh --diffusion` runs the independent Diffusion mode (5 processes: policy server → IsaacLab `--diffusion` → `diffusion_orchestrator` → `diffusion_ros_adapter` → `script/panel.py`). The two direct-policy modes skip MoveIt/Bridge/Planning/Perception/MoveIt Orchestrator. Ctrl+C stops all.
- ACT mode parameters: `ACT_CHECKPOINT` (default `outputs/ur5e_act_rgb_abs_novae/checkpoints/030000/pretrained_model`), `ACT_PORT` (default `27655`), `LEROBOT_ENV`.
- Diffusion mode parameters: `DIFFUSION_CHECKPOINT` (default `outputs/ur5e_diffusion_abs/checkpoints/030000/pretrained_model`), `DIFFUSION_PORT` (default `27658`), `DIFFUSION_INFERENCE_STEPS` (default `20`), `DIFFUSION_PREFETCH_THRESHOLD` (default `0`), `DIFFUSION_BLEND_STEPS` (default `4`), and `DIFFUSION_GRIPPER_CONFIRM_STEPS` (default `3`).
- `script/panel.py` requires PySide6 and must run under system ROS Python (not `isaaclab` conda env).

## Ownership & flow

| Directory | Package | Role |
|---|---|---|
| `src/custom_msgs/` | `custom_msgs` | Service + action definitions (shared by all modes) |
| `src/moveit/description/` | `description` | URDF/XACRO, meshes |
| `src/moveit/moveit_config/` | `moveit_config` | MoveIt 2 launch + config (TRAC-IK kinematics) |
| `src/moveit/planning/` | `planning` | `planning_node` (IK/FK/gripper via `/plan_execute`) |
| `src/moveit/orchestrator/` | `orchestrator` | MoveIt pick-and-place pipeline via `/pick_and_place` |
| `src/moveit/perception/` | `perception` | Red-object HSV+Depth detection, Gemini2 intrinsics |
| `src/moveit/pymoveit2/` | `pymoveit2` | Local fork of MoveIt2 Python client (v4.2.0, not a pip package) |
| `src/moveit/trac_ik/` | `trac_ik_lib`, `trac_ik_kinematics_plugin`, `trac_ik` | TRAC-IK IK solver (replaces KDL) |
| `src/act/` | `act` | ACT mode: `act_orchestrator` (`/pick_and_place`) + `act_ros_adapter` (policy executor) |
| `src/diffusion/` | `diffusion` | Diffusion mode: `diffusion_orchestrator` (`/pick_and_place`) + `diffusion_ros_adapter` (policy executor) |
| `src/recorder/` | `recorder` | LeRobot raw episode recorder (data collection only, not launched by `run.sh`) |
| `isaaclab/src/` | — | `run_isaaclab.py` (sim entrypoint), `control_node.py` (ROS node in sim), `bridge_node.py` (action server) |
| `mujoco/` | `mujoco` | MJCF scene, URDF, launch, and controller config; builds C++ `mujoco_scene_plugin` (serves `/spawn_cube` and `/reset`). `run_mujoco.py` — standalone MuJoCo viewer (conda env `mujoco`). |
| `script/` | — | `panel.py` (PySide6 debug/control GUI), ACT and Diffusion LeRobot-env HTTP inference servers |
| `tools/` | — | `lerobot_convert.py` (raw_data → LeRobotDataset) |
| `third_party/` | — | Git submodules (`mujoco_menagerie`, `ros2_robotiq_gripper`, `Universal_Robots_ROS2_*`) |

## Live ROS contracts

- **`/pick_and_place`** — `custom_msgs/action/PickAndPlace`. **Three interchangeable servers implement it** (never run together): `src/moveit/orchestrator` (MoveIt pipeline), `src/act/act_orchestrator` (one ACT cycle), and `src/diffusion/diffusion_orchestrator` (one Diffusion cycle). Each runs `/reset` → `/spawn_cube` → prepare → enable → wait for home. **One goal = one cycle**; `script/panel.py` loop mode resubmits a new goal 500 ms after success and stops the loop on failure.
- **`/plan_execute`** — `custom_msgs/srv/PlanExecute` (MoveIt mode only). Valid types: `ik_abs`, `ik_rel`, `fk_abs`, `fk_rel`, `gripper`. Positions are meters, **angles/RPY are degrees**, all arm commands take 6 space-separated values.
- **`/detect_object`** — `custom_msgs/srv/DetectObject` (MoveIt mode only). Returns x/y/z (meters) and **yaw in radians** — convert with `math.degrees()` before passing to `/plan_execute`.
- **`/spawn_cube`** and **`/reset`** — use `std_srvs/srv/Trigger` (not `custom_msgs/srv/SpawnCube`, which exists but is unused). Served by IsaacLab `control_node` in sim mode and by the MuJoCo `mujoco_scene_plugin` in MuJoCo mode.
- **Bridge ↔ Control topic pairs** (must stay in sync):
  - Arm: `/arm_controller/joint_trajectory` ↔ `/isaaclab/trajectory_done` (`Bool`)
  - Gripper: `/gripper_controller/command` (`Float64`) ↔ `/isaaclab/gripper_done` (`UInt8`)
  - Isaac execution uses **simulation time**; `bridge_node` uses wall time only for no-progress watchdogs. After a trajectory's time elapses, `control_node` waits for the arm to settle (3 consecutive frames within `arm_stable_threshold`, or `arm_settle_timeout` seconds — both ROS params, defaults `1e-4` / `5.0`) before publishing done.
- **Stop topic**: `/stop_motion` is the shared stop signal; `/isaaclab/stop_motion` remains a legacy IsaacLab-compatible topic.
- **Perception** subscribes to `/{env_prefix}/gemini2/rgb` and `/{env_prefix}/gemini2/depth` (`env_prefix` ROS param, default `env_0`). Gemini2 intrinsics (`FX/FY=686.3`, `CX=640`, `CY=360`), camera→base transform (`camera_to_base_rotation` is `diag(-1,1,-1)`, translation `[0, 0.5, 0.8]`), and ROI crop (`roi_top`/`roi_bottom`) are all ROS params — shared `src/moveit/perception/config/perception.yaml`; changing the scene layout breaks detection.
- `run.sh` is the source of truth for process ordering and backend selection; keep it synchronized with package manifests.

## Backend control paths

- **MoveIt / IsaacLab**: `planning_node` → `bridge_node` (action server, at `isaaclab/src/bridge_node.py`) → topics → `control_node` (sim).
- **`ik_rel`** in `planning_node` composes orientations via **quaternion multiplication** (`_quat_multiply(delta, cur)`), not RPY addition — adding Euler angles directly is wrong and skews Cartesian moves.
- **MoveIt / MuJoCo**: `planning_node` → `mujoco_ros2_control/ros2_control_node` (plugin). Same action endpoints, served directly by `controller_manager`. No bridge needed. Depends on system ROS apt packages: `ros-humble-mujoco-ros2-control`, `ros-humble-mujoco-vendor`, `ros-humble-mujoco-ros2-control-plugins`.
- The MuJoCo MJCF must keep actuator names synced with the ros2_control joint list in `mujoco/urdf/`. The gripper uses a tendon-based position actuator named `robotiq_85_left_knuckle_joint` (matching the URDF/MoveIt joint), driven by `robotiq_gripper_controller` (`GripperActionController`).
- **ACT**: `act_orchestrator` owns the cycle and publishes `/isaaclab/act/enabled` (`Bool`); `act_ros_adapter` subscribes to it, serves `/act_ros_adapter/prepare` (pre-warm one chunk), and publishes `/isaaclab/act/joint_target` (`JointTrajectory`, one 7-joint point). In `--act` mode `control_node` ignores `/arm_controller/joint_trajectory` and `/gripper_controller/command`, clamps per-step motion (`act_max_joint_step`, default `0.15` rad), interpolates over `act_command_period`, and holds position on an `act_watchdog_sec` (default `0.5`s) timeout. The adapter maps predicted gripper to `0.0`/`0.8` (the dataset stores measured knuckle position, ~0.63 at contact, not the command).
- ACT completion is inferred from joint state, not a fixed delay: the arm must first leave home (`>0.3` rad), then return within `0.2` rad and stay settled for `1.0`s; a `60`s `cycle_timeout` aborts the cycle. Home is `[0, -1.5708, 1.5708, 0, 1.5708, 0]` (equals `control_node.init_pos` and the training-data end pose).
- **Diffusion**: `diffusion_orchestrator` owns the same cycle semantics, but uses `/isaaclab/diffusion/enabled`, `/diffusion_ros_adapter/prepare`, and `/isaaclab/diffusion/joint_target`. The adapter sends the latest two synchronized RGB/state observations to the server, which returns 32 absolute actions. The adapter publishes at 15 Hz and, by default, requests the next chunk only after the current one drains so the observation is current; `control_node` uses separate `diffusion_*` watchdog, interpolation, and limit state.

## Known gotchas

- **Sim tuning knobs** in `isaaclab/src/run_isaaclab.py`: arm actuators use `stiffness=10000`/`damping=400`/`effort_limit_sim=300` Nm; gripper uses `stiffness=20`/`damping=1`/`effort_limit_sim=6`/`velocity_limit_sim=3`. `spawn_cube` places the cube at a random position (x∈[-0.1,0.1], y∈[0.45,0.55]) and random Z rotation (0-360°); `reset_env` deletes `/World/cube_0`.
- `moveit_config/config/joint_limits.yaml` caps `max_acceleration=5.0 rad/s²` for time parametrization — large joint rotations get short trajectories that the sim PD may not track; the settle watchdog absorbs the residual.
- `act_ros_adapter`'s `execute`-side helpers must not use ROS params for CLI values: `rclpy.init()` leaves non-ROS argv intact, so `ros2 run act act_ros_adapter --server-url ...` works, while `--ros-args -p server_url:=...` would be ignored.
- `act_orchestrator` uses `MultiThreadedExecutor(num_threads=2)` with `/joint_states` in a separate `MutuallyExclusiveCallbackGroup` — rclpy runs action execute callbacks on the executor, so a blocking callback would otherwise starve the subscription.
- **`step_act()` must never set `target_pos = current_pos`.** Setting the position target to the *measured* position zeroes the PD error, leaving gravity unopposed; repeating it every frame makes the target chase the sag, so the arm droops slowly. The idle and watchdog branches only `return` — they keep the last interpolated target (or `init_pos` from `_init_robot` before the first cycle). This differs from the one-shot `target_pos = current_pos` in `_stop_motion_callback`, which is harmless because nothing rewrites it afterwards.
