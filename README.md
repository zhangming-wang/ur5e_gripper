# ur5e_gripper

UR5e 机械臂 + Robotiq 2F-85 自适应夹爪的 ROS 2 Humble 仿真项目，集成 Isaac Lab 物理仿真。
提供两种可互换的控制模式：**MoveIt 规划**（`src/moveit/`）与 **ACT 端到端策略**（`src/act/`），
两者暴露同一个 `/pick_and_place` 接口。

## 项目结构

```
ur5e_gripper/
├── src/
│   ├── moveit/                       # MoveIt 规划控制模式
│   │   ├── description/              # URDF/XACRO + 网格
│   │   ├── moveit_config/            # MoveIt 2 配置与启动
│   │   ├── planning/                 # planning_node（/plan_execute）
│   │   ├── orchestrator/             # 11 步流水线（/pick_and_place）
│   │   ├── perception/               # 红色物体检测（/detect_object）
│   │   ├── pymoveit2/                # Python MoveIt 2 接口
│   │   └── trac_ik/                  # TRAC-IK 运动学插件
│   ├── act/                          # ACT 端到端控制模式（见 src/act/README.md）
│   ├── custom_msgs/                  # 服务/动作定义（两种模式共用）
│   └── recorder/                     # LeRobot 原始数据采集（不随 run.sh 启动）
├── isaaclab/src/                     # Isaac Lab 仿真入口 / 控制 / 桥接
├── mujoco/                           # MuJoCo ros2_control 后端
├── script/                           # panel.py、act_policy_server.py
├── tools/lerobot_convert.py          # raw_data → LeRobotDataset
├── test/                             # ACT 离线评估、loss 曲线
├── run.sh                            # 一键启动
└── install/                          # colcon 编译产物
```

## 依赖

```bash
# ROS 2 Humble
sudo apt install ros-humble-ros-base ros-humble-moveit ros-humble-ros2-control ros-humble-ros2-controllers

# PySide6 (GUI 调试面板)
pip3 install PySide6

# MuJoCo ros2_control 后端
sudo apt install ros-humble-mujoco-ros2-control \
  ros-humble-mujoco-vendor ros-humble-mujoco-ros2-control-plugins
```

项目跨三个 Python 运行时，`run.sh` 需要它们都存在（可用同名环境变量覆盖）：

| 用途 | conda 环境 | Python | 说明 |
|---|---|---|---|
| Isaac Lab 仿真 | `ISAACLAB_ENV`（默认 `isaaclab`） | 3.11 | 自带 ROS 2 库，不要与系统 ROS 混用 |
| ACT 推理 / 训练 | `LEROBOT_ENV`（默认 `lerobot`） | ≥3.12 | LeRobot 要求 ≥3.12，无法与系统 ROS 同进程 |
| 其余 ROS 节点 / 面板 | 系统环境 | 3.10 | `/opt/ros/humble` |

`CONDA_ROOT` 默认 `/home/dev/miniconda3`。

## 编译

```bash
cd ~/work/ur5e_gripper
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

## 一键启动

```bash
./run.sh                 # 默认 Isaac Sim（MoveIt 模式）
./run.sh --mujoco        # MuJoCo 后端（MoveIt 模式）
./run.sh --act           # ACT 端到端模式
```

`run.sh` 会导出 `ROS_DOMAIN_ID`（默认 46，可用同名环境变量覆盖）。

| 模式 | 进程数 | 启动顺序 |
|---|---|---|
| `--isaacsim`（默认） | 7 | MoveIt → Bridge → Planning → Perception → IsaacLab → Orchestrator → Panel |
| `--mujoco` | 6 | MoveIt → MuJoCo → Perception → Planning → Orchestrator → Panel |
| `--act` | 5 | ACT 推理服务 → IsaacLab `--act` → act_orchestrator → act_ros_adapter → Panel |

按 `Ctrl+C` 全部停止。

## 两种控制模式

两种模式都实现同一个 `/pick_and_place`（`custom_msgs/action/PickAndPlace`）action，
所以面板的单次/循环抓取逻辑通用。**一个 goal = 一轮**；循环模式由面板在成功后重发 goal，
失败则停止循环。

| | MoveIt 模式（`--isaacsim` / `--mujoco`） | ACT 模式（`--act`） |
|---|---|---|
| 决策 | 感知检测 → 几何位姿 → MoveIt 规划 | 端到端策略直接输出关节角 |
| 频率 | 低频（一次规划一整段轨迹） | 15Hz 连续闭环 |
| 完成判定 | 11 步流水线跑完 | 关节先离开、再回到 home |
| 需要的感知 | RGB + 深度 + HSV 检测 | 只要 RGB + 关节状态 |

ACT 模式的完整说明（ACT 原理、数据流水线、训练与评估、换模型要改什么）见
[`src/act/README.md`](src/act/README.md)。

---

## 测试命令（MoveIt 模式）

`/plan_execute` 只在 MoveIt 模式（`--isaacsim` / `--mujoco`）下可用；ACT 模式由策略直接控制。

先确保 `run.sh` 已启动且 `planning_node` 显示 `Planning node ready`，然后新开终端：

```bash
source /opt/ros/humble/setup.bash
source ~/work/ur5e_gripper/install/setup.bash
export ROS_DOMAIN_ID=46
```

### 机械臂 — IK 笛卡尔空间

```bash
# 相对移动: 沿 X 轴 +10cm（xyz 单位米, rpy 单位度）
ros2 service call /plan_execute custom_msgs/srv/PlanExecute \
  "{command_type: ik_rel, data: '0.1 0 0 0 0 0'}"
```

```bash
# 绝对位姿
ros2 service call /plan_execute custom_msgs/srv/PlanExecute \
  "{command_type: ik_abs, data: '0.3 -0.2 0.4 180 0 90'}"
```

### 机械臂 — FK 关节空间

```bash
# 绝对关节角（单位度）
ros2 service call /plan_execute custom_msgs/srv/PlanExecute \
  "{command_type: fk_abs, data: '0 -90 90 0 90 0'}"
```

```bash
# 相对关节增量
ros2 service call /plan_execute custom_msgs/srv/PlanExecute \
  "{command_type: fk_rel, data: '0 -10 10 0 0 0'}"
```

### 夹爪

```bash
# 全开  |  半开  |  全关
ros2 service call /plan_execute custom_msgs/srv/PlanExecute \
  "{command_type: gripper, data: '0.0'}"
```

```bash
ros2 service call /plan_execute custom_msgs/srv/PlanExecute \
  "{command_type: gripper, data: '0.4'}"
```

```bash
ros2 service call /plan_execute custom_msgs/srv/PlanExecute \
  "{command_type: gripper, data: '0.8'}"
```

> `data` 范围：0.0（全开）～ 0.8（闭合）

---

## 命令速查

| command_type | data 格式                      | 单位     |
| ------------ | ------------------------------ | -------- |
| `ik_rel`   | `dx dy dz droll dpitch dyaw` | 米 / 度  |
| `ik_abs`   | `x y z roll pitch yaw`       | 米 / 度  |
| `fk_rel`   | `dj1 dj2 dj3 dj4 dj5 dj6`    | 度       |
| `fk_abs`   | `j1 j2 j3 j4 j5 j6`          | 度       |
| `gripper`  | `position`                   | 0.0～0.8 |

---

## GUI 测试面板

```bash
# 必须用系统 Python 3.10，不能用 conda
conda deactivate
source /opt/ros/humble/setup.bash
source ~/work/ur5e_gripper/install/setup.bash
export ROS_DOMAIN_ID=46
python3 ~/work/ur5e_gripper/script/panel.py
```

“单次抓取”发送一次 `/pick_and_place`；“循环抓取”会在每轮成功后自动启动下一轮，运行中点击“停止循环”可取消当前任务。“复位”调用 `/reset` 清除方块并把机械臂送回 home（两种模式通用）。

---

## 数据采集与 ACT 训练

ACT 策略的数据链路（在 conda `lerobot` 环境执行，不是系统 ROS）：

```bash
# 1. 采集：单独启动 recorder（不随 run.sh 启动），在 MoveIt 模式下跑循环抓取即可累积 episode
ros2 run recorder recorder_node

# 2. 转换：raw_data -> LeRobotDataset（裁剪开头静止段、标签改为绝对下一状态）
python tools/lerobot_convert.py

# 3. 训练
lerobot-train --dataset.repo_id=ur5e_pick_place \
  --dataset.root=/home/dev/work/ur5e_gripper/lerobot_data_rgb_abs \
  --dataset.eval_split=0.2 --policy.type=act --policy.use_vae=false \
  --policy.chunk_size=50 --policy.n_action_steps=10 --policy.device=cuda \
  --policy.push_to_hub=false --batch_size=32 --steps=30000 --env_eval_freq=0 \
  --wandb.enable=false \
  --output_dir=/home/dev/work/ur5e_gripper/outputs/ur5e_act_rgb_abs_novae

# 4. 离线评估（对比"保持当前位置"基线）
python test/act_smoke_test.py \
  --checkpoint outputs/ur5e_act_rgb_abs_novae/checkpoints/030000/pretrained_model \
  --dataset-root lerobot_data_rgb_abs
```

详细原理与踩坑记录见 [`src/act/README.md`](src/act/README.md)。

---

## 架构

### MoveIt 模式（`--isaacsim` / `--mujoco`）

```
GUI / ros2 service call
        │
        ▼
   planning_node  ───────── action ──────────▶  bridge_node (IsaacLab)
   (MoveIt 规划)                              │ topics
        │                                     ▼
        └──── action ─────▶ MuJoCo / IsaacLab control
                                             │
                                             ▼
                                           仿真
```

### ACT 模式（`--act`）

```
Panel ──/pick_and_place──▶ act_orchestrator
                             │  1. /reset + /spawn_cube          → control_node
                             │  2. /act_ros_adapter/prepare      → adapter（预热 chunk）
                             │  3. /isaaclab/act/enabled = True  → adapter + control_node
                             │  4. 等关节回 home 后 enable=False
                             ▼
                        act_ros_adapter ──15Hz /isaaclab/act/joint_target──▶ control_node
                             │                                                   │
                             └──HTTP /infer_chunk──▶ script/act_policy_server.py  ▼
                                                      (conda lerobot, ACT 策略)  仿真
```

## 许可

项目代码和随附的上游模型分别遵循根目录及各自目录中的许可证文件。
