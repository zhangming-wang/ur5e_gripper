# ur5e_gripper

UR5e 机械臂 + Robotiq 2F-85 自适应夹爪的 ROS 2 Humble 仿真项目，集成 Isaac Lab 物理仿真。

## 项目结构

```
ur5e_gripper/
├── src/                              # ROS 2 包
│   ├── ur5e_gripper_description/     # 机器人描述（URDF/XACRO + 网格）
│   ├── ur5e_gripper_moveit_config/   # MoveIt 2 配置与启动
│   ├── ur5e_gripper_msgs/            # 自定义服务定义
│   └── pymoveit2/                    # Python MoveIt 2 接口
├── isaaclab/src/                     # Isaac Lab 控制 / 规划 / 桥接节点
│   ├── bridge_node.py                # MoveIt ↔ IsaacLab 桥接
│   ├── planning_node.py              # 逆解/正解/夹爪规划服务
│   ├── control_node.py               # IsaacLab 物理执行
│   └── run_isaaclab.py               # IsaacLab 仿真入口
├── script/
│   └── ur5e_gripper_panel.py         # PySide6 调试面板
├── run.sh                            # 一键启动 (MoveIt + Bridge + Planning + IsaacLab)
├── install/                          # colcon 编译产物
└── README.md
```

## 依赖

```bash
# ROS 2 Humble
sudo apt install ros-humble-ros-base ros-humble-moveit ros-humble-ros2-control ros-humble-ros2-controllers

# PySide6 (GUI 测试面板)
pip3 install PySide6
```

## 编译

```bash
cd ~/work/ur5e_gripper
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

## 一键启动

```bash
./run.sh
```

启动 4 个进程：MoveIt → Bridge → Planning → IsaacLab。按 `Ctrl+C` 全部停止。

---

## 测试命令

先确保 `run.sh` 已启动且 `planning_node` 显示 `Planning node ready`，然后新开终端：

```bash
source /opt/ros/humble/setup.bash
source ~/work/ur5e_gripper/install/setup.bash
```

### 机械臂 — IK 笛卡尔空间

```bash
# 相对移动: 沿 X 轴 +10cm（xyz 单位米, rpy 单位度）
ros2 service call /plan_execute ur5e_gripper_msgs/srv/PlanExecute \
  "{command_type: ik_rel, data: '0.1 0 0 0 0 0'}"
```

```bash
# 绝对位姿
ros2 service call /plan_execute ur5e_gripper_msgs/srv/PlanExecute \
  "{command_type: ik_abs, data: '0.3 -0.2 0.4 180 0 90'}"
```

### 机械臂 — FK 关节空间

```bash
# 绝对关节角（单位度）
ros2 service call /plan_execute ur5e_gripper_msgs/srv/PlanExecute \
  "{command_type: fk_abs, data: '0 -90 90 0 90 0'}"
```

```bash
# 相对关节增量
ros2 service call /plan_execute ur5e_gripper_msgs/srv/PlanExecute \
  "{command_type: fk_rel, data: '0 -10 10 0 0 0'}"
```

### 夹爪

```bash
# 全关  |  半开  |  全开
ros2 service call /plan_execute ur5e_gripper_msgs/srv/PlanExecute \
  "{command_type: gripper, data: '0.0'}"
```

```bash
ros2 service call /plan_execute ur5e_gripper_msgs/srv/PlanExecute \
  "{command_type: gripper, data: '0.4'}"
```

```bash
ros2 service call /plan_execute ur5e_gripper_msgs/srv/PlanExecute \
  "{command_type: gripper, data: '0.8'}"
```

> `data` 范围：0.0（闭合）～ 0.8（全开）

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
python3 ~/work/ur5e_gripper/script/ur5e_gripper_panel.py
```

---

## 架构

```
GUI / ros2 service call
        │
        ▼
  planning_node  ───────── action ──────────▶  bridge_node
  (MoveIt 规划)     FollowJointTrajectory      (协议转换)
                        GripperCommand              │
                        ┌───────────────────────────┤
                        │ topic                     │ topic
                        ▼                           ▼
                   control_node  ◀──────────  IsaacLab 仿真
                   (物理执行)    done signal
```

## 许可

BSD-3-Clause
