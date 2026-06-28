# ur5e_gripper

UR5e 机械臂 + Robotiq 2F-85 自适应夹爪的 ROS 2 Humble 仿真项目，自包含、可直接编译运行。

## 项目结构

```
ur5e_gripper/
├── ur5e_gripper_description/       # 机器人描述（URDF/XACRO + 网格）
│   ├── config/                     # UR5e 运动学/关节限位/物理/视觉参数
│   ├── urdf/                       # XACRO 描述文件
│   │   ├── ur5e.urdf.xacro         # ★ 入口（use_gripper:=true/false 切换）
│   │   ├── ur_macro.xacro          # 臂宏定义
│   │   ├── ur.ros2_control.xacro   # 臂 ros2_control
│   │   ├── inc/                    # 公共 XACRO 片段
│   │   └── robotiq_2f_85_macro.urdf.xacro  # 夹爪宏定义
│   ├── meshes/ur5e/                # UR5e 网格
│   ├── meshes/robotiq/             # 2F-85 网格
│   └── launch/                     # 可视化启动
├── ur5e_gripper_moveit_config/     # MoveIt 2 配置与启动
│   ├── config/                     # 运动规划/控制器/SRDF 参数
│   ├── srdf/                       # 碰撞禁对（臂 + 夹爪）
│   └── launch/moveit.launch.py     # ★ 主启动文件
├── ur5e_gripper_isaaclab/          # Isaac Lab 仿真集成
│   ├── urdf/                       # 纯 URDF + mesh + 导入生成的 USD
│   └── src/run_isaaclab.py         # ★ Isaac Lab 启动脚本
├── .gitignore
├── LICENSE
└── README.md
```

## 依赖

```bash
# 基础 ROS 2 Humble
sudo apt install ros-humble-ros-base ros-humble-moveit

# 仿真控制器
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
```

## 编译

```bash
cd ~/work/Universal_Robots
colcon build --symlink-install
source install/setup.bash
```

## 使用

### 可视化（RViz 查看模型）

```bash
ros2 launch ur5e_gripper_description description.launch.xml
```

### 臂-only 运动规划仿真

```bash
ros2 launch ur5e_gripper_moveit_config moveit.launch.py \
  use_fake_hardware:=true \
  launch_servo:=false
```

### 臂 + 夹爪运动规划仿真

```bash
ros2 launch ur5e_gripper_moveit_config moveit.launch.py \
  use_fake_hardware:=true \
  use_gripper:=true \
  launch_servo:=false
```

### 控制夹爪

```bash
# 张开
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd \
  control_msgs/action/GripperCommand "{command: {position: 0.0, max_effort: 50.0}}"

# 闭合
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd \
  control_msgs/action/GripperCommand "{command: {position: 0.8, max_effort: 50.0}}"
```

> `position` 范围：0.0（张开）～ 0.8（闭合）

## 启动参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `use_fake_hardware` | `false` | 使用 mock_components 仿真硬件 |
| `use_gripper` | `false` | 挂载 Robotiq 2F-85 夹爪 |
| `launch_servo` | `true` | MoveIt 伺服遥操作（仿真可关） |
| `launch_rviz` | `true` | 启动 RViz |
| `robot_ip` | `yyy` | 机械臂 IP（仿真忽略） |
| `safety_limits` | `true` | 关节安全限位 |

## 与官方仓库的关系

本项目从 [Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) 和 [ros2_robotiq_gripper](https://github.com/PickNikRobotics/ros2_robotiq_gripper) 的 `humble` 分支提取，精简为 UR5e + 2F-85 单机器人配置。XACRO、控制器配置与官方保持一致，差异仅为：

- 包名改为 `ur5e_gripper_*`（自包含）
- 夹爪网格随项目分发（不依赖外部包）
- SRDF 增加了适配器和夹爪的碰撞禁对（官方无夹爪）

### Isaac Lab 仿真（UR5e + 夹爪物理仿真）

```bash
cd ~/work/Universal_Robots/ur5e_gripper/ur5e_gripper_isaaclab
./run_isaaclab.sh
```

> 需要 Isaac Lab 环境：`conda activate isaaclab`

### 生成纯 URDF（给 Isaac Sim 导入用）

```bash
source ~/work/Universal_Robots/install/setup.bash
xacro ~/work/Universal_Robots/install/ur5e_gripper_description/share/ur5e_gripper_description/urdf/ur5e.urdf.xacro \
  name:=ur use_gripper:=true safety_limits:=false \
  | python3 -c "
import sys, re
c = sys.stdin.read()
c = re.sub(r'<ros2_control[^>]*>.*?</ros2_control>', '', c, flags=re.DOTALL)
c = c.replace('package://ur5e_gripper_description/', '')
print(c)
" > urdf/ur5e_with_gripper.urdf
```

## 许可

BSD-3-Clause
