# act — ACT 端到端控制模式

与 `src/moveit`（MoveIt 规划模式）和 `src/diffusion`（Diffusion Policy 模式）并列的
端到端控制模式。三种模式都提供同名的 `/pick_and_place`
（`custom_msgs/action/PickAndPlace`），因此 `script/panel.py` 的单次/循环抓取逻辑通用。

---

## 一、ACT 是什么

ACT = **Action Chunking Transformer**（[论文](https://tonyzhaozh.github.io/aloha)）。
核心思想：不等一步做一步，而是**一次预测未来一整段动作序列**。

| 项 | 内容 |
|---|---|
| 输入 | 1 路 RGB `3×256×256` + 关节状态 `7` 维 |
| 输出 | `(chunk_size, 7)` —— 未来 `chunk_size` 步的**绝对关节角**（本项目训练用 `50`，即 15Hz 下 3.33 秒；配置默认值是 `100`） |
| 结构 | ResNet18 提图像特征 → Transformer encoder(4层)/decoder(1层) → 动作头 |
| 损失 | L1，预测 chunk vs 真实未来 chunk |
| 配置 | `use_vae=false`（见下） |

**为什么关掉 VAE**：原版 ACT 用 VAE 处理"同一观测对应多种合理动作"的多模态情况。
本任务的专家轨迹由脚本生成、高度一致，没有多模态，关掉更稳，也避免 VAE 训练/推理
时 latent 处理不一致导致的退化。

---

## 二、数据流水线

链路：`IsaacSim 录制(raw_data) → 清洗+重打标签(convert) → 训练 → 离线评估 → 闭环部署`

### 2.1 采集

`src/recorder` 在每次 pick-and-place 循环中采集 RGB + 关节状态，按 episode 落盘
（由 `/pick_and_place/_action/feedback` 的 step 2/12 切分，默认 15Hz）。

### 2.2 清洗

`tools/lerobot_convert.py` 对每个 episode：

| 处理 | 说明 |
|---|---|
| 裁剪开头静止段 | 找到首个运动帧（`>1e-4 rad`），只保留起振前 2 帧 |
| 重打动作标签 | `action[t] = state[t+1]`（**绝对**，末帧保持） |
| 丢弃深度 | 只保留 RGB |
| 完整性校验 | 帧数、时间戳节拍、CSV 列名、帧号连续性；不合格直接报错 |

实测：`27757 → 25220` 帧（裁掉 9.1%）。

### 2.3 为什么标签必须用绝对（关键）

原始版本用差分 `action[t] = state[t+1] - state[t]`，导致了**模型坍缩**：

- 轨迹里静止帧占比极高（各关节 52%~84%），差分标签**绝大多数是 0**
- ACT 用 L1 loss → "恒输出 0"是最省力的解
- 实测：恒输出零的归一化 L1 仅约 `0.43`，模型确实收敛到了这个水平

改成绝对后，目标变成真实关节位姿，**不能用常数蒙对**，模型被迫依赖观测去预测。

> **自检方法**：任何标签设计都先算一次"常数基线"。如果恒输出某个常数就能得到很低的
> loss，说明标签信号太弱，必须先修数据再训练。

### 2.4 转换命令

```bash
conda activate lerobot
python tools/lerobot_convert.py          # 输出 lerobot_data_rgb_abs/，已存在则会报错退出
```

---

## 三、训练与评估

### 3.1 训练

```bash
conda activate lerobot
lerobot-train \
  --dataset.repo_id=ur5e_pick_place \
  --dataset.root=/home/dev/work/ur5e_gripper/lerobot_data_rgb_abs \
  --dataset.video_backend=pyav \
  --dataset.eval_split=0.2 \
  --policy.type=act \
  --policy.use_vae=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=32 --steps=30000 --eval_steps=2000 --env_eval_freq=0 \
  --save_freq=5000 --log_freq=100 --wandb.enable=false \
  --output_dir=/home/dev/work/ur5e_gripper/outputs/ur5e_act_rgb_abs_novae
```

启动日志应显示 `41 train / 11 eval`、`25220` 帧。显存不足时把 `--batch_size` 降到 16。

### 3.2 离线评估

```bash
conda activate lerobot
python test/act_smoke_test.py \
  --checkpoint outputs/ur5e_act_rgb_abs_novae/checkpoints/030000/pretrained_model \
  --dataset-root lerobot_data_rgb_abs
```

判据（脚本会自动打印）：

- `Model/baseline ratio < 1` —— 优于"保持当前位置"的常数基线
- `Verdict: PASS` —— 同时满足"优于基线"且"预测随初始场景变化"

> 注意：早期版本的评估脚本只加载配置、**随机初始化权重**，会给出假的"坍缩"结论。
> 现在用 `ACTPolicy.from_pretrained(..., strict=True)` 真正加载 `model.safetensors`。

---

## 四、运行时架构

### 4.1 为什么是 5 个进程

硬约束：**LeRobot 要求 Python ≥ 3.12，系统 ROS Humble 是 3.10，IsaacLab 是 3.11** ——
无法同进程，因此按环境切分，观测经 HTTP + base64 JPEG 跨进程传递。

| 节点 | 运行环境 | 职责 |
|---|---|---|
| `act_orchestrator` | 系统 ROS (Py3.10) | `/pick_and_place` action server；reset/spawn/prepare/enable/等 home |
| `act_ros_adapter` | 系统 ROS (Py3.10) | 策略执行器；订阅 enable，按 15Hz 发布关节目标 |
| `script/act_policy_server.py` | conda `lerobot` (Py3.12) | HTTP 推理服务 `/infer_chunk`（非 ROS） |

### 4.2 一轮流程

```text
step 1  reset      → enable(False) + /reset
step 2  spawn      → /spawn_cube
step 3  prepare    → /act_ros_adapter/prepare（预热 chunk 缓冲）
step 4  running    → enable(True)
        判定       → 先离开 home（>0.3 rad），再回到 home（±0.2 rad）并静止 1s
step 11 done       → enable(False) → success
超时 / 取消        → enable(False) → abort / canceled
```

一轮结束不复位，机械臂保持在最后姿态。控制权由 `enable` 话题仲裁：ACT 模式启动时
`control_node` 会忽略 `/arm_controller/joint_trajectory` 与
`/gripper_controller/command`。

完成判定**不用固定延时**，而是监测关节状态（先离开 home → 再回到 home 并静止），
`cycle_timeout`（默认 60s）只作为失败兜底。

### 4.3 三个关键设计

**(a) 绝对位置 → 插值 → 限幅**

- ACT 输出**绝对关节角**，不是速度
- adapter 15Hz 给点，`control_node` 100Hz 平滑插值（`act_command_period`），避免跳变
- `act_max_joint_step=0.15 rad` 限幅，防止策略偶发异常导致机械臂猛甩
- `act_watchdog_sec=0.5s`：命令断了保持**最后目标**

**(b) 分块执行 + 重叠预取（性能核心）**

- 一次推理约 150~200ms，而控制周期只有 66ms；若每步都推理会"一顿一顿"
- adapter 一次取整段 chunk，一边执行一边后台预取下一块 → 推理延迟被藏在缓冲里
- `prefetch_threshold`（默认 10）控制重推理时机，换模型时按 chunk 长度调整

**(c) 夹爪映射**

- 训练标签是**实测** knuckle 位置（接触时约 0.63），不是命令值 0.8
- adapter 把预测值映射为 `0.0`（开）/`0.8`（合），中间维持上一状态
- 直接发 0.63 会夹不紧

> **易踩的坑**：`step_act()` 空闲/超时时**必须保留 `target_pos`**，不能写成
> `current_pos`。把位置目标设成实测位置会让 PD 误差归零、失去抗重力力矩，而每帧
> 重设会持续跟随下沉，表现为**机械臂缓慢下垂**。同理 `_act_enabled_callback` 和
> `_stop_motion_callback` 里的一次性赋值是无害的，但绝不能每帧执行。

### 4.4 话题与服务

| 名称 | 类型 | 说明 |
|---|---|---|
| `/pick_and_place` | `custom_msgs/action/PickAndPlace` | 一轮 = 一个 goal |
| `/isaaclab/act/enabled` | `std_msgs/Bool` | 由 orchestrator 发布，adapter 与 control_node 订阅 |
| `/isaaclab/act/joint_target` | `trajectory_msgs/JointTrajectory` | adapter 发布，单点 7 关节 |
| `/act_ros_adapter/prepare` | `std_srvs/Trigger` | 预热 chunk 缓冲 |
| `/reset`, `/spawn_cube` | `std_srvs/Trigger` | 由 `control_node` 提供 |

---

## 五、换模型要改什么

所有策略都继承 `PreTrainedPolicy`，统一实现
`predict_action_chunk(batch) -> (B, chunk, action_dim)`。所以只要新模型输出
**绝对关节角 chunk**，运行时几乎不动：

| 组件 | 换模型要改吗 |
|---|---|
| `act_ros_adapter`（发关节目标、预取、夹爪映射） | 不用 |
| `act_orchestrator`（reset/spawn/等 home） | 不用 |
| `control_node` / `panel` / `run.sh` 结构 | 不用 |
| `script/act_policy_server.py` | 需泛化：按 `--policy-type` 加载任意策略 |

若新模型输出的不是绝对关节角（例如差分），需要在 adapter 的动作映射里还原。

### LeRobot 可选模型对比

| 模型 | 参数量 | 动作生成 | 视觉骨干 | 观测历史 | chunk | 语言 | 单卡可训 |
|---|---|---|---|---|---|---|---|
| **ACT** | ~50M | 直接回归 | ResNet18 | 1 帧 | 100 | ❌ | ✅ |
| Diffusion | 小 | 迭代去噪 | ResNet18 | 2 帧 | 64 | ❌ | ✅ |
| VQ-BeT | 小 | 码本+回归 | ResNet18 | 5 帧 | 5 | ❌ | ✅ |
| Multi-Task DiT | 中 | 扩散/流匹配 | CLIP ViT-B/16 | 2 帧 | 32 | ✅ | ✅ |
| SmolVLA | ~450M | flow matching | SmolVLM2-500M | 1 帧 | 50 | ✅ | ✅ |
| pi0 / pi05 | ~3B | flow matching | Gemma | 1 帧 | 50 | ✅ | 勉强 |

**关键区别**：

- **多模态动作分布**：脚本专家数据是确定性的，同一个观测只有一种走法，所以
  Diffusion/VQ-BeT 的"多模态建模"优势在本任务用不上——这也是纯回归的 ACT 够用的原因
- **语言条件**（Multi-Task DiT / SmolVLA / pi0）：只有数据里**指令真实变化**时才有意义。
  当前数据集 `total_tasks = 1`，语言通路没有区分度，换 VLA 只会得到"又大又慢的 ACT"
- **chunk 长度影响运行时**：本架构 chunk 越长越好用；VQ-BeT 只有 5 步，重推理间隔仅
  0.33s，适配成本高
- **图像预处理不同**：pi0/pi05 要 `224×224`，groot `256×256`，SmolVLA `512×512` 带
  padding；本项目数据是 `256×256`

---

## 六、启动

由仓库根目录的 `./run.sh --act` 启动（5 个进程），不要单独手动拉起。

```text
[1] script/act_policy_server.py      (lerobot 环境)
[2] isaaclab/src/run_isaaclab.py --act
[3] ros2 run act act_orchestrator
[4] ros2 run act act_ros_adapter
[5] script/panel.py
```

环境变量（可覆盖）：`ACT_CHECKPOINT`（默认
`outputs/ur5e_act_rgb_abs_novae/checkpoints/030000/pretrained_model`）、
`ACT_PORT`（默认 `27655`）、`LEROBOT_ENV`（默认 `lerobot`）。

关键 ROS 参数：`cycle_timeout`（60s）、`leave_timeout`（15s）、
`home_leave_threshold`（0.3）、`home_return_tolerance`（0.2）、
`home_settle_seconds`（1.0）。
