#!/bin/bash
# UR5e + Robotiq 2F-85 一键启动
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$PROJECT_DIR/install"
CONDA_ROOT="${CONDA_ROOT:-/home/dev/miniconda3}"
ISAACLAB_ENV="${ISAACLAB_ENV:-isaaclab}"

# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# export ROS_DOMAIN_ID=46

JOBS=""
_cleaned=0
MODE="mujoco"

kill_tree() {
    local _pid=$1
    for _child in $(ps -o pid= --ppid $_pid 2>/dev/null); do
        kill_tree $_child
    done
    kill -TERM $_pid 2>/dev/null || true
}

start_job() {
    local _command=$1
    local _pid
    local _pgid=""

    setsid bash -e -c "$_command" &
    _pid=$!
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        _pgid=$(ps -o pgid= -p "$_pid" 2>/dev/null | tr -d ' ' || true)
        [ -n "$_pgid" ] && break
        sleep 0.05
    done
    JOBS="$JOBS $_pid:$_pgid"
}

stop_job() {
    local _job=$1
    local _pid
    local _pgid

    IFS=: read -r _pid _pgid <<< "$_job"
    if [ -n "$_pgid" ]; then
        kill -TERM -- "-$_pgid" 2>/dev/null || true
    else
        kill_tree "$_pid"
    fi
}

force_stop_job() {
    local _job=$1
    local _pid
    local _pgid

    IFS=: read -r _pid _pgid <<< "$_job"
    if [ -n "$_pgid" ]; then
        kill -KILL -- "-$_pgid" 2>/dev/null || true
    fi
    kill -KILL "$_pid" 2>/dev/null || true
}

cleanup() {
    [ $_cleaned -eq 1 ] && return
    _cleaned=1
    echo ""
    echo "[INFO] 正在停止所有进程..."
    for job in $JOBS; do
        stop_job "$job"
    done
    sleep 1
    for job in $JOBS; do
        force_stop_job "$job"
        IFS=: read -r pid _ <<< "$job"
        wait "$pid" 2>/dev/null || true
    done
    echo "[INFO] 已全部停止"
}
trap cleanup EXIT INT TERM

usage() {
    echo "用法: $0 [--isaacsim | --mujoco]"
    echo "  不传参数时默认使用 MuJoCo"
    echo "  --isaacsim    Isaac Lab 仿真"
    echo "  --mujoco      MuJoCo 仿真"
    exit 1
}

case "${1:-}" in
    --isaacsim) MODE="isaacsim" ;;
    --mujoco)   MODE="mujoco" ;;
    "")         MODE="isaacsim" ;;
    *)          usage ;;
esac

if [ ! -f "$INSTALL_DIR/setup.bash" ]; then
    echo "[ERROR] Missing $INSTALL_DIR/setup.bash; build the workspace first" >&2
    exit 1
fi
if [ "$MODE" = "isaacsim" ] && [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "[ERROR] Conda initialization script not found: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi
if [ "$MODE" = "isaacsim" ]; then
    # Resolve the environment before starting any ROS processes.
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    if ! conda run -n "$ISAACLAB_ENV" true >/dev/null 2>&1; then
        echo "[ERROR] Conda environment not found: $ISAACLAB_ENV" >&2
        exit 1
    fi
fi

echo "=============================================="
echo " UR5e + Robotiq 2F-85 Gripper"
echo "  模式: $MODE"
echo "  ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "=============================================="

# ---- MoveIt (共用) ----
echo "[1] 启动 MoveIt..."
if [ "$MODE" = "mujoco" ]; then
    USE_SIM_TIME="true"
    LAUNCH_MOVEIT_RSP="false"
else
    USE_SIM_TIME="false"
    LAUNCH_MOVEIT_RSP="true"
fi
start_job "
    source /opt/ros/humble/setup.bash
    source '$INSTALL_DIR/setup.bash'
    ros2 launch moveit_config moveit.launch.py \
        use_gripper:=true use_sim_time:=$USE_SIM_TIME \
        launch_robot_state_publisher:=$LAUNCH_MOVEIT_RSP launch_rviz:=true
"

if [ "$MODE" = "isaacsim" ]; then

    # ---- Bridge ----
    echo "[2] 启动 Bridge..."
    start_job "
        source /opt/ros/humble/setup.bash
        source '$INSTALL_DIR/setup.bash'
        python3 '$PROJECT_DIR/isaaclab/src/bridge_node.py'
    "

    # ---- Planning ----
    echo "[3] 启动 Planning..."
    start_job "
        source /opt/ros/humble/setup.bash
        source '$INSTALL_DIR/setup.bash'
        ros2 run planning planning_node --ros-args -p use_sim_time:=$USE_SIM_TIME
    "

    # ---- Perception ----
    echo "[4] 启动 Perception..."
    start_job "
        source /opt/ros/humble/setup.bash
        source '$INSTALL_DIR/setup.bash'
        ros2 launch perception perception.launch.py use_sim_time:=$USE_SIM_TIME
    "

    # ---- IsaacLab ----
    echo "[5] 启动 IsaacLab..."
    start_job "
        source '$CONDA_ROOT/etc/profile.d/conda.sh'
        conda activate '$ISAACLAB_ENV'
        cd '$PROJECT_DIR' && python3 isaaclab/src/run_isaaclab.py
    "

    # ---- Orchestrator ----
    echo "[6] 启动 Orchestrator..."
    start_job "
        source /opt/ros/humble/setup.bash
        source '$INSTALL_DIR/setup.bash'
        ros2 run orchestrator orchestrator_node --ros-args \
            -p use_sim_time:=$USE_SIM_TIME -p plan_timeout_sec:=60.0
    "

    # ---- 面板 ----
    echo "[7] 启动调试面板..."
    start_job "
        source /opt/ros/humble/setup.bash
        source '$INSTALL_DIR/setup.bash'
        python3 '$PROJECT_DIR/script/panel.py' --ros-args -p use_sim_time:=$USE_SIM_TIME
    "

elif [ "$MODE" = "mujoco" ]; then

    # ---- MuJoCo ros2_control ----
    echo "[2] 启动 MuJoCo..."
    start_job "
        source /opt/ros/humble/setup.bash
        source '$INSTALL_DIR/setup.bash'
        ros2 launch mujoco mujoco.launch.py
    "

    # ---- Perception ----
    echo "[3] 启动 Perception..."
    start_job "
        source /opt/ros/humble/setup.bash
        source '$INSTALL_DIR/setup.bash'
        ros2 launch perception perception.launch.py use_sim_time:=$USE_SIM_TIME
    "

    # ---- Planning ----
    echo "[4] 启动 Planning..."
    start_job "
        source /opt/ros/humble/setup.bash
        source '$INSTALL_DIR/setup.bash'
        ros2 run planning planning_node --ros-args -p use_sim_time:=$USE_SIM_TIME
    "

    # ---- Orchestrator ----
    echo "[5] 启动 Orchestrator..."
    start_job "
        source /opt/ros/humble/setup.bash
        source '$INSTALL_DIR/setup.bash'
        ros2 run orchestrator orchestrator_node --ros-args -p use_sim_time:=$USE_SIM_TIME
    "

    # ---- 面板 ----
    echo "[6] 启动调试面板..."
    start_job "
        source /opt/ros/humble/setup.bash
        source '$INSTALL_DIR/setup.bash'
        python3 '$PROJECT_DIR/script/panel.py' --ros-args -p use_sim_time:=$USE_SIM_TIME
    "

fi

echo ""
echo "[INFO] 全部启动完成。按 Ctrl+C 停止"

while true; do
    for job in $JOBS; do
        IFS=: read -r pid _pgid <<< "$job"
        if ! kill -0 "$pid" 2>/dev/null; then
            status=0
            wait "$pid" 2>/dev/null || status=$?
            if [ "$status" -eq 0 ]; then
                status=1
            fi
            echo "[ERROR] Background job $pid exited with status $status" >&2
            exit "$status"
        fi
    done
    sleep 1
done
