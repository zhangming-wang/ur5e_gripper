#!/bin/bash
# UR5e + Robotiq 2F-85 一键启动
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$PROJECT_DIR/install"

export ROS_DOMAIN_ID=46

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

    setsid bash -c "$_command" &
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
    "")         MODE="mujoco" ;;
    *)          usage ;;
esac

echo "=============================================="
echo " UR5e + Robotiq 2F-85 Gripper"
echo "  模式: $MODE"
echo "  ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "=============================================="

# ---- MoveIt (共用) ----
echo "[1] 启动 MoveIt..."
if [ "$MODE" = "mujoco" ]; then
    USE_SIM_TIME="true"
else
    USE_SIM_TIME="false"
fi
start_job "
    source /opt/ros/humble/setup.bash
    [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
    ros2 launch moveit_config moveit.launch.py \
        use_gripper:=true use_sim_time:=$USE_SIM_TIME launch_rviz:=true
"

if [ "$MODE" = "isaacsim" ]; then

    # ---- Bridge ----
    echo "[2] 启动 Bridge..."
    start_job "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        python3 '$PROJECT_DIR/isaaclab/src/bridge_node.py'
    "

    # ---- Planning ----
    echo "[3] 启动 Planning..."
    start_job "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        ros2 run planning planning_node
    "

    # ---- Perception ----
    echo "[4] 启动 Perception..."
    start_job "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        ros2 launch perception perception.launch.py
    "

    # ---- IsaacLab ----
    echo "[5] 启动 IsaacLab..."
    start_job "
        source /home/dev/miniconda3/etc/profile.d/conda.sh
        conda activate isaaclab
        cd '$PROJECT_DIR' && python3 isaaclab/src/run_isaaclab.py
    "

    # ---- Orchestrator ----
    echo "[6] 启动 Orchestrator..."
    start_job "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        ros2 run orchestrator orchestrator_node
    "

    # ---- 面板 ----
    echo "[7] 启动调试面板..."
    start_job "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        python3 '$PROJECT_DIR/script/panel.py'
    "

elif [ "$MODE" = "mujoco" ]; then

    # ---- MuJoCo ros2_control ----
    echo "[2] 启动 MuJoCo..."
    start_job "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        ros2 launch mujoco mujoco.launch.py
    "

    # ---- Planning ----
    echo "[3] 启动 Planning..."
    start_job "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        ros2 run planning planning_node
    "

    # ---- 面板 ----
    echo "[4] 启动调试面板..."
    start_job "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        python3 '$PROJECT_DIR/script/panel.py'
    "

fi

echo ""
echo "[INFO] 全部启动完成。按 Ctrl+C 停止"
wait
