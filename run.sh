#!/bin/bash
# UR5e + Robotiq 2F-85 一键启动
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$PROJECT_DIR/install"

export ROS_DOMAIN_ID=46

PIDS=""
_cleaned=0
MODE="mujoco"

kill_tree() {
    local _pid=$1
    for _child in $(ps -o pid= --ppid $_pid 2>/dev/null); do
        kill_tree $_child
    done
    kill -TERM $_pid 2>/dev/null || true
}

cleanup() {
    [ $_cleaned -eq 1 ] && return
    _cleaned=1
    echo ""
    echo "[INFO] 正在停止所有进程..."
    for pid in $PIDS; do
        kill_tree $pid
    done
    sleep 1
    for pid in $PIDS; do
        kill -KILL $(ps -o pid= --ppid $pid 2>/dev/null) 2>/dev/null || true
        kill -KILL $pid 2>/dev/null || true
    done
    echo "[INFO] 已全部停止"
}
trap cleanup EXIT INT TERM

usage() {
    echo "用法: $0 --isaacsim | --mujoco"
    echo "  --isaacsim    Isaac Lab 仿真"
    echo "  --mujoco      MuJoCo 仿真"
    exit 1
}

case "${1:-}" in
    --isaacsim) MODE="isaacsim" ;;
    --mujoco)   MODE="mujoco" ;;
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
setsid bash -c "
    source /opt/ros/humble/setup.bash
    [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
    ros2 launch moveit_config moveit.launch.py \
        use_gripper:=true use_sim_time:=$USE_SIM_TIME launch_rviz:=true
" &
PIDS="$PIDS $!"

if [ "$MODE" = "isaacsim" ]; then

    # ---- Bridge ----
    echo "[2] 启动 Bridge..."
    setsid bash -c "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        python3 '$PROJECT_DIR/isaaclab/src/bridge_node.py'
    " &
    PIDS="$PIDS $!"

    # ---- Planning ----
    echo "[3] 启动 Planning..."
    setsid bash -c "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        ros2 run planning planning_node
    " &
    PIDS="$PIDS $!"

    # ---- Perception ----
    echo "[4] 启动 Perception..."
    setsid bash -c "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        ros2 run perception perception_node
    " &
    PIDS="$PIDS $!"

    # ---- IsaacLab ----
    echo "[5] 启动 IsaacLab..."
    setsid bash -c "
        source /home/dev/miniconda3/etc/profile.d/conda.sh
        conda activate isaaclab
        cd '$PROJECT_DIR' && python3 isaaclab/src/run_isaaclab.py
    " &
    PIDS="$PIDS $!"

    # ---- Orchestrator ----
    echo "[6] 启动 Orchestrator..."
    setsid bash -c "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        ros2 run orchestrator orchestrator_node
    " &
    PIDS="$PIDS $!"

    # ---- 面板 ----
    echo "[7] 启动调试面板..."
    setsid bash -c "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        python3 '$PROJECT_DIR/script/panel.py'
    " &
    PIDS="$PIDS $!"

elif [ "$MODE" = "mujoco" ]; then

    # ---- MuJoCo ros2_control ----
    echo "[2] 启动 MuJoCo..."
    setsid bash -c "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        ros2 launch mujoco mujoco.launch.py
    " &
    PIDS="$PIDS $!"

    # ---- Planning ----
    echo "[3] 启动 Planning..."
    setsid bash -c "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        ros2 run planning planning_node
    " &
    PIDS="$PIDS $!"

    # ---- 面板 ----
    echo "[4] 启动调试面板..."
    setsid bash -c "
        source /opt/ros/humble/setup.bash
        [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
        python3 '$PROJECT_DIR/script/panel.py'
    " &
    PIDS="$PIDS $!"

fi

echo ""
echo "[INFO] 全部启动完成。按 Ctrl+C 停止"
wait
