#!/bin/bash
# UR5e + Robotiq 2F-85 一键启动
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$PROJECT_DIR/install"

ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-40}"
export ROS_DOMAIN_ID

PIDS=""
_cleaned=0

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

echo "=============================================="
echo " UR5e + Robotiq 2F-85 Gripper"
echo "  ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "=============================================="

# ---- MoveIt ----
echo "[1/4] 启动 MoveIt..."
setsid bash -c "
    source /opt/ros/humble/setup.bash
    [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
    ros2 launch ur5e_gripper_moveit_config moveit.launch.py \
        use_gripper:=true use_sim_time:=false launch_rviz:=true
" &
PIDS="$PIDS $!"

# ---- Bridge ----
echo "[2/4] 启动 Bridge..."
setsid bash -c "
    source /opt/ros/humble/setup.bash
    [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
    python3 '$PROJECT_DIR/ur5e_gripper_isaaclab/src/bridge_node.py'
" &
PIDS="$PIDS $!"

# ---- Planning ----
echo "[3/4] 启动 Planning..."
setsid bash -c "
    source /opt/ros/humble/setup.bash
    [ -f '$INSTALL_DIR/setup.bash' ] && source '$INSTALL_DIR/setup.bash'
    python3 '$PROJECT_DIR/ur5e_gripper_isaaclab/src/planning_node.py'
" &
PIDS="$PIDS $!"

# ---- IsaacLab ----
echo "[4/4] 启动 IsaacLab..."
setsid bash -c "
    source /home/dev/miniconda3/etc/profile.d/conda.sh
    conda activate isaaclab
    cd '$PROJECT_DIR' && python3 ur5e_gripper_isaaclab/src/run_isaaclab.py
" &
PIDS="$PIDS $!"

echo ""
echo "[INFO] 全部启动完成。按 Ctrl+C 停止"
wait
