#!/usr/bin/env python3
"""在 MuJoCo 中加载并显示 UR5e + Robotiq 2F-85 组合场景。

用法:
    python mujoco/src/run_mujoco.py              # 交互 viewer
    python mujoco/src/run_mujoco.py --headless   # 无窗口

键盘(窗口内): R 复位, G 开爪, C 闭爪, Esc 退出
"""

import argparse
import os

import mujoco
import mujoco.viewer

XML_DIR = os.path.join(os.path.dirname(__file__), "..", "xml")
UR5E_SCENE = os.path.join(XML_DIR, "universal_robots_ur5e", "scene.xml")
GRIPPER_XML = os.path.join(XML_DIR, "robotiq_2f85", "2f85.xml")

GRIPPER_OPEN = 90.0
GRIPPER_CLOSE = 30.0


def build_model():
    ur5e = mujoco.MjSpec.from_file(UR5E_SCENE)
    gripper = mujoco.MjSpec.from_file(GRIPPER_XML)
    wrist3 = next(b for b in ur5e.bodies if b.name == "wrist_3_link")
    site = next(s for s in wrist3.sites if s.name == "attachment_site")
    ur5e.attach(gripper, site=site)
    model = ur5e.compile()
    if model is None:
        raise RuntimeError("compile failed")
    return model


def find_gripper_actuator(model):
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name == "/fingers_actuator":
            return i
    return -1


def run_viewer(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    gripper_actuator = find_gripper_actuator(model)

    def key_cb(keycode):
        if keycode == ord("R"):
            mujoco.mj_resetDataKeyframe(model, data, 0)
            print("[run_mujoco] home")
        elif keycode == ord("G") and gripper_actuator >= 0:
            data.ctrl[gripper_actuator] = GRIPPER_OPEN
            print("[run_mujoco] gripper open")
        elif keycode == ord("C") and gripper_actuator >= 0:
            data.ctrl[gripper_actuator] = GRIPPER_CLOSE
            print("[run_mujoco] gripper close")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 120
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


def run_headless(model, seconds=10.0):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    gripper_actuator = find_gripper_actuator(model)
    if gripper_actuator >= 0:
        data.ctrl[gripper_actuator] = GRIPPER_OPEN
    nsteps = int(seconds / model.opt.timestep)
    for i in range(nsteps):
        mujoco.mj_step(model, data)
        if i % 1000 == 0:
            print(f"[run_mujoco] t={data.time:.2f}s")
    print("[run_mujoco] done")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo UR5e + Robotiq 2F-85")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    model = build_model()
    print(f"[run_mujoco] loaded: {model.nq} qpos, {model.nu} actuators")
    if args.headless:
        run_headless(model)
    else:
        run_viewer(model)


if __name__ == "__main__":
    main()
