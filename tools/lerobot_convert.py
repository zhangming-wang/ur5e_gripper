#!/usr/bin/env python3
"""raw_data → LeRobotDataset v3 转换脚本

在 lerobot conda 环境运行:
  conda activate lerobot
  python tools/lerobot_convert.py

输入:  raw_data/episode_XXXXXX/{frames_rgb, frames_depth, states.csv, meta.json}
输出:  lerobot_data/ur5e_pick_place/  (v3 格式: parquet + MP4 + meta)
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

from lerobot.datasets import LeRobotDataset

RAW_DIR = "/home/dev/work/ur5e_gripper/raw_data"
OUT_ROOT = "/home/dev/work/ur5e_gripper/lerobot_data"
REPO_ID = "ur5e_pick_place"
FPS = 15
IMG_SIZE = (256, 256)

FEATURES = {
    "observation.images.cam_high": {
        "dtype": "video",
        "shape": (3, 256, 256),
        "names": ["channels", "height", "width"],
    },
    "observation.images.depth": {
        "dtype": "video",
        "shape": (1, 256, 256),
        "names": ["channels", "height", "width"],
        "info": {"is_depth_map": True},
    },
    "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
}


def load_episode(ep_dir):
    meta_path = os.path.join(ep_dir, "meta.json")
    csv_path = os.path.join(ep_dir, "states.csv")
    with open(meta_path) as f:
        meta = json.load(f)
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in r[2:]] for r in reader]  # 跳过 frame_index/timestamp
    return meta, np.array(rows, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--skip-depth", action="store_true", help="深度编码失败时降级: 只转 RGB")
    args = parser.parse_args()

    ep_dirs = sorted(
        d for d in (os.path.join(args.raw_dir, n) for n in os.listdir(args.raw_dir))
        if os.path.isdir(d) and os.path.basename(d).startswith("episode_")
    )
    if not ep_dirs:
        print(f"没有找到 episode 目录: {args.raw_dir}")
        sys.exit(1)
    print(f"待转换: {len(ep_dirs)} 集")

    features = dict(FEATURES)
    if args.skip_depth:
        features.pop("observation.images.depth")

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        robot_type="ur5e",
        root=args.out_root,
        features=features,
    )

    for ep_i, ep_dir in enumerate(ep_dirs):
        meta, states = load_episode(ep_dir)
        n_csv = states.shape[0]
        rgb_dir = os.path.join(ep_dir, "frames_rgb")
        n_frames = len([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
        n = min(n_csv, n_frames)
        if n_csv != n_frames:
            print(f"  ⚠ {os.path.basename(ep_dir)}: csv={n_csv} 帧={n_frames}, 取 {n}")

        actions = np.vstack(
            [states[1:] - states[:-1], np.zeros((1, states.shape[1]), dtype=states.dtype)]
        )

        for i in range(n):
            rgb = cv2.imread(os.path.join(rgb_dir, f"frame_{i:06d}.jpg"))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)

            frame = {
                "observation.images.cam_high": rgb,
                "observation.state": states[i].astype(np.float32),
                "action": actions[i].astype(np.float32),
                "task": meta["prompt"],
            }

            if not args.skip_depth:
                depth = cv2.imread(
                    os.path.join(ep_dir, "frames_depth", f"frame_{i:06d}.png"),
                    cv2.IMREAD_UNCHANGED,
                )
                depth = cv2.resize(depth, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
                frame["observation.images.depth"] = depth[..., None]  # (256,256,1) uint16

            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"  ✓ {os.path.basename(ep_dir)} ({n} 帧)")

    dataset.finalize()
    print(f"\n完成! 数据集: {os.path.join(args.out_root, REPO_ID)}")


if __name__ == "__main__":
    main()
