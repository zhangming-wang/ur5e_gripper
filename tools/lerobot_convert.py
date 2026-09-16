#!/usr/bin/env python3
"""Convert IsaacSim recordings to an RGB-only LeRobotDataset v3 dataset.

Run in the LeRobot environment:
  conda activate lerobot
  python tools/lerobot_convert.py

Input:
  raw_data/episode_XXXXXX/{frames_rgb, states.csv, meta.json}
Output:
  lerobot_data_rgb_abs/  (Parquet + MP4 + metadata)

Actions are the absolute joint state at the next retained frame. Leading idle
frames are trimmed so the same initial observation is not labelled as both a
long hold and the start of motion.
"""

import argparse
import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets import LeRobotDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "raw_data"
OUT_ROOT = PROJECT_DIR / "lerobot_data_rgb_abs"
REPO_ID = "ur5e_pick_place"
IMG_SIZE = (256, 256)
MOTION_THRESHOLD = 1e-4
PRE_MOTION_FRAMES = 2

STATE_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "robotiq_85_left_knuckle_joint",
]

FEATURES = {
    "observation.images.cam_high": {
        "dtype": "video",
        "shape": (3, 256, 256),
        "names": ["channels", "height", "width"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": STATE_JOINTS,
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": STATE_JOINTS,
    },
}


@dataclass(frozen=True)
class Episode:
    path: Path
    prompt: str
    fps: float
    states: np.ndarray
    rgb_paths: list[Path]
    start_frame: int

    @property
    def retained_frames(self) -> int:
        return len(self.states) - self.start_frame


def _indexed_frame_paths(directory: Path, suffix: str, expected_count: int) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Missing frame directory: {directory}")

    paths = sorted(directory.glob(f"frame_*.{suffix}"))
    expected_names = [f"frame_{i:06d}.{suffix}" for i in range(expected_count)]
    actual_names = [path.name for path in paths]
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(expected_names))
        raise ValueError(
            f"Invalid frame sequence in {directory}: expected={expected_count}, actual={len(paths)}, "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return paths


def _load_episode(path: Path, motion_threshold: float, pre_motion_frames: int) -> Episode:
    meta_path = path / "meta.json"
    csv_path = path / "states.csv"
    if not meta_path.is_file() or not csv_path.is_file():
        raise ValueError(f"Missing meta.json or states.csv in {path}")

    with meta_path.open() as file:
        meta = json.load(file)

    with csv_path.open(newline="") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {csv_path}") from exc
        expected_header = ["frame_index", "timestamp_s", *STATE_JOINTS]
        if header != expected_header:
            raise ValueError(f"Unexpected CSV header in {csv_path}: {header}")

        rows = list(reader)

    if len(rows) < 2:
        raise ValueError(f"Episode needs at least 2 frames: {path}")
    frame_indices = [int(row[0]) for row in rows]
    if frame_indices != list(range(len(rows))):
        raise ValueError(f"Non-contiguous frame_index values in {csv_path}")

    fps = float(meta.get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid fps in {meta_path}: {fps}")
    timestamps = np.asarray([float(row[1]) for row in rows], dtype=np.float64)
    expected_period = 1.0 / fps
    if not np.isfinite(timestamps).all() or not np.allclose(
        np.diff(timestamps), expected_period, rtol=0.0, atol=5e-4
    ):
        raise ValueError(f"Invalid timestamp cadence in {csv_path}; expected {expected_period:.6f}s")

    states = np.asarray([[float(value) for value in row[2:]] for row in rows], dtype=np.float32)
    if states.shape != (len(rows), len(STATE_JOINTS)) or not np.isfinite(states).all():
        raise ValueError(f"Invalid state data in {csv_path}: shape={states.shape}")

    meta_frames = int(meta.get("num_frames", -1))
    if meta_frames != len(states):
        raise ValueError(f"Frame count mismatch in {path}: meta={meta_frames}, csv={len(states)}")
    if meta.get("state_joints") != STATE_JOINTS:
        raise ValueError(f"Unexpected state_joints in {meta_path}: {meta.get('state_joints')}")

    rgb_paths = _indexed_frame_paths(path / "frames_rgb", "jpg", len(states))
    motion = np.max(np.abs(np.diff(states, axis=0)), axis=1)
    moving = np.flatnonzero(motion > motion_threshold)
    if moving.size == 0:
        raise ValueError(f"No motion above {motion_threshold:g} rad in {path}")
    start_frame = max(0, int(moving[0]) - pre_motion_frames)

    return Episode(
        path=path,
        prompt=str(meta["prompt"]),
        fps=fps,
        states=states,
        rgb_paths=rgb_paths,
        start_frame=start_frame,
    )


def _load_and_validate_episodes(
    raw_dir: Path, motion_threshold: float, pre_motion_frames: int
) -> tuple[list[Episode], int]:
    episode_paths = sorted(path for path in raw_dir.glob("episode_*") if path.is_dir())
    if not episode_paths:
        raise ValueError(f"No episode directories found in {raw_dir}")

    episodes = [
        _load_episode(path, motion_threshold=motion_threshold, pre_motion_frames=pre_motion_frames)
        for path in episode_paths
    ]
    fps_values = {episode.fps for episode in episodes}
    if len(fps_values) != 1:
        raise ValueError(f"Episodes use different FPS values: {sorted(fps_values)}")
    fps = fps_values.pop()
    rounded_fps = round(fps)
    if not np.isclose(fps, rounded_fps):
        raise ValueError(f"LeRobot requires an integer FPS, got {fps}")
    return episodes, rounded_fps


def _convert_episode(dataset: LeRobotDataset, episode: Episode) -> None:
    states = episode.states[episode.start_frame :]
    actions = np.concatenate([states[1:], states[-1:]], axis=0)

    for output_index, source_index in enumerate(range(episode.start_frame, len(episode.states))):
        rgb = cv2.imread(str(episode.rgb_paths[source_index]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"Failed to read {episode.rgb_paths[source_index]}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)

        dataset.add_frame(
            {
                "observation.images.cam_high": rgb,
                "observation.state": states[output_index],
                "action": actions[output_index],
                "task": episode.prompt,
            }
        )

    dataset.save_episode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--motion-threshold", type=float, default=MOTION_THRESHOLD)
    parser.add_argument("--pre-motion-frames", type=int, default=PRE_MOTION_FRAMES)
    args = parser.parse_args()

    if args.motion_threshold < 0:
        parser.error("--motion-threshold must be non-negative")
    if args.pre_motion_frames < 0:
        parser.error("--pre-motion-frames must be non-negative")
    if args.out_root.exists():
        parser.error(f"output directory already exists: {args.out_root}")

    episodes, fps = _load_and_validate_episodes(
        args.raw_dir,
        motion_threshold=args.motion_threshold,
        pre_motion_frames=args.pre_motion_frames,
    )
    source_frames = sum(len(episode.states) for episode in episodes)
    retained_frames = sum(episode.retained_frames for episode in episodes)
    starts = [episode.start_frame for episode in episodes]
    print(
        f"Validated {len(episodes)} episodes: fps={fps}, source_frames={source_frames}, "
        f"retained_frames={retained_frames}, trimmed={source_frames - retained_frames} "
        f"({100.0 * (source_frames - retained_frames) / source_frames:.1f}%)"
    )
    print(f"Leading trim range: {min(starts)}-{max(starts)} frames")

    args.out_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.out_root.name}.", dir=args.out_root.parent
    ) as temporary_dir:
        staging_root = Path(temporary_dir) / "dataset"
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=fps,
            robot_type="ur5e",
            root=staging_root,
            features=FEATURES,
        )

        for index, episode in enumerate(episodes):
            _convert_episode(dataset, episode)
            print(
                f"  [{index + 1:02d}/{len(episodes):02d}] {episode.path.name}: "
                f"start={episode.start_frame}, frames={episode.retained_frames}"
            )

        dataset.finalize()
        staging_root.rename(args.out_root)
    print(f"Dataset written to {args.out_root}")


if __name__ == "__main__":
    main()
