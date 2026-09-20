#!/usr/bin/env python3
"""Evaluate a Diffusion Policy checkpoint against absolute-action held-out data."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import resolve_episode_indices
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_DIR / "outputs/ur5e_diffusion_abs/checkpoints/last/pretrained_model"
DEFAULT_DATASET_ROOT = PROJECT_DIR / "lerobot_data_rgb_abs"
IMAGE_KEY = "observation.images.cam_high"
STATE_KEY = "observation.state"


def resolve_checkpoint(path: Path) -> Path:
    for candidate in (path, path / "pretrained_model", path / "checkpoints/last/pretrained_model"):
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {path}")


def resolve_eval_episodes(metadata: LeRobotDatasetMetadata, count: int | None) -> list[int]:
    if count is not None:
        return list(range(metadata.total_episodes - min(count, metadata.total_episodes), metadata.total_episodes))
    # Training used a 20% held-out split and preserves episode ordering per task.
    task_to_episodes: dict[str, list[int]] = {}
    for episode_index, tasks in enumerate(metadata.episodes["tasks"]):
        task_to_episodes.setdefault(tasks[0] if tasks else "", []).append(episode_index)
    result = []
    for episodes in task_to_episodes.values():
        result.extend(episodes[-math.ceil(len(episodes) * 0.2):])
    return result


def sample_indices(dataset: LeRobotDataset, episodes: list[int], horizon: int, samples_per_episode: int) -> list[int]:
    episode_column = np.asarray(dataset.hf_dataset["episode_index"]).reshape(-1)
    frame_column = np.asarray(dataset.hf_dataset["frame_index"]).reshape(-1)
    indices = []
    for episode in episodes:
        rows = np.flatnonzero(episode_column == episode)
        if rows.size <= horizon:
            raise ValueError(f"episode {episode} is too short for horizon={horizon}")
        first_frame = int(frame_column[rows[0]])
        last_frame = int(frame_column[rows[-1]])
        # Observation [-1, 0] and action [-1, ..., horizon-2] must be unpadded.
        valid = rows[(frame_column[rows] >= first_frame + 1) & (frame_column[rows] <= last_frame - (horizon - 2))]
        if not valid.size:
            raise ValueError(f"episode {episode} has no unpadded Diffusion samples")
        positions = np.linspace(0, valid.size - 1, min(samples_per_episode, valid.size)).astype(int)
        indices.extend(int(valid[position]) for position in np.unique(positions))
    return indices


def normalize_min_max(values: np.ndarray, stats: dict, eps: float) -> np.ndarray:
    lower = np.asarray(stats["min"], dtype=np.float64)
    upper = np.asarray(stats["max"], dtype=np.float64)
    return 2.0 * (values - lower) / np.maximum(upper - lower, eps) - 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default="ur5e_pick_place")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--samples-per-episode", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--video-backend", default="pyav")
    args = parser.parse_args()
    if args.samples_per_episode <= 0 or (args.eval_episodes is not None and args.eval_episodes <= 0):
        parser.error("episode and sample counts must be positive")

    checkpoint = resolve_checkpoint(args.checkpoint)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    config = DiffusionConfig.from_pretrained(checkpoint)
    config.device = device
    config.pretrained_backbone_weights = None
    if args.num_inference_steps is not None:
        config.num_inference_steps = args.num_inference_steps
    policy = DiffusionPolicy.from_pretrained(checkpoint, config=config, strict=True)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    episodes = resolve_eval_episodes(metadata, args.eval_episodes)
    observation_offsets = [index - config.n_obs_steps + 1 for index in range(config.n_obs_steps)]
    action_offsets = [index - config.n_obs_steps + 1 for index in range(config.horizon)]
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps={
            IMAGE_KEY: [offset / metadata.fps for offset in observation_offsets],
            STATE_KEY: [offset / metadata.fps for offset in observation_offsets],
            "action": [offset / metadata.fps for offset in action_offsets],
        },
        video_backend=args.video_backend,
    )
    indices = sample_indices(dataset, episodes, config.horizon, args.samples_per_episode)
    action_start = config.n_obs_steps - 1
    action_end = action_start + config.n_action_steps
    raw_predictions, raw_targets, raw_baselines, normalized_predictions = [], [], [], []
    print(
        f"Checkpoint: {checkpoint}\n"
        f"Diffusion config: device={device}, obs={config.n_obs_steps}, horizon={config.horizon}, "
        f"actions={config.n_action_steps}, steps={policy.diffusion.num_inference_steps}\n"
        f"Evaluating {len(indices)} unpadded windows from episodes {episodes}"
    )
    for count, index in enumerate(indices, start=1):
        sample = dataset[index]
        if sample["action_is_pad"].any():
            raise ValueError(f"padded action window at dataset index {index}")
        batch = preprocessor({IMAGE_KEY: sample[IMAGE_KEY], STATE_KEY: sample[STATE_KEY]})
        batch[IMAGE_KEY] = batch[IMAGE_KEY].unsqueeze(0)
        batch[STATE_KEY] = batch[STATE_KEY].unsqueeze(0)
        if batch[IMAGE_KEY].ndim != 5 or batch[STATE_KEY].shape[:2] != (1, config.n_obs_steps):
            raise ValueError(
                f"unexpected observation shapes: image={tuple(batch[IMAGE_KEY].shape)}, "
                f"state={tuple(batch[STATE_KEY].shape)}"
            )
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(batch)
            prediction = postprocessor(normalized.clone())
        prediction = prediction.squeeze(0).cpu().numpy()
        normalized = normalized.squeeze(0).cpu().numpy()
        target = sample["action"].cpu().numpy()[action_start:action_end]
        state = sample[STATE_KEY].cpu().numpy()[-1]
        if prediction.shape != target.shape or prediction.shape[0] != config.n_action_steps:
            raise ValueError(f"prediction={prediction.shape}, target={target.shape}")
        raw_predictions.append(prediction)
        raw_targets.append(target)
        raw_baselines.append(np.repeat(state[None, :], config.n_action_steps, axis=0))
        normalized_predictions.append(normalized)
        print(f"  evaluated {count:03d}/{len(indices):03d}", end="\r", flush=True)

    predictions = np.stack(raw_predictions)
    targets = np.stack(raw_targets)
    baselines = np.stack(raw_baselines)
    normalized_predictions = np.stack(normalized_predictions)
    normalizer = next(step for step in preprocessor.steps if hasattr(step, "stats") and "action" in step.stats)
    normalized_targets = normalize_min_max(targets, normalizer.stats["action"], normalizer.eps)
    normalized_baselines = normalize_min_max(baselines, normalizer.stats["action"], normalizer.eps)
    model_score = float(np.mean(np.abs(normalized_predictions - normalized_targets)))
    baseline_score = float(np.mean(np.abs(normalized_baselines - normalized_targets)))
    raw_mae = np.mean(np.abs(predictions - targets), axis=(0, 1))
    horizons = sorted({0, config.n_action_steps // 4, config.n_action_steps // 2, config.n_action_steps - 1})
    horizon_errors = {step: float(np.mean(np.abs(predictions[:, step] - targets[:, step]))) for step in horizons}
    print(" " * 50, end="\r")
    print(f"Normalized L1: model={model_score:.4f}, baseline={baseline_score:.4f}, ratio={model_score / max(baseline_score, 1e-12):.3f}")
    print("Raw MAE by joint (rad):", np.round(raw_mae, 6))
    print("Raw MAE by horizon:", {step: round(value, 6) for step, value in horizon_errors.items()})
    print("PASS" if model_score < baseline_score else "FAIL: model does not beat hold-current baseline")


if __name__ == "__main__":
    main()
