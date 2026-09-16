#!/usr/bin/env python3
"""Evaluate a trained ACT checkpoint on held-out dataset episodes.

The script compares ACT against a representation-aware constant baseline:
absolute actions hold the current joint state, while delta actions predict zero.
It also measures whether predictions vary across episodes that start from nearly
the same robot state but show the cube at different image locations.

Run in the LeRobot environment:
  conda activate lerobot
  python test/act_smoke_test.py \
    --checkpoint outputs/ur5e_act_rgb_abs_novae/checkpoints/last/pretrained_model \
    --dataset-root lerobot_data_rgb_abs
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import resolve_episode_indices
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import NormalizerProcessorStep, RenameObservationsProcessorStep
from lerobot.processor.rename_processor import rename_batch_keys

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_DIR / "outputs/ur5e_act_rgb_abs_novae/checkpoints/last/pretrained_model"
)
DEFAULT_DATASET_ROOT = PROJECT_DIR / "lerobot_data_rgb_abs"
DEFAULT_REPO_ID = "ur5e_pick_place"


def _resolve_checkpoint(path: Path) -> Path:
    candidates = [
        path,
        path / "pretrained_model",
        path / "checkpoints/last/pretrained_model",
    ]
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(
        f"No pretrained model found under {path}. Expected config.json and model.safetensors."
    )


def _to_numpy(column) -> np.ndarray:
    return np.asarray(column)


def _detect_action_representation(
    dataset: LeRobotDataset, state_key: str
) -> tuple[str, float, float]:
    states = _to_numpy(dataset.hf_dataset[state_key]).astype(np.float64)
    actions = _to_numpy(dataset.hf_dataset["action"]).astype(np.float64)
    episode_indices = _to_numpy(dataset.hf_dataset["episode_index"]).reshape(-1)
    same_episode = episode_indices[1:] == episode_indices[:-1]

    current = states[:-1][same_episode]
    following = states[1:][same_episode]
    action = actions[:-1][same_episode]
    delta_error = float(np.mean(np.abs(action - (following - current))))
    absolute_error = float(np.mean(np.abs(action - following)))

    tolerance = 1e-5
    if absolute_error <= tolerance and absolute_error < delta_error:
        representation = "absolute"
    elif delta_error <= tolerance and delta_error < absolute_error:
        representation = "delta"
    else:
        raise ValueError(
            "Could not infer action representation: "
            f"absolute_next_mae={absolute_error:.6g}, delta_mae={delta_error:.6g}"
        )
    return representation, absolute_error, delta_error


def _normalize(values: np.ndarray, stats: dict, normalization_mode, eps: float) -> np.ndarray:
    mode = getattr(normalization_mode, "value", str(normalization_mode))
    values = values.astype(np.float64)
    if mode == "MEAN_STD":
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        return (values - mean) / (std + eps)
    elif mode == "MIN_MAX":
        lower = np.asarray(stats["min"], dtype=np.float64)
        upper = np.asarray(stats["max"], dtype=np.float64)
    elif mode == "QUANTILES":
        lower = np.asarray(stats["q01"], dtype=np.float64)
        upper = np.asarray(stats["q99"], dtype=np.float64)
    elif mode == "QUANTILE10":
        lower = np.asarray(stats["q10"], dtype=np.float64)
        upper = np.asarray(stats["q90"], dtype=np.float64)
    elif mode == "IDENTITY":
        return values
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    denominator = upper - lower
    denominator = np.where(denominator == 0, eps, denominator)
    return 2.0 * (values - lower) / denominator - 1.0


def _sample_indices(
    dataset: LeRobotDataset, episode_indices: list[int], chunk_size: int, samples_per_episode: int
) -> tuple[list[int], list[int]]:
    episodes = _to_numpy(dataset.hf_dataset["episode_index"]).reshape(-1)
    frame_indices = _to_numpy(dataset.hf_dataset["frame_index"]).reshape(-1)
    sample_indices = []
    initial_indices = []

    for episode_index in episode_indices:
        rows = np.flatnonzero(episodes == episode_index)
        if rows.size < chunk_size:
            raise ValueError(
                f"Episode {episode_index} has {rows.size} frames, shorter than chunk_size={chunk_size}"
            )
        initial_indices.append(int(rows[0]))
        last_start_frame = int(frame_indices[rows[-1]]) - chunk_size + 1
        valid = rows[frame_indices[rows] <= last_start_frame]
        positions = np.linspace(0, len(valid) - 1, num=min(samples_per_episode, len(valid)))
        sample_indices.extend(int(valid[position]) for position in np.unique(positions.astype(int)))

    return sample_indices, initial_indices


def _resolve_eval_episodes(
    checkpoint: Path,
    metadata: LeRobotDatasetMetadata,
    requested: int | None,
) -> tuple[list[int], str]:
    if requested is not None:
        count = min(requested, metadata.total_episodes)
        episodes = list(range(metadata.total_episodes - count, metadata.total_episodes))
        return episodes, "explicit final episodes"

    train_config_path = checkpoint / "train_config.json"
    if not train_config_path.is_file():
        raise ValueError(
            f"Cannot infer held-out episodes because {train_config_path} is missing; "
            "pass --eval-episodes only if those final episodes were excluded from training"
        )

    with train_config_path.open() as file:
        train_config = json.load(file)
    dataset_config = train_config.get("dataset", {})
    configured_repo = dataset_config.get("repo_id")
    if configured_repo and configured_repo != metadata.repo_id:
        raise ValueError(
            f"Checkpoint was trained on repo_id={configured_repo}, not {metadata.repo_id}"
        )

    eval_split = float(dataset_config.get("eval_split", 0.0))
    if not 0.0 < eval_split < 1.0:
        raise ValueError(
            f"Checkpoint has dataset.eval_split={eval_split}; no held-out split can be inferred"
        )

    base_episodes = resolve_episode_indices(
        dataset_config.get("episodes"),
        metadata.total_episodes,
        dataset_config.get("exclude_episodes"),
    )
    if base_episodes is None:
        base_episodes = list(range(metadata.total_episodes))

    task_to_episodes: dict[str, list[int]] = {}
    episode_tasks = metadata.episodes["tasks"]
    for episode_index in base_episodes:
        task_key = episode_tasks[episode_index][0] if episode_tasks[episode_index] else ""
        task_to_episodes.setdefault(task_key, []).append(episode_index)

    eval_episodes = []
    for episodes in task_to_episodes.values():
        eval_count = math.ceil(len(episodes) * eval_split)
        eval_episodes.extend(episodes[-eval_count:])
    if not eval_episodes:
        raise ValueError("Checkpoint train_config produced an empty evaluation split")

    return eval_episodes, f"train_config eval_split={eval_split:g}"


def _predict_chunk(
    sample: dict,
    rename_map: dict[str, str],
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
) -> tuple[np.ndarray, np.ndarray]:
    renamed_sample = rename_batch_keys(sample, rename_map)
    observation = {
        key: value for key, value in renamed_sample.items() if key.startswith("observation.")
    }
    if "task" in sample:
        observation["task"] = sample["task"]
    batch = preprocessor(observation)
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(batch)
        prediction = postprocessor(normalized.clone())
    return (
        normalized.squeeze(0).detach().cpu().numpy(),
        prediction.squeeze(0).detach().cpu().numpy(),
    )


def _evaluate_indices(
    dataset: LeRobotDataset,
    indices: list[int],
    representation: str,
    state_key: str,
    rename_map: dict[str, str],
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalized_predictions = []
    predictions = []
    targets = []
    baselines = []
    states = []

    for count, index in enumerate(indices, start=1):
        sample = dataset[index]
        normalized_prediction, prediction = _predict_chunk(
            sample, rename_map, policy, preprocessor, postprocessor
        )
        target = sample["action"].cpu().numpy()
        if sample["action_is_pad"].any():
            raise ValueError(f"Unexpected padded action chunk at dataset index {index}")
        if prediction.shape != target.shape or normalized_prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shapes raw={prediction.shape}, normalized={normalized_prediction.shape} "
                f"do not match target shape {target.shape}"
            )
        if not np.isfinite(prediction).all() or not np.isfinite(normalized_prediction).all():
            raise ValueError(f"Non-finite model prediction at dataset index {index}")

        state = sample[state_key].cpu().numpy()
        baseline = np.zeros_like(target)
        if representation == "absolute":
            baseline[:] = state

        normalized_predictions.append(normalized_prediction)
        predictions.append(prediction)
        targets.append(target)
        baselines.append(baseline)
        states.append(state)
        print(f"  evaluated {count:03d}/{len(indices):03d}", end="\r", flush=True)

    print(" " * 40, end="\r")
    return tuple(
        np.stack(values)
        for values in (normalized_predictions, predictions, targets, baselines, states)
    )


def _print_metrics(
    normalized_predictions: np.ndarray,
    predictions: np.ndarray,
    normalized_targets: np.ndarray,
    targets: np.ndarray,
    normalized_baselines: np.ndarray,
) -> tuple[float, float]:
    model_error = np.abs(normalized_predictions - normalized_targets)
    baseline_error = np.abs(normalized_baselines - normalized_targets)
    model_per_joint = np.mean(model_error, axis=(0, 1))
    baseline_per_joint = np.mean(baseline_error, axis=(0, 1))
    model_score = float(model_per_joint.mean())
    baseline_score = float(baseline_per_joint.mean())
    if not np.isfinite(model_score) or not np.isfinite(baseline_score):
        raise ValueError(
            f"Non-finite metric: model={model_score}, baseline={baseline_score}"
        )

    print("Normalized L1 by joint:")
    print("  model:   ", np.round(model_per_joint, 4))
    print("  baseline:", np.round(baseline_per_joint, 4))
    print(f"Overall normalized L1: model={model_score:.4f}, baseline={baseline_score:.4f}")
    print(f"Model/baseline ratio: {model_score / max(baseline_score, 1e-12):.3f}")
    raw_error = np.abs(predictions - targets)
    print("Raw MAE by joint (rad):", np.round(raw_error.mean(axis=(0, 1)), 6))

    horizons = sorted(
        {
            0,
            normalized_predictions.shape[1] // 4,
            normalized_predictions.shape[1] // 2,
            normalized_predictions.shape[1] - 1,
        }
    )
    horizon_scores = [float(np.mean(model_error[:, horizon])) for horizon in horizons]
    rounded_scores = [round(score, 4) for score in horizon_scores]
    print("Normalized L1 by horizon:", dict(zip(horizons, rounded_scores, strict=True)))
    return model_score, baseline_score


def _print_initial_conditioning(
    normalized_predictions: np.ndarray,
    normalized_targets: np.ndarray,
    states: np.ndarray,
    normalized_states: np.ndarray,
) -> tuple[float, bool]:
    if len(normalized_predictions) < 2:
        print("Initial-scene prediction variation requires at least 2 held-out episodes.")
        return float("nan"), False

    prediction_variation = float(np.mean(np.std(normalized_predictions, axis=0)))
    target_variation = float(np.mean(np.std(normalized_targets, axis=0)))
    ratio = (
        prediction_variation / target_variation
        if target_variation > 1e-8
        else float("nan")
    )
    raw_state_spread = np.ptp(states, axis=0)
    normalized_state_spread = np.ptp(normalized_states, axis=0)
    state_is_controlled = bool(np.max(normalized_state_spread) <= 0.05)
    print("Initial-state spread by joint (rad):", np.round(raw_state_spread, 6))
    print(
        "Initial-state max normalized spread: "
        f"{np.max(normalized_state_spread):.4f} "
        f"({'controlled' if state_is_controlled else 'too large to isolate RGB conditioning'})"
    )
    print(
        "Initial-scene prediction variation: "
        f"model={prediction_variation:.4f}, target={target_variation:.4f}, ratio={ratio:.3f}"
    )
    return ratio, state_is_controlled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Number of final episodes to hold out; defaults to the checkpoint train_config split",
    )
    parser.add_argument("--samples-per-episode", type=int, default=6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--video-backend", default="pyav")
    args = parser.parse_args()

    if args.eval_episodes is not None and args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    if args.samples_per_episode <= 0:
        parser.error("--samples-per-episode must be positive")
    if not args.dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {args.dataset_root}")

    checkpoint = _resolve_checkpoint(args.checkpoint)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = ACTConfig.from_pretrained(checkpoint)
    config.device = device

    print(f"Checkpoint: {checkpoint}")
    print(
        f"ACT config: device={device}, chunk_size={config.chunk_size}, "
        f"n_action_steps={config.n_action_steps}, use_vae={config.use_vae}, "
        f"kl_weight={config.kl_weight}"
    )
    policy = ACTPolicy.from_pretrained(checkpoint, config=config, strict=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    held_out_episodes, split_source = _resolve_eval_episodes(
        checkpoint, metadata, args.eval_episodes
    )
    delta_timestamps = {
        "action": [index / metadata.fps for index in range(config.chunk_size)]
    }
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=held_out_episodes,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )

    rename_step = next(
        (step for step in preprocessor.steps if isinstance(step, RenameObservationsProcessorStep)),
        None,
    )
    rename_map = rename_step.rename_map if rename_step is not None else {}
    state_key = next(
        (source for source, target in rename_map.items() if target == "observation.state"),
        "observation.state",
    )
    if state_key not in dataset.hf_dataset.column_names:
        raise ValueError(f"Dataset does not contain the checkpoint state feature: {state_key}")

    representation, absolute_error, delta_error = _detect_action_representation(dataset, state_key)
    print(
        f"Dataset: episodes={metadata.total_episodes}, frames={metadata.total_frames}, "
        f"held_out={held_out_episodes} ({split_source}), action={representation}"
    )
    print(
        f"Action representation check: absolute_next_mae={absolute_error:.3g}, "
        f"delta_mae={delta_error:.3g}"
    )

    sample_indices, initial_indices = _sample_indices(
        dataset,
        held_out_episodes,
        chunk_size=config.chunk_size,
        samples_per_episode=args.samples_per_episode,
    )
    normalizer = next(
        (step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)),
        None,
    )
    if normalizer is None or "action" not in normalizer.stats:
        raise ValueError("Checkpoint preprocessor does not contain action normalization statistics")
    if "observation.state" not in normalizer.stats:
        raise ValueError("Checkpoint preprocessor does not contain state normalization statistics")
    action_mode = config.normalization_mapping["ACTION"]
    state_mode = config.normalization_mapping["STATE"]

    print(f"Evaluating {len(sample_indices)} uniformly spaced chunks...")
    normalized_predictions, predictions, targets, baselines, _ = _evaluate_indices(
        dataset,
        sample_indices,
        representation,
        state_key,
        rename_map,
        policy,
        preprocessor,
        postprocessor,
    )
    normalized_targets = _normalize(targets, normalizer.stats["action"], action_mode, normalizer.eps)
    normalized_baselines = _normalize(
        baselines, normalizer.stats["action"], action_mode, normalizer.eps
    )
    model_score, baseline_score = _print_metrics(
        normalized_predictions,
        predictions,
        normalized_targets,
        targets,
        normalized_baselines,
    )

    print(f"Evaluating {len(initial_indices)} initial scenes...")
    (
        initial_normalized_predictions,
        _,
        initial_targets,
        _,
        initial_states,
    ) = _evaluate_indices(
        dataset,
        initial_indices,
        representation,
        state_key,
        rename_map,
        policy,
        preprocessor,
        postprocessor,
    )
    initial_normalized_targets = _normalize(
        initial_targets, normalizer.stats["action"], action_mode, normalizer.eps
    )
    initial_normalized_states = _normalize(
        initial_states, normalizer.stats["observation.state"], state_mode, normalizer.eps
    )
    variation_ratio, state_is_controlled = _print_initial_conditioning(
        initial_normalized_predictions,
        initial_normalized_targets,
        initial_states,
        initial_normalized_states,
    )

    print("Verdict:")
    if model_score >= baseline_score:
        print("  FAIL: model does not beat the representation-aware constant baseline.")
    elif not state_is_controlled:
        print("  WARN: model beats the baseline, but initial states differ too much to isolate RGB use.")
    elif not np.isfinite(variation_ratio):
        print("  WARN: model beats the baseline, but initial-scene variation could not be measured.")
    elif np.isfinite(variation_ratio) and variation_ratio < 0.1:
        print("  WARN: model beats the baseline but predictions barely change across initial scenes.")
    else:
        print("  PASS: model beats the baseline and predictions respond to initial-scene variation.")


if __name__ == "__main__":
    main()
