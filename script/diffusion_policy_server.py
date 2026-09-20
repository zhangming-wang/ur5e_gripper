#!/usr/bin/env python3
"""Serve a trained Diffusion Policy over localhost HTTP without ROS imports."""

from __future__ import annotations

import argparse
import base64
import collections
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.policies import make_pre_post_processors, prepare_observation_for_inference
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


IMAGE_KEY = "observation.images.cam_high"
STATE_KEY = "observation.state"
IMAGE_SIZE = (256, 256)
STATE_DIM = 7


class PolicyRuntime:
    def __init__(self, checkpoint: Path, device: str, num_inference_steps: int | None):
        self.device = device
        self._lock = threading.Lock()
        self.config = DiffusionConfig.from_pretrained(checkpoint)
        self.config.device = device
        self.config.pretrained_backbone_weights = None
        if num_inference_steps is not None:
            self.config.num_inference_steps = num_inference_steps
        self.policy = DiffusionPolicy.from_pretrained(checkpoint, config=self.config, strict=True)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self._images: collections.deque[np.ndarray] = collections.deque(maxlen=self.config.n_obs_steps)
        self._states: collections.deque[np.ndarray] = collections.deque(maxlen=self.config.n_obs_steps)
        self.policy.reset()

    def reset(self) -> None:
        with self._lock:
            self._images.clear()
            self._states.clear()
            self.policy.reset()

    @staticmethod
    def _decode_image(encoded: str) -> np.ndarray:
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("image is not valid base64") from exc
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("image JPEG could not be decoded")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image)

    def _append_observation(self, image: np.ndarray, state: np.ndarray) -> None:
        if not self._images:
            for _ in range(self.config.n_obs_steps):
                self._images.append(image.copy())
                self._states.append(state.copy())
        else:
            self._images.append(image)
            self._states.append(state)

    def _predict_history(self) -> tuple[list[list[float]], float]:
        if len(self._images) != self.config.n_obs_steps or len(self._states) != self.config.n_obs_steps:
            raise ValueError("observation history is incomplete")
        frames = []
        for history_image, history_state in zip(self._images, self._states):
            frames.append(
                self.preprocessor(
                    prepare_observation_for_inference(
                        {IMAGE_KEY: history_image, STATE_KEY: history_state}, torch.device(self.device)
                    )
                )
            )
        batch = {
            IMAGE_KEY: torch.stack([frame[IMAGE_KEY] for frame in frames], dim=1),
            STATE_KEY: torch.stack([frame[STATE_KEY] for frame in frames], dim=1),
        }
        started = time.perf_counter()
        with torch.inference_mode():
            normalized = self.policy.predict_action_chunk(batch)
            prediction = self.postprocessor(normalized.clone())
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        chunk = prediction.squeeze(0).detach().cpu().numpy()
        expected = (self.config.n_action_steps, STATE_DIM)
        if chunk.shape != expected or not np.isfinite(chunk).all():
            raise ValueError(f"policy returned invalid chunk shape/value: {chunk.shape}, expected {expected}")
        return chunk.astype(np.float64).tolist(), elapsed_ms

    def infer_chunk(self, encoded_image: str, state_values: Any) -> tuple[list[list[float]], float]:
        state = np.asarray(state_values, dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"state must be {STATE_DIM} finite values, got shape {state.shape}")
        image = self._decode_image(encoded_image)
        with self._lock:
            self._append_observation(image, state)
            return self._predict_history()

    def infer_history(self, encoded_images: Any, state_sequences: Any) -> tuple[list[list[float]], float]:
        if not isinstance(encoded_images, list) or not isinstance(state_sequences, list):
            raise ValueError("images and states must be lists")
        if len(encoded_images) != self.config.n_obs_steps or len(state_sequences) != self.config.n_obs_steps:
            raise ValueError(f"expected {self.config.n_obs_steps} observations")
        images = [self._decode_image(encoded) for encoded in encoded_images]
        states = [np.asarray(values, dtype=np.float32) for values in state_sequences]
        if any(state.shape != (STATE_DIM,) or not np.isfinite(state).all() for state in states):
            raise ValueError(f"each state must be {STATE_DIM} finite values")
        with self._lock:
            self._images.clear()
            self._states.clear()
            self._images.extend(images)
            self._states.extend(states)
            return self._predict_history()


class RequestHandler(BaseHTTPRequestHandler):
    runtime: PolicyRuntime

    def log_message(self, format: str, *args: Any) -> None:
        if self.path != "/infer_chunk":
            super().log_message(format, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        self._send_json(200, {
            "ok": True,
            "policy": "diffusion",
            "device": self.runtime.device,
            "n_obs_steps": self.runtime.config.n_obs_steps,
            "n_action_steps": self.runtime.config.n_action_steps,
            "num_inference_steps": self.runtime.policy.diffusion.num_inference_steps,
        })

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/infer_chunk", "/reset"):
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            if self.path == "/reset":
                self.runtime.reset()
                self._send_json(200, {"ok": True})
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 8 * 1024 * 1024:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(content_length))
            if "images" in payload or "states" in payload:
                chunk, latency_ms = self.runtime.infer_history(payload["images"], payload["states"])
            else:
                chunk, latency_ms = self.runtime.infer_chunk(payload["image"], payload["state"])
            self._send_json(200, {"ok": True, "chunk": chunk, "latency_ms": latency_ms})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:  # pragma: no cover - process boundary
            self._send_json(500, {"ok": False, "error": f"inference failed: {exc}"})


def _resolve_checkpoint(path: Path) -> Path:
    for candidate in (path, path / "pretrained_model", path / "checkpoints/last/pretrained_model"):
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {path}; expected config.json and model.safetensors")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=27658)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.num_inference_steps is not None and not 1 <= args.num_inference_steps <= 100:
        parser.error("--num-inference-steps must be between 1 and 100")
    checkpoint = _resolve_checkpoint(args.checkpoint)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Diffusion] Loading checkpoint: {checkpoint}", flush=True)
    runtime = PolicyRuntime(checkpoint, device, args.num_inference_steps)
    handler = type("DiffusionRequestHandler", (RequestHandler,), {"runtime": runtime})
    server = HTTPServer((args.host, args.port), handler)
    print(
        f"[Diffusion] Ready on http://{args.host}:{args.port} "
        f"(device={device}, obs={runtime.config.n_obs_steps}, actions={runtime.config.n_action_steps}, "
        f"steps={runtime.policy.diffusion.num_inference_steps})",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
