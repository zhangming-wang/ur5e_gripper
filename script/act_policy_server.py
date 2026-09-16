#!/usr/bin/env python3
"""Serve the trained ACT policy over a small localhost HTTP API.

This process intentionally has no ROS imports.  LeRobot currently requires a
different Python runtime from both the system ROS installation and IsaacLab,
so observations cross the boundary as JSON plus a base64 JPEG.

Endpoints:
  GET  /health
  POST /reset
  POST /infer_chunk  {"image": "<base64 JPEG>", "state": [7 floats]}

``/infer_chunk`` returns the full ``chunk_size``-step action chunk in one
network forward pass.  The adapter executes those actions locally and requests
the next chunk in the background before the buffer drains, so the ~200 ms
inference latency never stalls motion.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.policies import prepare_observation_for_inference
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


IMAGE_SIZE = (256, 256)
STATE_DIM = 7


class PolicyRuntime:
    def __init__(self, checkpoint: Path, device: str):
        self.checkpoint = checkpoint
        self.device = device
        self._lock = threading.Lock()

        config = ACTConfig.from_pretrained(checkpoint)
        config.device = device
        # The checkpoint contains the backbone weights.  Avoid an unnecessary
        # torchvision download before those weights are loaded.
        config.pretrained_backbone_weights = None
        self.policy = ACTPolicy.from_pretrained(checkpoint, config=config, strict=True)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.chunk_size = int(config.chunk_size)
        self.policy.reset()

    def reset(self) -> None:
        with self._lock:
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
        if image.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
            raise ValueError(f"unexpected image shape after resize: {image.shape}")
        return np.ascontiguousarray(image)

    def infer_chunk(self, encoded_image: str, state_values: Any) -> tuple[list[list[float]], float]:
        state = np.asarray(state_values, dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"state must be {STATE_DIM} finite values, got shape {state.shape}")
        image = self._decode_image(encoded_image)

        with self._lock:
            started = time.perf_counter()
            observation = {
                "observation.images.cam_high": image,
                "observation.state": state,
            }
            observation = prepare_observation_for_inference(
                observation, torch.device(self.device)
            )
            batch = self.preprocessor(observation)
            with torch.inference_mode():
                normalized = self.policy.predict_action_chunk(batch)
                prediction = self.postprocessor(normalized.clone())
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0

        chunk = prediction.squeeze(0).detach().cpu().numpy()
        if chunk.shape != (self.chunk_size, STATE_DIM) or not np.isfinite(chunk).all():
            raise ValueError(f"policy returned invalid chunk shape/value: {chunk.shape}")
        return chunk.astype(np.float64).tolist(), elapsed_ms


class RequestHandler(BaseHTTPRequestHandler):
    runtime: PolicyRuntime

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the chunk-request stream out of the terminal. Errors are returned
        # to the client and logged by the adapter.
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
        self._send_json(
            200,
            {
                "ok": True,
                "policy": "act",
                "device": self.runtime.device,
                "chunk_size": self.runtime.chunk_size,
            },
        )

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
            chunk, latency_ms = self.runtime.infer_chunk(
                encoded_image=payload["image"],
                state_values=payload["state"],
            )
            self._send_json(200, {"ok": True, "chunk": chunk, "latency_ms": latency_ms})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive process boundary
            self._send_json(500, {"ok": False, "error": f"inference failed: {exc}"})


def _resolve_checkpoint(path: Path) -> Path:
    candidates = [path, path / "pretrained_model", path / "checkpoints/last/pretrained_model"]
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found under {path}; expected config.json and model.safetensors"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=27655)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    checkpoint = _resolve_checkpoint(args.checkpoint)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[ACT] Loading checkpoint: {checkpoint}", flush=True)
    runtime = PolicyRuntime(checkpoint, device)
    handler = type("ACTRequestHandler", (RequestHandler,), {"runtime": runtime})
    server = HTTPServer((args.host, args.port), handler)
    print(
        f"[ACT] Ready on http://{args.host}:{args.port} "
        f"(device={device}, chunk_size={runtime.chunk_size})",
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
