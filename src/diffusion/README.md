# Diffusion Policy Mode

Run the trained Diffusion Policy directly in IsaacLab:

```bash
./run.sh --diffusion
```

This mode is independent from MoveIt and ACT. It starts a LeRobot inference
server, IsaacLab direct control, a Diffusion adapter, a Diffusion orchestrator,
and the common control panel.

The policy receives two synchronized RGB and seven-joint observations. The
adapter sends the latest adjacent observation pair to the server, which returns
32 absolute joint targets per request. The adapter publishes them at 15 Hz and
requests the next chunk after the current 32 actions finish, so inference uses
the latest observation pair rather than a stale prefetched observation. The
first four arm targets of each new chunk are blended from the previous target.
The gripper requires three consistent predictions to change state and follows a
single `open -> closed -> released` sequence per cycle.

Topics specific to this mode:

```text
/isaaclab/diffusion/enabled
/isaaclab/diffusion/joint_target
/diffusion_ros_adapter/prepare
```

The panel continues to send the shared `/pick_and_place` action. Do not run
`--act` and `--diffusion` together because both provide that action server.

Useful overrides:

```bash
DIFFUSION_INFERENCE_STEPS=100 ./run.sh --diffusion  # Full training schedule
DIFFUSION_PREFETCH_THRESHOLD=8 ./run.sh --diffusion  # Experimental async prefetch
DIFFUSION_BLEND_STEPS=6 ./run.sh --diffusion
DIFFUSION_GRIPPER_CONFIRM_STEPS=4 ./run.sh --diffusion
```

The runtime defaults to 20 denoising steps. This checkpoint passed the offline
baseline test and a complete IsaacLab cycle at 20 steps, while reducing the
hold between 32-action chunks. The full 100-step schedule remains available
for comparison.

Evaluate the checkpoint offline in the LeRobot environment:

```bash
conda activate lerobot
python test/diffusion_smoke_test.py \
  --checkpoint outputs/ur5e_diffusion_abs/checkpoints/030000/pretrained_model
```
