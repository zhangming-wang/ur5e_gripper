#!/usr/bin/env python3
"""ACT 冒烟测试 — 加载训练好的模型，对比预测动作 vs 专家动作

用法 (lerobot conda 环境):
  python test/act_smoke_test.py

判定:
  平均误差 < 0.1 rad 且方向一致  → 模型健康，可进闭环评测
  预测恒值/方向随机             → 训练失败，排查数据/归一化
"""

import sys

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

CKPT = "/home/dev/work/ur5e_gripper/outputs/ur5e_act_kl1/checkpoints/last/pretrained_model"
DATASET_ROOT = "/home/dev/work/ur5e_gripper/lerobot_data_rgb"
REPO_ID = "ur5e_pick_place"
FPS = 15
N_SAMPLES = 8


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    cfg = ACTConfig.from_pretrained(CKPT)
    policy = ACTPolicy(cfg).to(device).eval()
    print(f"模型加载: {CKPT}")
    print(f"chunk_size={cfg.chunk_size}, n_action_steps={cfg.n_action_steps}")

    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=CKPT)

    delta_timestamps = {"action": [i / FPS for i in range(cfg.chunk_size)]}
    ds = LeRobotDataset(REPO_ID, root=DATASET_ROOT, delta_timestamps=delta_timestamps)

    # 每集取"动作最大"的帧（机械臂在动的时刻），保证 chunk 不越集边界
    ep_idx = np.array(ds.hf_dataset["episode_index"])
    actions_all = np.array(ds.hf_dataset["action"])  # (num_frames, 7)
    norms = np.linalg.norm(actions_all, axis=1)
    active = []
    for ep in range(ds.num_episodes):
        frames = np.where(ep_idx == ep)[0]
        valid = frames[frames <= frames[-1] - cfg.chunk_size - 5]
        if len(valid) == 0:
            continue
        m = int(valid[np.argmax(norms[valid])])
        active.append(m)
    sample_idxs = np.array(active[:N_SAMPLES])
    print(f"采样帧 (每集动作最大处): {sample_idxs.tolist()}")

    errors_first = []
    errors_chunk = []
    preds = []
    gts = []

    for i in sample_idxs:
        sample = ds[int(i)]
        batch = {
            "observation.images.cam_high": sample["observation.images.cam_high"].unsqueeze(0),
            "observation.state": sample["observation.state"].unsqueeze(0),
            "action": sample["action"].unsqueeze(0),
        }
        batch = preprocessor(batch)
        obs = {
            k: batch[k].to(device)
            for k in ("observation.images.cam_high", "observation.state")
        }

        with torch.inference_mode():
            pred = policy.predict_action_chunk(obs).cpu()

        pred_rad = postprocessor(pred)
        if hasattr(pred_rad, "action"):
            pred_rad = pred_rad.action
        pred_rad = pred_rad[0].numpy()  # (chunk, 7)

        gt = sample["action"].numpy()  # (chunk, 7) 原始弧度

        err = np.abs(pred_rad - gt)
        errors_first.append(err[0].mean())
        errors_chunk.append(err[:20].mean())
        preds.append(pred_rad[0])
        gts.append(gt[0])

        print(
            f"frame {i:6d}: 首步误差={err[0].mean():.4f} rad | "
            f"前20步误差={err[:20].mean():.4f} rad | "
            f"pred0={np.round(pred_rad[0], 3)} gt0={np.round(gt[0], 3)}"
        )

    mf = float(np.mean(errors_first))
    mc = float(np.mean(errors_chunk))
    pred_std = float(np.std(np.array(preds), axis=0).mean())
    gt_std = float(np.std(np.array(gts), axis=0).mean())
    print("\n" + "=" * 60)
    print(f"平均首步误差: {mf:.4f} rad")
    print(f"平均前20步误差: {mc:.4f} rad")
    print(f"预测跨样本标准差: {pred_std:.4f} | 专家跨样本标准差: {gt_std:.4f}")
    if pred_std < 0.2 * gt_std:
        print("❌ 模式坍塌 — 模型输出接近常数，不随输入变化 (预测std << 专家std)")
        print("   对策: 降 kl_weight / 加数据 / 调 lr 后重训")
    elif mf < 0.1:
        print("✅ 模型健康 — 预测与专家动作同量级，可进闭环评测")
    elif mf < 0.3:
        print("⚠️ 半成品 — 方向对但幅度偏差，考虑补数据或多训")
    else:
        print("❌ 训练失败 — 检查数据对齐/归一化/训练步数")


if __name__ == "__main__":
    main()
