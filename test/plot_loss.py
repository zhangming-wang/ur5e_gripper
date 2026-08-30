#!/usr/bin/env python3
"""训练 loss 曲线绘制 — 读 lerobot-train 的 tee 日志画 loss 图

用法 (lerobot conda 环境):
  python test/plot_loss.py [日志路径] [输出图片路径]
  默认: outputs/ur5e_act_kl1_train.log → outputs/loss_curve.png
"""

import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_LOG = "/home/dev/work/ur5e_gripper/outputs/ur5e_act_kl1_train.log"
DEFAULT_OUT = "/home/dev/work/ur5e_gripper/outputs/loss_curve.png"

# 匹配两种常见格式: "step 100: loss=0.123" 或 "[100/10000] loss: 0.123"
PATTERNS = [
    re.compile(r"step\s+(\d+)[^\d]*loss[=:\s]+([0-9.eE+-]+)"),
    re.compile(r"\[(\d+)/\d+\].*?loss[:=\s]+([0-9.eE+-]+)"),
]


def parse_loss(log_path):
    steps, losses = [], []
    with open(log_path) as f:
        for line in f:
            for pat in PATTERNS:
                m = pat.search(line)
                if m:
                    steps.append(int(m.group(1)))
                    losses.append(float(m.group(2)))
                    break
    return steps, losses


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

    steps, losses = parse_loss(log_path)
    if not losses:
        print(f"没在 {log_path} 里找到 loss 行 (共匹配 {len(losses)} 个)")
        sys.exit(1)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, linewidth=1.2)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"ACT training loss ({len(losses)} points)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)

    print(f"曲线已保存: {out_path}")
    print(f"loss: 首={losses[0]:.4f} 末={losses[-1]:.4f} "
          f"min={min(losses):.4f} @step {steps[losses.index(min(losses))]}")
    if losses[-1] > min(losses) * 1.15:
        print("⚠️ 末段 loss 高于最小值 15%+ — 可能开始过拟合")
    else:
        print("✅ loss 下降后保持稳定")


if __name__ == "__main__":
    main()
