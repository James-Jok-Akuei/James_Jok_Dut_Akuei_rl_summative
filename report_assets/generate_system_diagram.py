from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, xy, text, width=0.24, height=0.14, facecolor="#f3efe7", edgecolor="#425466"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10, color="#1f2d3a")
    return box


def add_arrow(ax, start, end, text=None):
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=15, linewidth=1.5, color="#374151")
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.025, text, ha="center", va="bottom", fontsize=9, color="#4b5563")


def main() -> None:
    figure, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    figure.patch.set_facecolor("#fcfaf6")

    obs = add_box(ax, (0.05, 0.68), "Observation Vector\nPosition, battery,\nwind, distances, flags", facecolor="#d8ebf2")
    agent = add_box(ax, (0.38, 0.68), "RL Agent\nDQN / REINFORCE /\nPPO / A2C", facecolor="#f6e8c3")
    action = add_box(ax, (0.71, 0.68), "Action Selection\nMove, hover,\npickup, deliver,\nrecharge", facecolor="#f0d8cf")
    env = add_box(ax, (0.38, 0.37), "Custom Drone Environment", width=0.28, height=0.12, facecolor="#d9ead3")
    reward = add_box(ax, (0.05, 0.13), "Reward Signal\nProgress, pickup,\ndelivery, penalties", facecolor="#efe0f6")
    render = add_box(ax, (0.71, 0.13), "Renderer + JSON Export\nPygame GUI and\nserialized mission state", facecolor="#e8eef9")

    add_arrow(ax, (0.29, 0.75), (0.38, 0.75), "observations")
    add_arrow(ax, (0.62, 0.75), (0.71, 0.75), "actions")
    add_arrow(ax, (0.83, 0.68), (0.56, 0.49))
    add_arrow(ax, (0.49, 0.37), (0.17, 0.27), "rewards")
    add_arrow(ax, (0.17, 0.27), (0.45, 0.68))
    add_arrow(ax, (0.52, 0.49), (0.17, 0.68), "state")
    add_arrow(ax, (0.66, 0.43), (0.83, 0.27), "visualization")

    ax.text(
        0.5,
        0.95,
        "Mission-Based Reinforcement Learning System Diagram",
        ha="center",
        va="center",
        fontsize=16,
        color="#1f2d3a",
        weight="bold",
    )

    output_path = Path(__file__).resolve().parent / "system_diagram.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved diagram to {output_path}")


if __name__ == "__main__":
    main()
