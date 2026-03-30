from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class ReinforcePolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def save_reinforce_checkpoint(
    path: Path,
    policy: ReinforcePolicy,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "metadata": metadata,
        },
        path,
    )


def load_reinforce_checkpoint(path: Path, device: str = "cpu") -> tuple[ReinforcePolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy = ReinforcePolicy(
        observation_dim=checkpoint["observation_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.to(device)
    policy.eval()
    return policy, checkpoint.get("metadata", {})
