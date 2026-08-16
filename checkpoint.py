from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from model import build_model

CHECKPOINT_VERSION = 1


def save_checkpoint(
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    class_to_idx: dict[str, int],
    destination: str | Path,
    *,
    architecture: str,
    hidden_units: int,
    output_size: int,
    epochs: int,
    learning_rate: float,
) -> Path:
    """Save model state and the metadata required to rebuild it."""

    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": architecture,
        "hidden_units": hidden_units,
        "output_size": output_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "class_to_idx": class_to_idx,
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    source: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a model from a checkpoint created by :func:`save_checkpoint`."""

    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a metadata dictionary")

    version = checkpoint.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {version!r}; expected {CHECKPOINT_VERSION}"
        )
    required = {
        "architecture",
        "hidden_units",
        "output_size",
        "class_to_idx",
        "model_state_dict",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {', '.join(missing)}")

    class_to_idx = checkpoint["class_to_idx"]
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("checkpoint class_to_idx must be a non-empty dictionary")

    network = build_model(
        checkpoint["architecture"],
        hidden_units=int(checkpoint["hidden_units"]),
        output_size=int(checkpoint["output_size"]),
        pretrained=False,
    )
    network.load_state_dict(checkpoint["model_state_dict"])
    network.class_to_idx = {
        str(label): int(index) for label, index in class_to_idx.items()
    }
    network.to(device)
    network.eval()
    return network, checkpoint
