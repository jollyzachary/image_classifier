from __future__ import annotations

import torch

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def select_device(requested: str) -> torch.device:
    """Resolve a requested compute device and fail clearly when unavailable."""

    if requested not in DEVICE_CHOICES:
        choices = ", ".join(DEVICE_CHOICES)
        raise ValueError(f"unsupported device {requested!r}; choose {choices}")

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but it is not available")
    return torch.device(requested)
