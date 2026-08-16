from pathlib import Path

import pytest
import torch

from checkpoint import load_checkpoint


def test_load_checkpoint_rejects_unknown_version(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"checkpoint_version": 999}, checkpoint_path)

    with pytest.raises(ValueError, match="unsupported checkpoint version"):
        load_checkpoint(checkpoint_path, torch.device("cpu"))
