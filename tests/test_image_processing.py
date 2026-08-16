from pathlib import Path

import torch
import pytest
from PIL import Image

from image_processing import process_image


def test_process_image_returns_normalized_model_input(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (400, 300), color=(120, 80, 40)).save(image_path)

    tensor = process_image(image_path)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_process_image_rejects_missing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError, match=str(missing_path)):
        process_image(missing_path)
