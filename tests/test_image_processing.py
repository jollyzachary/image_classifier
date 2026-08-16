from pathlib import Path

import pytest
import torch
from PIL import Image

from image_processing import display_predictions, process_image


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


def test_display_predictions_saves_visualization(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    output_path = tmp_path / "prediction.png"
    Image.new("RGB", (400, 300), color=(220, 170, 30)).save(image_path)

    display_predictions(
        image_path,
        [0.8, 0.15, 0.05],
        ["sunflower", "water lily", "passion flower"],
        output_path=output_path,
        show=False,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
