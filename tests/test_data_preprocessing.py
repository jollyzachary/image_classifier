from pathlib import Path

import pytest
from PIL import Image

from data_preprocessing import load_and_preprocess


def save_sample(root: Path, split: str, class_name: str) -> None:
    class_directory = root / split / class_name
    class_directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(80, 120, 160)).save(
        class_directory / "sample.png"
    )


def test_load_and_preprocess_requires_matching_classes(tmp_path: Path) -> None:
    for split in ("train", "valid"):
        for class_name in ("a", "b"):
            save_sample(tmp_path, split, class_name)
    for class_name in ("a", "c"):
        save_sample(tmp_path, "test", class_name)

    with pytest.raises(ValueError, match="test classes must match"):
        load_and_preprocess(tmp_path, batch_size=1)


def test_load_and_preprocess_validates_loader_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        load_and_preprocess(tmp_path, batch_size=0)
    with pytest.raises(ValueError, match="num_workers"):
        load_and_preprocess(tmp_path, num_workers=-1)
