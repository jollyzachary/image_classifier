from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import zero_shot


def test_labels_are_trimmed_and_deduplicated() -> None:
    assert zero_shot.normalize_labels([" cat ", "DOG", "Cat", "dog house"]) == [
        "cat",
        "DOG",
        "dog house",
    ]


def test_at_least_two_distinct_labels_are_required() -> None:
    with pytest.raises(ValueError, match="at least two"):
        zero_shot.normalize_labels(["cat", " CAT "])


def test_classification_normalizes_pipeline_results(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 24), "navy").save(image_path)

    def fake_classifier(
        image: Image.Image, **kwargs: object
    ) -> list[dict[str, object]]:
        assert image.mode == "RGB"
        assert kwargs["candidate_labels"] == ["cat", "dog", "car"]
        return [
            {"label": "dog", "score": 0.8},
            {"label": "cat", "score": 0.15},
            {"label": "car", "score": 0.05},
        ]

    assert zero_shot.classify_image(
        image_path,
        ["cat", "dog", "car"],
        fake_classifier,
        top_k=2,
    ) == [
        {"label": "dog", "score": 0.8},
        {"label": "cat", "score": 0.15},
    ]


def test_zero_shot_defaults_are_stable() -> None:
    arguments = zero_shot.build_parser().parse_args(
        ["image.jpg", "--labels", "cat", "dog"]
    )

    assert arguments.model == zero_shot.DEFAULT_MODEL
    assert arguments.device == "auto"
    assert arguments.top_k == 5
    assert arguments.output is None
    assert arguments.output_dir is None
