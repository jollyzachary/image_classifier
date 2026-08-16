from __future__ import annotations

from pathlib import Path

from PIL import Image

import classify_any


def test_objects_are_sorted_by_visible_area() -> None:
    payload = {
        classify_any.OBJECT_TASK: {
            "bboxes": [[0, 0, 10, 10], [5, 5, 40, 30]],
            "labels": ["small object", "large object"],
        }
    }

    objects = classify_any.parse_objects(payload)

    assert [item["label"] for item in objects] == ["large object", "small object"]
    assert objects[0]["area"] == 875.0


def test_general_analysis_needs_no_candidate_labels(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (80, 60), "gold").save(image_path)

    def fake_runner(image: Image.Image, task: str) -> dict[str, object]:
        assert image.size == (80, 60)
        if task == classify_any.CAPTION_TASK:
            return {task: "A yellow sports car parked near a building."}
        return {
            task: {
                "bboxes": [[3, 4, 75, 55], [50, 0, 79, 30]],
                "labels": ["sports car", "building"],
            }
        }

    result = classify_any.analyze_image(image_path, fake_runner)

    assert result["primary_subject"] == "sports car"
    assert result["description"].startswith("A yellow sports car")
    assert [item["label"] for item in result["objects"]] == [
        "sports car",
        "building",
    ]


def test_caption_selects_specific_subject_over_largest_region() -> None:
    objects = [
        {"label": "plate", "area": 1000.0},
        {"label": "coffee cup", "area": 400.0},
    ]

    subject = classify_any.select_primary_subject(
        "A cup of coffee rests on a plate.", objects
    )

    assert subject == "coffee cup"


def test_classify_any_defaults_are_stable() -> None:
    arguments = classify_any.build_parser().parse_args(["image.jpg"])

    assert arguments.device == "auto"
    assert arguments.output is None
    assert arguments.output_dir is None
    assert arguments.json_output is None


def test_reusable_classifier_keeps_one_runner(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (20, 20), "green").save(first)
    Image.new("RGB", (20, 20), "blue").save(second)
    calls: list[str] = []

    def fake_runner(image: Image.Image, task: str) -> dict[str, object]:
        calls.append(task)
        if task == classify_any.CAPTION_TASK:
            return {task: "A colored square."}
        return {task: {"bboxes": [], "labels": []}}

    classifier = classify_any.AnyImageClassifier("cpu", runner=fake_runner)
    results = classifier.classify_many([first, second])

    assert [result["primary_subject"] for result in results] == ["scene", "scene"]
    assert calls == [
        classify_any.CAPTION_TASK,
        classify_any.OBJECT_TASK,
        classify_any.CAPTION_TASK,
        classify_any.OBJECT_TASK,
    ]
