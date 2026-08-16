from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from runtime import DEVICE_CHOICES, select_device

DEFAULT_MODEL = "microsoft/Florence-2-base-ft"
DEFAULT_REVISION = "f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"
CAPTION_TASK = "<DETAILED_CAPTION>"
OBJECT_TASK = "<OD>"
TaskRunner = Callable[[Image.Image, str], dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Identify and describe images without a checkpoint or candidate labels."
        )
    )
    parser.add_argument("images", nargs="+", help="Image paths to analyze.")
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Compute device (default: auto).",
    )
    visual_output = parser.add_mutually_exclusive_group()
    visual_output.add_argument(
        "--output",
        help="Optional annotated-image path; valid for one input image.",
    )
    visual_output.add_argument(
        "--output-dir",
        help="Optional directory for one annotated result per input image.",
    )
    parser.add_argument("--json", dest="json_output", help="Optional JSON result path.")
    parser.add_argument("--show", action="store_true")
    return parser


def create_task_runner(device: torch.device) -> TaskRunner:
    """Load the pinned Florence-2 model and return a reusable task runner."""

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "classify-any mode requires: python -m pip install "
            "-r requirements-vision.txt"
        ) from error

    processor = AutoProcessor.from_pretrained(
        DEFAULT_MODEL,
        revision=DEFAULT_REVISION,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        revision=DEFAULT_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to(device)
    model.eval()

    def run(image: Image.Image, task: str) -> dict[str, Any]:
        inputs = processor(text=task, images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        return processor.post_process_generation(
            generated_text,
            task=task,
            image_size=image.size,
        )

    return run


def parse_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    detection = payload.get(OBJECT_TASK, {})
    boxes = detection.get("bboxes", []) if isinstance(detection, dict) else []
    labels = detection.get("labels", []) if isinstance(detection, dict) else []
    if not isinstance(boxes, list) or not isinstance(labels, list):
        raise ValueError("model returned invalid object-detection data")

    objects: list[dict[str, Any]] = []
    for box, label in zip(boxes, labels, strict=False):
        if not isinstance(box, list) or len(box) != 4 or not isinstance(label, str):
            continue
        coordinates = [float(value) for value in box]
        width = max(0.0, coordinates[2] - coordinates[0])
        height = max(0.0, coordinates[3] - coordinates[1])
        objects.append(
            {
                "label": label.strip(),
                "box": coordinates,
                "area": width * height,
            }
        )
    objects.sort(key=lambda item: float(item["area"]), reverse=True)
    return objects


def select_primary_subject(caption: str, objects: list[dict[str, Any]]) -> str:
    """Choose the most specific detected label supported by the caption."""

    if not objects:
        return "scene"
    caption_words = set(re.findall(r"[a-z0-9]+", caption.casefold()))
    supported: list[tuple[int, float, str]] = []
    for detected in objects:
        label = str(detected["label"])
        label_words = set(re.findall(r"[a-z0-9]+", label.casefold()))
        if label_words and label_words.issubset(caption_words):
            supported.append((len(label_words), float(detected["area"]), label))
    if supported:
        return max(supported)[2]
    return str(objects[0]["label"])


def analyze_image(image_path: str | Path, runner: TaskRunner) -> dict[str, Any]:
    """Return a caption, primary subject, and localized objects for one image."""

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as source:
        image = source.convert("RGB")
        caption_payload = runner(image, CAPTION_TASK)
        object_payload = runner(image, OBJECT_TASK)

    caption = caption_payload.get(CAPTION_TASK)
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("model returned no image description")
    caption = caption.strip()
    objects = parse_objects(object_payload)
    primary_subject = select_primary_subject(caption, objects)
    return {
        "image": str(path),
        "primary_subject": primary_subject,
        "description": caption,
        "objects": objects,
    }


class AnyImageClassifier:
    """Reusable local vision interface for applications and command-line tools."""

    def __init__(
        self,
        device: str | torch.device = "auto",
        *,
        runner: TaskRunner | None = None,
    ) -> None:
        self.device = select_device(device) if isinstance(device, str) else device
        self._runner = runner or create_task_runner(self.device)

    def classify(self, image_path: str | Path) -> dict[str, Any]:
        return analyze_image(image_path, self._runner)

    def classify_many(self, image_paths: list[str | Path]) -> list[dict[str, Any]]:
        return [self.classify(image_path) for image_path in image_paths]


def display_analysis(
    analysis: dict[str, Any],
    *,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Render object locations and the generated scene description."""

    image_path = Path(str(analysis["image"]))
    with Image.open(image_path) as source:
        image = source.convert("RGB").copy()

    figure, axis = plt.subplots(figsize=(12, 8.4))
    figure.patch.set_facecolor("#090b0e")
    axis.set_facecolor("#090b0e")
    axis.imshow(image)
    axis.axis("off")

    colors = ("#e5b94f", "#d4dbe5", "#7f8da1", "#b8895e")
    for index, detected in enumerate(analysis["objects"]):
        x1, y1, x2, y2 = detected["box"]
        color = colors[index % len(colors)]
        axis.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2.2,
                edgecolor=color,
                facecolor="none",
            )
        )
        axis.text(
            x1,
            max(0, y1 - 8),
            str(detected["label"]).upper(),
            color="#090b0e",
            fontsize=9,
            fontweight="bold",
            va="bottom",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 4},
        )

    figure.suptitle(
        f"PRIMARY SUBJECT  /  {str(analysis['primary_subject']).upper()}",
        x=0.055,
        y=0.97,
        ha="left",
        color="#f5f2e8",
        fontsize=18,
        fontweight="bold",
    )
    wrapped_description = "\n".join(
        textwrap.wrap(str(analysis["description"]), width=105)
    )
    figure.text(
        0.055,
        0.04,
        wrapped_description,
        color="#a4adba",
        fontsize=10,
        va="bottom",
    )
    figure.text(
        0.945,
        0.04,
        "FLORENCE-2  /  LOCAL",
        ha="right",
        color="#687382",
        fontsize=8,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.055, right=0.945, top=0.9, bottom=0.14)

    if output_path is not None:
        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            destination,
            dpi=140,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
    if show:
        plt.show()
    plt.close(figure)


def visualization_path(args: argparse.Namespace, image_path: Path) -> Path | None:
    if args.output:
        return Path(args.output).expanduser()
    if args.output_dir:
        return Path(args.output_dir).expanduser() / f"{image_path.stem}-analysis.png"
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output and len(args.images) != 1:
        raise ValueError("--output can only be used with one input image")

    classifier = AnyImageClassifier(args.device)
    payload: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "revision": DEFAULT_REVISION,
        "device": str(classifier.device),
        "analyses": [],
    }
    for image_argument in args.images:
        analysis = classifier.classify(image_argument)
        payload["analyses"].append(analysis)
        print(f"\n{analysis['image']}")
        print(f"Primary subject: {analysis['primary_subject']}")
        print(f"Description: {analysis['description']}")
        if analysis["objects"]:
            labels = ", ".join(str(item["label"]) for item in analysis["objects"])
            print(f"Detected objects: {labels}")

        output_path = visualization_path(args, Path(str(analysis["image"])))
        if args.show or output_path:
            display_analysis(analysis, output_path=output_path, show=args.show)

    if args.json_output:
        destination = Path(args.json_output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
