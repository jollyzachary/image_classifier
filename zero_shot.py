from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from image_processing import display_predictions
from runtime import DEVICE_CHOICES, select_device

DEFAULT_MODEL = "google/siglip2-base-patch16-224"
Classifier = Callable[..., list[dict[str, Any]]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify one or more images against new candidate labels without "
            "training a checkpoint."
        )
    )
    parser.add_argument("images", nargs="+", help="Image paths to classify.")
    label_source = parser.add_mutually_exclusive_group(required=True)
    label_source.add_argument(
        "--labels",
        nargs="+",
        help="Candidate labels to rank for every image.",
    )
    label_source.add_argument(
        "--labels-file",
        help="Text file containing one candidate label per line.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model identifier (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Compute device (default: auto).",
    )
    visual_output = parser.add_mutually_exclusive_group()
    visual_output.add_argument(
        "--output",
        help="Optional visualization path; valid for a single input image.",
    )
    visual_output.add_argument(
        "--output-dir",
        help="Optional directory for one visualization per input image.",
    )
    parser.add_argument("--json", dest="json_output", help="Optional JSON result path.")
    parser.add_argument("--show", action="store_true")
    return parser


def normalize_labels(labels: Sequence[str]) -> list[str]:
    """Trim candidate labels, preserve order, and reject ambiguous input."""

    normalized: list[str] = []
    seen: set[str] = set()
    for label in labels:
        clean_label = label.strip()
        key = clean_label.casefold()
        if clean_label and key not in seen:
            normalized.append(clean_label)
            seen.add(key)
    if len(normalized) < 2:
        raise ValueError("provide at least two distinct candidate labels")
    return normalized


def load_labels(inline: Sequence[str] | None, labels_file: str | None) -> list[str]:
    if inline is not None:
        return normalize_labels(inline)
    if labels_file is None:
        raise ValueError("candidate labels are required")

    path = Path(labels_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"labels file not found: {path}")
    return normalize_labels(path.read_text(encoding="utf-8").splitlines())


def create_classifier(model_id: str, device: torch.device) -> Classifier:
    """Load the Transformers zero-shot image pipeline."""

    try:
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError(
            "zero-shot mode requires: python -m pip install -r requirements-vision.txt"
        ) from error

    classifier = pipeline(
        task="zero-shot-image-classification",
        model=model_id,
        device=device,
    )
    if getattr(classifier.tokenizer, "model_max_length", 0) > 64:
        classifier.tokenizer.model_max_length = 64
    return classifier


def classify_image(
    image_path: str | Path,
    labels: Sequence[str],
    classifier: Classifier,
    *,
    top_k: int,
) -> list[dict[str, float | str]]:
    """Rank candidate labels for one image and return normalized records."""

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    with Image.open(path) as image:
        raw_results = classifier(
            image.convert("RGB"),
            candidate_labels=list(labels),
        )

    results: list[dict[str, float | str]] = []
    for item in raw_results[: min(top_k, len(raw_results))]:
        label = item.get("label")
        score = item.get("score")
        if not isinstance(label, str) or not isinstance(score, (float, int)):
            raise ValueError("model returned an invalid zero-shot prediction")
        results.append({"label": label, "score": float(score)})
    if not results:
        raise ValueError("model returned no zero-shot predictions")
    return results


def visualization_path(args: argparse.Namespace, image_path: Path) -> Path | None:
    if args.output:
        return Path(args.output).expanduser()
    if args.output_dir:
        return Path(args.output_dir).expanduser() / f"{image_path.stem}-prediction.png"
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output and len(args.images) != 1:
        raise ValueError("--output can only be used with one input image")
    labels = load_labels(args.labels, args.labels_file)
    device = select_device(args.device)
    classifier = create_classifier(args.model, device)

    payload: dict[str, Any] = {
        "model": args.model,
        "device": str(device),
        "candidate_labels": labels,
        "predictions": [],
    }
    for image_argument in args.images:
        image_path = Path(image_argument).expanduser()
        results = classify_image(
            image_path,
            labels,
            classifier,
            top_k=args.top_k,
        )
        payload["predictions"].append({"image": str(image_path), "results": results})

        print(f"\n{image_path}")
        print("rank\tscore\tlabel")
        for rank, result in enumerate(results, start=1):
            print(f"{rank}\t{result['score']:.4f}\t{result['label']}")

        output_path = visualization_path(args, image_path)
        if args.show or output_path:
            display_predictions(
                image_path,
                [float(result["score"]) for result in results],
                [str(result["label"]) for result in results],
                output_path=output_path,
                show=args.show,
                figure_title="OPEN-VOCABULARY VISION",
                subtitle=f"SigLIP 2 · {len(labels)} candidate concepts · local inference",
                chart_title="SEMANTIC MATCHES",
                axis_label="MATCH SCORE",
            )

    if args.json_output:
        destination = Path(args.json_output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
