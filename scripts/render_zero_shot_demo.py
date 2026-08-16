from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the recorded open-vocabulary demonstration."
    )
    parser.add_argument(
        "--results",
        default="./artifacts/open-vocabulary-results.json",
        help="JSON output created by zero_shot.py.",
    )
    parser.add_argument(
        "--output",
        default="./docs/assets/open-vocabulary-demo.png",
        help="Destination for the rendered demonstration.",
    )
    parser.add_argument(
        "--title",
        default="OPEN-VOCABULARY CLASSIFICATION",
        help="Figure title.",
    )
    parser.add_argument(
        "--subtitle",
        default="One model · eight new concepts · no task-specific training",
        help="Figure subtitle.",
    )
    parser.add_argument(
        "--footer",
        default="SIGLIP 2  /  LOCAL INFERENCE",
        help="Figure footer.",
    )
    return parser


def load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"results file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("results file does not contain predictions")
    return payload


def render(
    payload: dict[str, Any],
    destination: Path,
    *,
    title: str = "OPEN-VOCABULARY CLASSIFICATION",
    subtitle: str = "One model · eight new concepts · no task-specific training",
    footer: str = "SIGLIP 2  /  LOCAL INFERENCE",
) -> None:
    predictions = payload["predictions"]
    if len(predictions) != 4:
        raise ValueError("results must contain exactly four predictions")

    figure, axes = plt.subplots(2, 2, figsize=(16, 9.2))
    figure.patch.set_facecolor("#090b0e")
    for axis, prediction in zip(axes.flat, predictions, strict=True):
        image_path = Path(prediction["image"])
        results = prediction.get("results")
        if not image_path.is_file() or not isinstance(results, list) or not results:
            raise ValueError(f"invalid prediction record for {image_path}")

        with Image.open(image_path) as image:
            display_image = ImageOps.fit(
                image.convert("RGB"),
                (1200, 650),
                method=Image.Resampling.LANCZOS,
            )
        top_label = str(results[0]["label"]).upper()

        axis.imshow(display_image)
        axis.set_facecolor("#11151a")
        axis.axis("off")
        axis.set_title(
            f"TOP MATCH  /  {top_label}",
            loc="left",
            color="#f5f2e8",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.suptitle(
        title,
        x=0.055,
        y=0.975,
        ha="left",
        color="#f5f2e8",
        fontsize=22,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.925,
        subtitle,
        color="#8d98a7",
        fontsize=12,
    )
    figure.text(
        0.945,
        0.035,
        footer,
        ha="right",
        color="#687382",
        fontsize=9,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.055, right=0.945, top=0.875, bottom=0.075, wspace=0.05, hspace=0.18
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=120,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def main() -> int:
    args = build_parser().parse_args()
    payload = load_payload(Path(args.results).expanduser())
    destination = Path(args.output).expanduser()
    render(
        payload,
        destination,
        title=args.title,
        subtitle=args.subtitle,
        footer=args.footer,
    )
    print(f"Rendered open-vocabulary demonstration: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
