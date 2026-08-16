from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the recorded any-image analysis demonstration."
    )
    parser.add_argument(
        "--results",
        default="./docs/assets/classify-any-results.json",
        help="JSON output created by classify_any.py.",
    )
    parser.add_argument(
        "--output",
        default="./docs/assets/classify-any-demo.png",
        help="Destination for the rendered demonstration.",
    )
    return parser


def load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"results file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    analyses = payload.get("analyses")
    if not isinstance(analyses, list) or len(analyses) != 4:
        raise ValueError("results must contain exactly four analyses")
    return payload


def render(payload: dict[str, Any], destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 9.2))
    figure.patch.set_facecolor("#090b0e")
    for axis, analysis in zip(axes.flat, payload["analyses"], strict=True):
        image_path = Path(str(analysis["image"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"demo image not found: {image_path}")
        with Image.open(image_path) as image:
            display_image = ImageOps.fit(
                image.convert("RGB"),
                (1200, 650),
                method=Image.Resampling.LANCZOS,
            )

        axis.imshow(display_image)
        axis.set_facecolor("#11151a")
        axis.axis("off")
        axis.set_title(
            f"PRIMARY  /  {str(analysis['primary_subject']).upper()}",
            loc="left",
            color="#f5f2e8",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        caption = "\n".join(textwrap.wrap(str(analysis["description"]), width=76))
        axis.text(
            0,
            -0.055,
            caption,
            transform=axis.transAxes,
            color="#8d98a7",
            fontsize=9,
            va="top",
        )

    figure.suptitle(
        "CLASSIFY ANY IMAGE",
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
        "Primary subject · scene description · localized objects · no candidate labels",
        color="#8d98a7",
        fontsize=12,
    )
    figure.text(
        0.945,
        0.025,
        "FLORENCE-2  /  LOCAL INFERENCE",
        ha="right",
        color="#687382",
        fontsize=9,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.945,
        top=0.875,
        bottom=0.075,
        wspace=0.05,
        hspace=0.28,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> int:
    args = build_parser().parse_args()
    payload = load_payload(Path(args.results).expanduser())
    destination = Path(args.output).expanduser()
    render(payload, destination)
    print(f"Rendered any-image demonstration: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
