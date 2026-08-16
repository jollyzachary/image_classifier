from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data_preprocessing import IMAGENET_MEAN, IMAGENET_STD

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def process_image(image_path: str | Path) -> torch.Tensor:
    """Load an image and return a normalized ``3 x 224 x 224`` tensor."""

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as image:
        return INFERENCE_TRANSFORM(image.convert("RGB"))


def display_predictions(
    image_path: str | Path,
    probabilities: list[float],
    labels: list[str],
    *,
    output_path: str | Path | None = None,
    show: bool = True,
    figure_title: str = "IMAGE CLASSIFIER ENGINE",
    subtitle: str = "VGG16 transfer learning · ranked inference",
    chart_title: str = "TOP PREDICTIONS",
    axis_label: str = "PROBABILITY",
) -> None:
    """Render an image and ranked score chart, optionally saving the figure."""

    with Image.open(Path(image_path).expanduser()) as image:
        display_image = image.convert("RGB").copy()

    figure, (image_axis, chart_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 6.4),
        gridspec_kw={"width_ratios": (1.08, 1)},
    )
    figure.patch.set_facecolor("#0b0d10")
    image_axis.set_facecolor("#0b0d10")
    chart_axis.set_facecolor("#11151a")

    image_axis.imshow(display_image)
    image_axis.axis("off")
    image_axis.set_title(
        "INPUT IMAGE",
        color="#aeb6c2",
        fontsize=10,
        fontweight="bold",
        loc="left",
        pad=12,
    )

    positions = np.arange(len(labels))
    colors = ["#e5b94f", *(["#566171"] * max(0, len(labels) - 1))]
    bars = chart_axis.barh(positions, probabilities, color=colors, height=0.5)
    chart_axis.set_yticks(positions, labels=labels)
    chart_axis.invert_yaxis()
    chart_axis.set_xlabel(axis_label, color="#8f99a8", fontsize=9, labelpad=12)
    chart_axis.set_xlim(0, 1)
    chart_axis.set_title(
        chart_title,
        color="#aeb6c2",
        fontsize=10,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    chart_axis.tick_params(axis="x", colors="#76808d")
    chart_axis.tick_params(axis="y", colors="#f3f4f6", labelsize=11)
    chart_axis.grid(axis="x", color="#2a3038", linewidth=0.7, alpha=0.8)
    chart_axis.set_axisbelow(True)
    for spine in chart_axis.spines.values():
        spine.set_visible(False)
    for bar, probability in zip(bars, probabilities, strict=True):
        chart_axis.text(
            min(probability + 0.025, 0.93),
            bar.get_y() + bar.get_height() / 2,
            f"{probability:.2%}",
            va="center",
            color="#f3f4f6",
            fontsize=10,
            fontweight="bold",
        )

    figure.suptitle(
        figure_title,
        color="#f5f2e8",
        fontsize=17,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    figure.text(
        0.06,
        0.925,
        subtitle,
        color="#7f8996",
        fontsize=10,
    )
    figure.subplots_adjust(left=0.06, right=0.97, top=0.84, bottom=0.09, wspace=0.18)
    if output_path is not None:
        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            destination,
            dpi=160,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
    if show:
        plt.show()
    plt.close(figure)
