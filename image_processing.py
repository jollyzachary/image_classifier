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
) -> None:
    """Display an image beside a horizontal probability chart."""

    with Image.open(Path(image_path).expanduser()) as image:
        display_image = image.convert("RGB").copy()

    figure, (image_axis, chart_axis) = plt.subplots(2, 1, figsize=(7, 9))
    image_axis.imshow(display_image)
    image_axis.axis("off")

    positions = np.arange(len(labels))
    chart_axis.barh(positions, probabilities)
    chart_axis.set_yticks(positions, labels=labels)
    chart_axis.invert_yaxis()
    chart_axis.set_xlabel("Probability")
    chart_axis.set_xlim(0, 1)
    figure.tight_layout()
    plt.show()
