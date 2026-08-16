from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from checkpoint import load_checkpoint
from image_processing import display_predictions, process_image
from label_mapping import load_label_mapping
from runtime import DEVICE_CHOICES, select_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict image classes from a trained checkpoint."
    )
    parser.add_argument("image_path", help="Path to the image to classify.")
    parser.add_argument("checkpoint", help="Path to a saved checkpoint.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--category-names",
        help="Optional JSON mapping from class identifiers to display names.",
    )
    device_options = parser.add_mutually_exclusive_group()
    device_options.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="cpu",
        help="Compute device (default: cpu; auto selects the best available device).",
    )
    device_options.add_argument(
        "--gpu",
        action="store_true",
        help="Equivalent to --device cuda.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the image and a probability chart.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for a saved prediction visualization.",
    )
    return parser


def predict(
    image_path: str | Path,
    network: nn.Module,
    device: torch.device,
    *,
    top_k: int,
    category_names: dict[str, str] | None = None,
) -> tuple[list[float], list[str], list[str]]:
    """Return probabilities, class identifiers, and display labels."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    class_to_idx = getattr(network, "class_to_idx", None)
    if not class_to_idx:
        raise ValueError("checkpoint does not include a class-to-index mapping")

    tensor = process_image(image_path).unsqueeze(0).to(device)
    network.eval()
    with torch.no_grad():
        probabilities = torch.exp(network(tensor))

    count = min(top_k, probabilities.size(1))
    top_probabilities, top_indices = probabilities.topk(count, dim=1)
    inverse_mapping = {int(index): str(label) for label, index in class_to_idx.items()}
    try:
        class_ids = [inverse_mapping[int(index)] for index in top_indices[0].cpu()]
    except KeyError as error:
        raise ValueError(
            f"checkpoint class mapping has no label for output index {error.args[0]}"
        ) from error
    probability_values = [float(value) for value in top_probabilities[0].cpu()]
    labels = [
        category_names.get(class_id, class_id) if category_names else class_id
        for class_id in class_ids
    ]
    return probability_values, class_ids, labels


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = select_device("cuda" if args.gpu else args.device)
    network, _ = load_checkpoint(args.checkpoint, device)
    category_names = (
        load_label_mapping(args.category_names) if args.category_names else None
    )
    probabilities, class_ids, labels = predict(
        args.image_path,
        network,
        device,
        top_k=args.top_k,
        category_names=category_names,
    )

    print("rank\tprobability\tclass\tlabel")
    for rank, (probability, class_id, label) in enumerate(
        zip(probabilities, class_ids, labels, strict=True), start=1
    ):
        print(f"{rank}\t{probability:.4f}\t{class_id}\t{label}")

    if args.show or args.output:
        display_predictions(
            args.image_path,
            probabilities,
            labels,
            output_path=args.output,
            show=args.show,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
