from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from torchvision.datasets import Flowers102

DEFAULT_CLASS_IDS = (54, 73, 77)
SOURCE_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a compact Oxford 102 Flowers demonstration dataset."
    )
    parser.add_argument(
        "--download-root",
        default="./data/torchvision",
        help="Local cache for the original Flowers102 download.",
    )
    parser.add_argument(
        "--output",
        default="./data/flowers-demo",
        help="Empty destination for train, valid, and test folders.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=DEFAULT_CLASS_IDS,
        help="One-indexed Flowers102 class identifiers.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=20,
        help="Maximum held-out test images per class (default: 20).",
    )
    return parser


def prepare_split(
    dataset: Flowers102,
    destination: Path,
    selected_classes: set[int],
    limit_per_class: int | None,
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for index in range(len(dataset)):
        image, zero_based_target = dataset[index]
        class_id = int(zero_based_target) + 1
        if class_id not in selected_classes:
            continue
        if limit_per_class is not None and counts[class_id] >= limit_per_class:
            continue

        class_directory = destination / str(class_id)
        class_directory.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(
            class_directory / f"image_{index:05d}.jpg",
            format="JPEG",
            quality=95,
        )
        counts[class_id] += 1
    return counts


def main() -> int:
    args = build_parser().parse_args()
    if args.test_limit < 1:
        raise ValueError("test limit must be positive")

    selected_classes = set(args.classes)
    if len(selected_classes) < 2:
        raise ValueError("select at least two distinct classes")
    if any(
        class_id < 1 or class_id > len(Flowers102.classes)
        for class_id in selected_classes
    ):
        raise ValueError("class identifiers must be between 1 and 102")

    output = Path(args.output).expanduser()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    download_root = Path(args.download_root).expanduser()
    split_names = {"train": "train", "val": "valid", "test": "test"}
    manifest_counts: dict[str, dict[str, int]] = {}

    for source_split, destination_split in split_names.items():
        dataset = Flowers102(
            root=download_root,
            split=source_split,
            download=True,
        )
        limit = args.test_limit if source_split == "test" else None
        counts = prepare_split(
            dataset,
            output / destination_split,
            selected_classes,
            limit,
        )
        manifest_counts[destination_split] = {
            str(class_id): counts[class_id] for class_id in sorted(selected_classes)
        }

    classes = {
        str(class_id): Flowers102.classes[class_id - 1]
        for class_id in sorted(selected_classes)
    }
    manifest = {
        "source": SOURCE_URL,
        "classes": classes,
        "counts": manifest_counts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
