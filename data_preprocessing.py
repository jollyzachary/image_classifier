from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return training and evaluation transforms for ImageNet-based models."""

    training = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return training, evaluation


def load_and_preprocess(
    data_dir: str | Path,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load ``train``, ``valid``, and ``test`` image-folder splits."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    root = Path(data_dir).expanduser()
    split_paths = {name: root / name for name in ("train", "valid", "test")}
    missing = [str(path) for path in split_paths.values() if not path.is_dir()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"missing dataset split directories: {joined}")

    training_transform, evaluation_transform = build_transforms()
    train_data = datasets.ImageFolder(
        split_paths["train"], transform=training_transform
    )
    valid_data = datasets.ImageFolder(
        split_paths["valid"], transform=evaluation_transform
    )
    test_data = datasets.ImageFolder(
        split_paths["test"], transform=evaluation_transform
    )

    if len(train_data.classes) < 2:
        raise ValueError("the dataset must contain at least two classes")
    for split_name, split_data in (("valid", valid_data), ("test", test_data)):
        if split_data.class_to_idx != train_data.class_to_idx:
            raise ValueError(
                f"{split_name} classes must match the train split exactly"
            )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    trainloader = DataLoader(train_data, shuffle=True, **loader_options)
    validloader = DataLoader(valid_data, shuffle=False, **loader_options)
    testloader = DataLoader(test_data, shuffle=False, **loader_options)
    return trainloader, validloader, testloader
