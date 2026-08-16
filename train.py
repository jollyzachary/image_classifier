from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim

from checkpoint import save_checkpoint
from data_preprocessing import load_and_preprocess
from model import SUPPORTED_ARCHITECTURES, build_model, evaluate_model, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a transfer-learning image classifier."
    )
    parser.add_argument(
        "data_directory",
        help="Dataset root containing train, valid, and test directories.",
    )
    parser.add_argument(
        "--save-dir",
        default=".",
        help="Directory for checkpoint.pth (default: current directory).",
    )
    parser.add_argument(
        "--arch",
        choices=SUPPORTED_ARCHITECTURES,
        default="vgg16",
        help="Pretrained backbone architecture.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-units", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Require CUDA instead of using the CPU.",
    )
    return parser


def select_device(require_gpu: bool) -> torch.device:
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("--gpu was requested, but CUDA is not available")
    return torch.device("cuda" if require_gpu else "cpu")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if args.batch_size < 1 or args.hidden_units < 1 or args.epochs < 1:
        raise ValueError("batch size, hidden units, and epochs must be positive")
    if args.num_workers < 0:
        raise ValueError("number of workers cannot be negative")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.gpu)
    trainloader, validloader, testloader = load_and_preprocess(
        args.data_directory,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    output_size = len(trainloader.dataset.classes)
    print(
        f"Training {args.arch} classifier with {output_size} classes on {device.type}"
    )
    network = build_model(
        args.arch,
        hidden_units=args.hidden_units,
        output_size=output_size,
    )
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(network.classifier.parameters(), lr=args.learning_rate)

    train_model(
        network,
        trainloader,
        validloader,
        criterion,
        optimizer,
        device,
        epochs=args.epochs,
    )
    test_loss, test_accuracy = evaluate_model(network, testloader, criterion, device)
    print(f"Test loss {test_loss:.4f} | test accuracy {test_accuracy:.2%}")

    checkpoint_path = Path(args.save_dir).expanduser() / "checkpoint.pth"
    saved_path = save_checkpoint(
        network,
        optimizer,
        trainloader.dataset.class_to_idx,
        checkpoint_path,
        architecture=args.arch,
        hidden_units=args.hidden_units,
        output_size=output_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print(f"Saved checkpoint to {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
