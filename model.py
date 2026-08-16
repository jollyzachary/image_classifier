from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

SUPPORTED_ARCHITECTURES = ("vgg16",)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    training_loss: float
    validation_loss: float
    validation_accuracy: float


def build_model(
    architecture: str,
    *,
    hidden_units: int,
    output_size: int,
    pretrained: bool = True,
) -> nn.Module:
    """Build a VGG16 transfer-learning classifier."""

    if architecture not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)
        raise ValueError(f"unsupported architecture {architecture!r}; choose {supported}")
    if hidden_units < 1 or output_size < 2:
        raise ValueError("hidden_units must be positive and output_size must be at least 2")

    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    network = models.vgg16(weights=weights)
    for parameter in network.features.parameters():
        parameter.requires_grad = False

    input_features = network.classifier[0].in_features
    network.classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, output_size),
        nn.LogSoftmax(dim=1),
    )
    return network


def evaluate_model(
    network: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return average loss and accuracy for a dataset loader."""

    network.eval()
    total_loss = 0.0
    correct = 0
    sample_count = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            log_probabilities = network(inputs)
            loss = criterion(log_probabilities, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (log_probabilities.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("cannot evaluate an empty dataset")
    return total_loss / sample_count, correct / sample_count


def train_model(
    network: nn.Module,
    trainloader: DataLoader,
    validloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epochs: int,
) -> list[EpochMetrics]:
    """Train the classifier head and report validation metrics each epoch."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    history: list[EpochMetrics] = []
    network.to(device)

    for epoch in range(1, epochs + 1):
        network.train()
        running_loss = 0.0
        sample_count = 0

        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            log_probabilities = network(inputs)
            loss = criterion(log_probabilities, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise ValueError("cannot train on an empty dataset")

        validation_loss, validation_accuracy = evaluate_model(
            network, validloader, criterion, device
        )
        metrics = EpochMetrics(
            epoch=epoch,
            training_loss=running_loss / sample_count,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
        )
        history.append(metrics)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {metrics.training_loss:.4f} | "
            f"validation loss {metrics.validation_loss:.4f} | "
            f"validation accuracy {metrics.validation_accuracy:.2%}"
        )

    return history
