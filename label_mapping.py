from __future__ import annotations

import json
from pathlib import Path


def load_label_mapping(file_path: str | Path = "cat_to_name.json") -> dict[str, str]:
    """Load a JSON mapping from dataset class identifiers to display names."""

    path = Path(file_path).expanduser()
    with path.open("r", encoding="utf-8") as file:
        mapping = json.load(file)
    if not isinstance(mapping, dict):
        raise ValueError("category mapping must be a JSON object")
    return {str(key): str(value) for key, value in mapping.items()}
