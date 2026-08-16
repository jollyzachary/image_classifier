import json
from pathlib import Path

import pytest

from label_mapping import load_label_mapping


def test_load_label_mapping_normalizes_keys_and_values(tmp_path: Path) -> None:
    mapping_path = tmp_path / "labels.json"
    mapping_path.write_text(json.dumps({1: "rose", "2": "daisy"}), encoding="utf-8")

    assert load_label_mapping(mapping_path) == {"1": "rose", "2": "daisy"}


def test_load_label_mapping_requires_object(tmp_path: Path) -> None:
    mapping_path = tmp_path / "labels.json"
    mapping_path.write_text(json.dumps(["rose"]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_label_mapping(mapping_path)
