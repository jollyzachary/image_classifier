import pytest

import runtime


def test_auto_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime.torch.backends.mps, "is_available", lambda: False)

    assert runtime.select_device("auto").type == "cpu"


def test_unavailable_mps_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime.torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS was requested"):
        runtime.select_device("mps")
