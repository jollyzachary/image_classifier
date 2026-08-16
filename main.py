from __future__ import annotations

import sys

import predict
import train


def main(argv: list[str] | None = None) -> int:
    """Dispatch the unified ``train`` and ``predict`` commands."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {"train", "predict"}:
        print("Usage: python main.py {train|predict} [arguments]")
        return 2

    command, command_arguments = arguments[0], arguments[1:]
    if command == "train":
        return train.main(command_arguments)
    return predict.main(command_arguments)


if __name__ == "__main__":
    raise SystemExit(main())
