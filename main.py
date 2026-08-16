from __future__ import annotations

import sys

import classify_any
import predict
import train
import zero_shot


def main(argv: list[str] | None = None) -> int:
    """Dispatch the unified training and inference commands."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {"train", "predict", "zero-shot", "classify-any"}
    if not arguments or arguments[0] not in commands:
        print(
            "Usage: python main.py {train|predict|zero-shot|classify-any} [arguments]"
        )
        return 2

    command, command_arguments = arguments[0], arguments[1:]
    if command == "train":
        return train.main(command_arguments)
    if command == "predict":
        return predict.main(command_arguments)
    if command == "zero-shot":
        return zero_shot.main(command_arguments)
    return classify_any.main(command_arguments)


if __name__ == "__main__":
    raise SystemExit(main())
