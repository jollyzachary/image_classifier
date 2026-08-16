import predict
import train
import zero_shot


def test_training_defaults_are_stable() -> None:
    arguments = train.build_parser().parse_args(["dataset"])

    assert arguments.arch == "vgg16"
    assert arguments.epochs == 5
    assert arguments.hidden_units == 512
    assert arguments.device == "cpu"
    assert arguments.gpu is False
    assert arguments.metrics_output is None


def test_prediction_defaults_are_stable() -> None:
    arguments = predict.build_parser().parse_args(["image.jpg", "checkpoint.pth"])

    assert arguments.top_k == 5
    assert arguments.device == "cpu"
    assert arguments.gpu is False
    assert arguments.show is False
    assert arguments.output is None


def test_zero_shot_parser_accepts_batch_input() -> None:
    arguments = zero_shot.build_parser().parse_args(
        ["first.jpg", "second.jpg", "--labels", "cat", "dog", "car"]
    )

    assert arguments.images == ["first.jpg", "second.jpg"]
    assert arguments.labels == ["cat", "dog", "car"]
    assert arguments.device == "auto"
