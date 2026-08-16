import predict
import train


def test_training_defaults_are_stable() -> None:
    arguments = train.build_parser().parse_args(["dataset"])

    assert arguments.arch == "vgg16"
    assert arguments.epochs == 5
    assert arguments.hidden_units == 512
    assert arguments.gpu is False


def test_prediction_defaults_are_stable() -> None:
    arguments = predict.build_parser().parse_args(["image.jpg", "checkpoint.pth"])

    assert arguments.top_k == 5
    assert arguments.gpu is False
    assert arguments.show is False
