# Image Classifier Engine

A local transfer-learning pipeline for training an image classifier on a
folder-structured dataset and running top-k predictions from the command line.

The maintained implementation uses a pretrained VGG16 backbone from
TorchVision, replaces its classifier head for the target dataset, and stores a
portable checkpoint containing model weights and class metadata. The project
does not require a hosted service.

## Capabilities

- Loads separate training, validation, and test splits with `ImageFolder`.
- Applies data augmentation during training and deterministic preprocessing for
  evaluation.
- Freezes VGG16 feature layers and trains a dataset-specific classifier head.
- Reports validation metrics after every epoch and evaluates the final model on
  the test split.
- Saves architecture, hyperparameters, class mappings, model weights, and
  optimizer state in one checkpoint.
- Returns top-k predictions with optional human-readable category names.
- Runs on CPU by default and supports CUDA when `--gpu` is supplied.

## Architecture

```text
ImageFolder dataset
        │
        ├── train transforms ──┐
        └── eval transforms  ──┤
                               ▼
                    frozen VGG16 features
                               │
                    trainable classifier head
                               │
                validation and test evaluation
                               │
             versioned checkpoint + class mapping
                               │
                     top-k image prediction
```

The implementation keeps data loading, model construction, training,
checkpointing, and inference separate. That makes the components usable from
Python as well as through the command-line entry points, without introducing a
framework or service layer that the project does not need.

## Project structure

```text
data_preprocessing.py  Dataset validation, transforms, and loaders
model.py               Model construction, training, and evaluation
checkpoint.py          Portable checkpoint save and load functions
image_processing.py    Inference preprocessing and visualization
train.py               Training command
predict.py             Prediction command
main.py                Unified command dispatcher
cat_to_name.json       Oxford 102 Flowers category labels
notebooks/             Archived coursework notebook
tests/                 Lightweight unit tests
```

## Requirements

- Python 3.10 or newer
- PyTorch and TorchVision
- NumPy, Pillow, and Matplotlib

Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Dataset layout

The dataset root must contain `train`, `valid`, and `test` directories. Each
split uses one subdirectory per class:

```text
dataset/
├── train/
│   ├── class_a/
│   └── class_b/
├── valid/
│   ├── class_a/
│   └── class_b/
└── test/
    ├── class_a/
    └── class_b/
```

The original project used the
[Oxford 102 Flowers dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
Dataset files and trained checkpoints are intentionally excluded from the
repository.

## Train a classifier

```bash
python train.py ./dataset \
  --epochs 5 \
  --learning-rate 0.001 \
  --hidden-units 512 \
  --save-dir ./artifacts
```

Add `--gpu` to require CUDA. The command fails clearly if CUDA is requested but
unavailable.

The same operation is available through the unified entry point:

```bash
python main.py train ./dataset --epochs 5 --save-dir ./artifacts
```

## Run a prediction

```bash
python predict.py ./example.jpg ./artifacts/checkpoint.pth \
  --top-k 5 \
  --category-names cat_to_name.json
```

Add `--show` to display the image and probability chart, or `--gpu` to require
CUDA. The unified form is:

```bash
python main.py predict ./example.jpg ./artifacts/checkpoint.pth --top-k 5
```

## Checkpoints and safety

Checkpoints contain tensor weights and the metadata needed to reconstruct the
classifier. Load only checkpoints you created or obtained from a trusted
source. No pretrained project checkpoint is distributed in this repository.

The checkpoint format is versioned and stores state dictionaries rather than
serialized model objects. A checkpoint records the architecture, classifier
dimensions, training settings, and class-to-index mapping required for
reconstruction.

## Legacy notebook

The original Udacity project notebook is preserved in `notebooks/` with its
execution outputs removed. It documents the project's starting point; the
maintained command-line modules are the canonical implementation.

## Development

Install the development dependency and run the lightweight checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Acknowledgments

This project began as part of Udacity's AI Programming with Python Nanodegree.
It uses PyTorch, TorchVision pretrained weights, and the Oxford 102 Flowers
dataset created by Maria-Elena Nilsback and Andrew Zisserman. Third-party data
and model weights remain subject to their respective terms.

## License

Original code in this repository is available under the [MIT License](LICENSE).
