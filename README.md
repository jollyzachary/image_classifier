# Image Classification Program

This program allows for the classification of any user-defined dataset of images. It utilizes deep learning techniques to train a model on the provided dataset and make predictions on new images.

## Overview

The image classification program consists of the following components:

1. `train.py`: This script is used to train a new network on a dataset and save the trained model as a checkpoint.

2. `predict.py`: This script is used to make predictions on new images using a trained model.

3. `model.py`: This module defines the architecture and functionality of the neural network model.

4. `data_preprocessing.py`: This module handles the loading and preprocessing of the image dataset.

5. `checkpoint.py`: This module provides functions to save and load model checkpoints.

6. `image_processing.py`: This module contains functions for processing and transforming images.

## Train Loss

Train loss, also known as training loss or training error, is a metric that measures how well the model is performing on the training data. It indicates the discrepancy between the predicted output of the model and the true labels in the training set. The goal is to minimize the train loss, which means the model is learning to fit the training data better.

## Validation Loss

Validation loss is a metric that measures how well the model is performing on a separate validation dataset that is not used for training. It provides an estimate of the model's generalization performance on unseen data. The goal is to minimize the validation loss, ensuring that the model generalizes well to new data and avoids overfitting.

## Validation Accuracy

Validation accuracy is a metric that measures the model's performance in terms of the percentage of correctly predicted labels in the validation dataset. It provides an intuitive understanding of the model's accuracy in making predictions. The goal is to maximize the validation accuracy, indicating that the model is making accurate predictions on unseen data.

## Getting Started

To use the image classification program, follow these steps:

1. Prepare your dataset by organizing it into separate folders for each class/category.

2. Run `train.py` with the appropriate arguments to train the model on your dataset.

3. Use `predict.py` to make predictions on new images using the trained model.

> Note: The program uses the latest version of torchvision by default. However, if you prefer to use the deprecated version of torchvision, you can uncomment the relevant lines of code in `model.py` and comment out the corresponding lines. Here is an example:

```python
# Load the pre-trained model
# model = models.vgg16(weights=VGG16_Weights.DEFAULT)
model = models.vgg16(pretrained=True)
Make sure to comment out the line model = models.vgg16(weights=VGG16_Weights.DEFAULT) by adding a # at the beginning of the line, and uncomment the line model = models.vgg16(pretrained=True) by removing the # at the beginning of the line.
```


## Dependencies
Python 3.x
PyTorch
torchvision
numpy
matplotlib
PIL
License
This project is licensed under the MIT License.
