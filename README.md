# Digit Classification Pipeline (MNIST)

A complete end-to-end handwritten digit classification system built using the MNIST dataset. This project includes a custom IDX file loader, a CNN built from scratch in PyTorch, a full training and evaluation pipeline, and a command-line inference script for predicting digits from images.

## Features

* Custom binary loader for MNIST .idx files (images + labels)
* Convolutional Neural Network implemented from scratch
* Training loop with loss tracking, accuracy metrics, and progress bars
* Best model checkpointing (`best_model.pth`)
* Command-line pipeline to classify any input digit image
* Achieved 98.81% accuracy on the MNIST test set

## Project Structure

```
├── MNIST_Dataset/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── best_model.pth
├── pipeline.py
├── train.ipynb
├── requirements.txt
└── README.md
```

## Dataset

MNIST is provided in the IDX binary file format. A custom loader parses the header using Python's `struct` module and reshapes the remaining byte stream into tensors of shape:

```
(N, 1, 28, 28)
```

This provides full control over preprocessing and data handling without relying on external libraries like `torchvision`.

## Model Architecture

The CNN used for classification:

* Conv2d: 1 → 32, kernel 3×3, ReLU
* MaxPool2d: 2×2
* Conv2d: 32 → 64, kernel 3×3, ReLU
* MaxPool2d: 2×2
* Flatten
* Fully Connected: 64×7×7 → 128
* Fully Connected: 128 → 10

## Training and Evaluation

Training is conducted in `train.ipynb`.

* Loss function: CrossEntropyLoss
* Optimizer: Adam (lr = 0.001)
* Metrics: Accuracy per epoch
* Model saving: Best-performing model automatically saved as `best_model.pth`

Evaluation is also handled in the notebook. Final test accuracy: 98.81%

## Inference Pipeline (`pipeline.py`)

`pipeline.py` performs the following steps:

* Loads the trained model (`best_model.pth`)
* Reads an image from a provided file path
* Converts to grayscale, resizes to 28×28, normalizes, and reshapes
* Runs the CNN to predict the digit
* Prints the predicted class

Example usage:

```
python pipeline.py --image path/to/digit.png
```

## Installation and How to Run

### 1. Create a virtual environment

```
python -m venv venv
```

### 2. Activate the environment

```
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. (Optional) Train the model yourself

Modify dataset paths if needed in `train.ipynb`.

### 5. Run inference

```
python pipeline.py --image path/to/digit.png
```

## Results

* Test Accuracy: 98.81%
* Best model saved as: `best_model.pth`
* Command-line prediction implemented

## Future Improvements

* Add a Streamlit web application for interactive digit prediction
* Add data augmentation for improved robustness
* Experiment with deeper architectures (LeNet-5, ResNet-like blocks)
* Move training pipeline into standalone `train.py` for improved modularity
