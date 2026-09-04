# Project 05: Handwritten Digit Image Classifier

Train a neural network to recognize handwritten digits from 0 through 9. Each
image is an 8 x 8 grayscale grid, represented by 64 pixel-intensity features.

## What you will learn

- Represent an image as numerical pixel values
- Understand grayscale intensity, image dimensions, and flattened features
- Split multiclass data with stratification
- Scale pixels before neural-network training
- Compare a majority baseline with a multilayer perceptron (MLP)
- Interpret accuracy, macro F1, confusion matrices, and per-class recall
- Inspect misclassified images instead of relying only on one score
- Save a complete preprocessing-and-model pipeline

## Structure

```text
project-05-digit-image-classifier/
|-- models/
|-- reports/
|-- src/
|   |-- inspect_data.py
|   `-- train.py
|-- .gitignore
|-- requirements.txt
`-- README.md
```

The dataset ships with scikit-learn, so Stage 1 requires no external download.

## Setup

```powershell
cd "F:\Machine Learning\project-05-digit-image-classifier"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Stage 1: Inspect the images

```powershell
python src/inspect_data.py
```

Answer these questions:

1. How many images and classes are present?
2. What is the shape of one image?
3. What do pixel values 0 and 16 represent?
4. Why does the model receive 64 features per image?
5. Is the class distribution approximately balanced?

## Stage 2: Train the neural network

```powershell
python src/train.py
```

The script saves:

- `models/digit_classifier.joblib`;
- `reports/confusion_matrix.png`;
- `reports/misclassified_digits.png`;
- `reports/training_loss.png`.

## Experiments

1. Identify the two digit pairs the model confuses most often.
2. Remove pixel scaling and compare convergence and test performance.
3. Change the hidden layer from `(64,)` to `(128, 64)`.
4. Change regularization `alpha` from `0.0001` to `0.01`.
5. Explain why test accuracy alone cannot reveal which digits are difficult.

## Transfer-learning extension

This project first builds the neural-network foundation without requiring a GPU
or a large framework download. A later extension can use a pretrained CNN on a
larger color-image dataset, where convolution and transfer learning become
meaningful. An 8 x 8 digit dataset is too small for pretrained natural-image
networks to be a sensible choice.

