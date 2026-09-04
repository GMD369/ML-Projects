"""Train and evaluate an MLP handwritten-digit classifier."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "digit_classifier.joblib"
MATRIX_PATH = PROJECT_DIR / "reports" / "confusion_matrix.png"
ERRORS_PATH = PROJECT_DIR / "reports" / "misclassified_digits.png"
LOSS_PATH = PROJECT_DIR / "reports" / "training_loss.png"
RANDOM_STATE = 42


def print_metrics(name: str, actual: np.ndarray, predicted: np.ndarray) -> None:
    print(f"\n{name}")
    print(f"  Accuracy: {accuracy_score(actual, predicted):.3f}")
    print(f"  Macro F1: {f1_score(actual, predicted, average='macro'):.3f}")


def main() -> None:
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=digits.target,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    print_metrics("Majority baseline", y_test, baseline.predict(x_test))

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(128,64),
                    activation="relu",
                    solver="adam",
                    alpha=0.01,
                    batch_size=64,
                    learning_rate_init=0.001,
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print_metrics("MLP neural network", y_test, predictions)
    print("\nPer-class test metrics:")
    print(classification_report(y_test, predictions, digits=3))

    classifier = model.named_steps["classifier"]
    print(f"Training iterations: {classifier.n_iter_}")
    print(f"Final training loss: {classifier.loss_:.5f}")

    matrix = confusion_matrix(y_test, predictions)
    fig, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axis)
    axis.set_title("Digit Confusion Matrix")
    axis.set_xlabel("Predicted digit")
    axis.set_ylabel("Actual digit")
    fig.tight_layout()
    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MATRIX_PATH, dpi=150)
    plt.close(fig)

    error_indices = np.flatnonzero(predictions != y_test)
    shown = error_indices[:20]
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, shown):
        axis.imshow(x_test[index].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
        axis.set_title(f"Actual {y_test[index]} / Pred {predictions[index]}")
        axis.axis("off")
    fig.suptitle("Misclassified Test Images")
    fig.tight_layout()
    fig.savefig(ERRORS_PATH, dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    pd.Series(classifier.loss_curve_).plot(ax=axis)
    axis.set_title("Neural Network Training Loss")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(LOSS_PATH, dpi=150)
    plt.close(fig)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nMisclassified test images: {len(error_indices)} / {len(y_test)}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved confusion matrix to: {MATRIX_PATH}")
    print(f"Saved error grid to: {ERRORS_PATH}")
    print(f"Saved loss curve to: {LOSS_PATH}")
    print(f"Train images: {len(x_train):,} | Test images: {len(x_test):,}")


if __name__ == "__main__":
    main()
