"""Inspect and visualize scikit-learn's handwritten digits dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_DIR / "reports" / "sample_digits.png"


def main() -> None:
    digits = load_digits()
    counts = pd.Series(digits.target).value_counts().sort_index()

    print(f"Images: {len(digits.images):,}")
    print(f"Classes: {len(digits.target_names)} ({digits.target_names.tolist()})")
    print(f"Image shape: {digits.images[0].shape}")
    print(f"Flattened features per image: {digits.data.shape[1]}")
    print(
        f"Pixel intensity range: {digits.images.min():.0f} "
        f"to {digits.images.max():.0f}"
    )
    print("\nClass distribution:")
    print(counts.to_string())

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for digit, axis in enumerate(axes.flat):
        index = int((digits.target == digit).nonzero()[0][0])
        axis.imshow(digits.images[index], cmap="gray_r", vmin=0, vmax=16)
        axis.set_title(f"Label: {digit}")
        axis.axis("off")
    fig.suptitle("One Example of Each Digit")
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"\nSaved sample grid to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

