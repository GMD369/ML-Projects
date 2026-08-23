"""Download and briefly inspect the Ames Housing regression dataset."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "ames_housing.csv"


def main() -> None:
    if DATA_PATH.exists():
        print(f"Dataset already exists: {DATA_PATH}")
        frame = pd.read_csv(DATA_PATH)
    else:
        print("Downloading Ames Housing from OpenML...")
        dataset = fetch_openml(name="house_prices", as_frame=True, parser="auto")
        frame = dataset.frame
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(DATA_PATH, index=False)
        print(f"Saved dataset to: {DATA_PATH}")

    missing = frame.isna().sum().sort_values(ascending=False).head(10)
    print(f"\nShape: {frame.shape[0]:,} rows x {frame.shape[1]:,} columns")
    print("Target: SalePrice")
    print("\nSalePrice summary:")
    print(frame["SalePrice"].describe().round(2).to_string())
    print("\nColumns with the most missing values:")
    print(missing.to_string())


if __name__ == "__main__":
    main()

