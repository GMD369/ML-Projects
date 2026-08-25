"""Download and inspect the Mall Customers dataset."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "mall_customers.csv"
DATA_URLS = [
    (
        "https://raw.githubusercontent.com/sharmaroshan/"
        "Clustering-of-Mall-Customers/master/Mall_Customers.csv"
    ),
    (
        "https://raw.githubusercontent.com/abcom-mltutorials/"
        "mall/main/Mall_Customers.csv"
    ),
]


def download_dataset() -> pd.DataFrame:
    """Try maintained mirrors and report a useful error if all are unavailable."""
    errors: list[str] = []
    for url in DATA_URLS:
        try:
            return pd.read_csv(url)
        except Exception as error:  # Network/HTTP errors differ across platforms.
            errors.append(f"{url}: {error}")

    details = "\n".join(errors)
    raise RuntimeError(f"Could not download the dataset from any mirror:\n{details}")


def main() -> None:
    if DATA_PATH.exists():
        print(f"Dataset already exists: {DATA_PATH}")
        data = pd.read_csv(DATA_PATH)
    else:
        print("Downloading Mall Customers data...")
        data = download_dataset()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(DATA_PATH, index=False)
        print(f"Saved dataset to: {DATA_PATH}")

    print(f"\nShape: {data.shape[0]:,} rows x {data.shape[1]:,} columns")
    print("Target: None (unsupervised learning)")
    print("\nColumns:")
    for column in data.columns:
        print(f"  - {column}: {data[column].dtype}")
    print(f"\nMissing values: {int(data.isna().sum().sum())}")
    print(f"Duplicate customer IDs: {data['CustomerID'].duplicated().sum()}")
    print("\nNumeric summary:")
    print(data.describe().round(2).to_string())


if __name__ == "__main__":
    main()
