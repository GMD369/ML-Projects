"""Download and inspect the UCI SMS Spam Collection dataset."""

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "sms_spam.csv"
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_dataset() -> pd.DataFrame:
    with urlopen(DATA_URL, timeout=60) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        with archive.open("SMSSpamCollection") as source:
            return pd.read_csv(
                source,
                sep="\t",
                header=None,
                names=["label", "message"],
                encoding="utf-8",
            )


def main() -> None:
    if DATA_PATH.exists():
        print(f"Dataset already exists: {DATA_PATH}")
        data = pd.read_csv(DATA_PATH)
    else:
        print("Downloading the UCI SMS Spam Collection...")
        data = download_dataset()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(DATA_PATH, index=False)
        print(f"Saved dataset to: {DATA_PATH}")

    counts = data["label"].value_counts()
    rates = data["label"].value_counts(normalize=True).mul(100)
    lengths = data.assign(characters=data["message"].str.len()).groupby("label")[
        "characters"
    ].agg(["mean", "median"])

    print(f"\nShape: {data.shape[0]:,} rows x {data.shape[1]:,} columns")
    print("Target: label")
    print("\nClass distribution:")
    for label in counts.index:
        print(f"  {label}: {counts[label]:,} ({rates[label]:.2f}%)")
    print("\nMessage length by class (characters):")
    print(lengths.round(1).to_string())
    print(f"\nMissing values: {int(data.isna().sum().sum())}")
    print(f"Exact duplicate rows: {data.duplicated().sum()}")


if __name__ == "__main__":
    main()

