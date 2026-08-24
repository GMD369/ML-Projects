"""Download and inspect the IBM Telco Customer Churn dataset."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "telco_customer_churn.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)


def main() -> None:
    if DATA_PATH.exists():
        print(f"Dataset already exists: {DATA_PATH}")
        data = pd.read_csv(DATA_PATH)
    else:
        print("Downloading IBM Telco Customer Churn data...")
        data = pd.read_csv(DATA_URL)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(DATA_PATH, index=False)
        print(f"Saved dataset to: {DATA_PATH}")

    churn_counts = data["Churn"].value_counts()
    churn_rates = data["Churn"].value_counts(normalize=True).mul(100)
    blank_total_charges = data["TotalCharges"].astype(str).str.strip().eq("").sum()

    print(f"\nShape: {data.shape[0]:,} rows x {data.shape[1]:,} columns")
    print("Target: Churn")
    print("\nTarget distribution:")
    for label in churn_counts.index:
        print(f"  {label}: {churn_counts[label]:,} ({churn_rates[label]:.2f}%)")
    print(f"\nBlank TotalCharges values: {blank_total_charges}")
    print(f"Duplicate customer IDs: {data['customerID'].duplicated().sum()}")


if __name__ == "__main__":
    main()

