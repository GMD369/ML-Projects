from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "ames_housing.csv"
OUTPUT_PATH = PROJECT_DIR / "reports" / "sale_price_distribution.png"

data = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")
sns.histplot(data=data, x="SalePrice", bins=40, kde=True)

plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale price ($)")
plt.ylabel("Number of houses")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
plt.show()

print(f"Saved chart to: {OUTPUT_PATH}")