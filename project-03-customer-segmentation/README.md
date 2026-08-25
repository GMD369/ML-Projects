# Project 03: Customer Segmentation

Group shopping-mall customers by income and spending behavior using K-Means.
This is **unsupervised learning**: there is no correct target column telling the
model which segment each customer belongs to.

## What you will learn

- Understand unsupervised learning and clustering
- Explore customer behavior without a target variable
- Scale features before distance-based modeling
- Select `k` using inertia and silhouette score
- Fit K-Means and visualize its clusters
- Profile clusters and translate them into business segments
- Recognize the limits of subjective cluster interpretation

## Structure

```text
project-03-customer-segmentation/
|-- data/
|-- models/
|-- reports/
|-- src/
|   |-- download_data.py
|   `-- train.py
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Setup

```powershell
cd "F:\Machine Learning\project-03-customer-segmentation"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Stage 1: Download and inspect

```powershell
python src/download_data.py
```

Answer these questions:

1. How many customers and columns are present?
2. Which column is only an identifier?
3. Is there a target variable in this project?
4. What do annual income and spending score represent?
5. Why might raw Euclidean distance unfairly emphasize one feature?

## Stage 2: Compare possible cluster counts

```powershell
python src/train.py
```

The script standardizes annual income and spending score, compares values of
`k` from 2 through 10, and trains a five-cluster model. It creates:

- `reports/k_selection.png`: inertia and silhouette scores;
- `reports/customer_segments.png`: the customer clusters;
- `reports/cluster_profiles.csv`: the average customer in each cluster;
- `models/customer_segmentation.joblib`: scaler and K-Means artifacts.

## Experiments

1. Explain why K-Means needs a chosen value of `k`.
2. Select `k` using both the elbow plot and silhouette score.
3. Give every resulting cluster a business-friendly name.
4. Add `Age` as a third clustering feature and compare silhouette scores.
5. Explain why cluster numbers such as 0 and 3 have no natural ranking.

