# Project 02: Customer Churn Prediction

Predict whether a telecom customer will leave the company. Unlike Project 1,
this is a **binary classification** problem: the target is either `Yes` or `No`.

## What you will learn

- Understand classification targets and class imbalance
- Split data with stratification
- Preprocess numeric and categorical columns in one pipeline
- Compare a majority-class baseline with logistic regression
- Interpret accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix
- Change the decision threshold based on business priorities
- Save a reusable trained pipeline

## Structure

```text
project-02-customer-churn/
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

You may create a separate environment for this project:

```powershell
cd "F:\Machine Learning\project-02-customer-churn"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Stage 1: Download and inspect

```powershell
python src/download_data.py
```

Answer these questions before training:

1. How many customers and columns are present?
2. What percentage of customers churned?
3. Is the target balanced?
4. Why is `customerID` unsuitable as a predictive feature?
5. Why must the blank values in `TotalCharges` become missing numeric values?

## Stage 2: Train the first classifier

```powershell
python src/train.py
```

The script compares a majority-class baseline with logistic regression. It saves
the trained pipeline and a confusion-matrix image.

Do not evaluate churn using accuracy alone. A model can achieve high accuracy by
mostly predicting that customers will stay while failing to identify customers
who are actually leaving.

## Experiments

1. Run the baseline and logistic-regression model.
2. Explain false positives and false negatives in business language.
3. Add `class_weight="balanced"` to logistic regression and compare recall.
4. Try decision thresholds of `0.30`, `0.50`, and `0.70`.
5. Decide which metric matters most if contacting a customer is cheap.

