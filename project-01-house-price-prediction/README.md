# Project 01: House Price Prediction

Build a regression model that predicts house sale prices from the Ames Housing
dataset. The project is intentionally split into small stages so you understand
the workflow instead of only running a finished model.

## What you will learn

- Frame a business problem as supervised regression
- Explore numerical and categorical features
- Prevent data leakage with train/test splitting and pipelines
- Handle missing values and encode categories
- Establish a baseline and train a stronger model
- Evaluate predictions with MAE, RMSE, and R-squared
- Save a trained pipeline for later use

## Project structure

```text
project-01-house-price-prediction/
|-- data/                  # Downloaded data (not committed)
|-- models/                # Saved trained models
|-- notebooks/             # Your experiments and notes
|-- reports/               # Charts and written findings
|-- src/
|   |-- download_data.py   # Downloads Ames Housing from OpenML
|   `-- train.py           # Baseline and first ML pipeline
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Setup

Open PowerShell in this directory and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the environment's Python directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Stage 1: Get and inspect the data

```powershell
python src/download_data.py
```

This downloads the Ames Housing dataset from OpenML and writes
`data/ames_housing.csv`. It requires internet only on the first run.

Before training, answer these questions using the printed summary or your own
notebook:

1. How many rows and input features are present?
2. What is the target column, and what does it represent?
3. Which five columns contain the most missing values?
4. Why should `SalePrice` not be included among the input features?

## Stage 2: Train the first models

```powershell
python src/train.py
```

The script compares:

- a **median baseline**, which predicts the same typical price for every house;
- a **Random Forest pipeline**, which preprocesses mixed data and learns useful
  patterns.

Record the resulting metrics in `reports/experiment_notes.md`. A useful model
should beat the baseline MAE and RMSE on unseen test data.

## Your first challenges

Complete these in order:

1. Run both scripts and record the metrics.
2. Open `src/train.py` and identify where data leakage is prevented.
3. Change `n_estimators` from `200` to `400`; measure whether performance and
   training time improve.
4. Try `max_depth=15`, then explain whether limiting tree depth helps.
5. Add one visualization showing the distribution of `SalePrice`.

Do not judge a model using training accuracy alone. The held-out test set is the
important measurement in this project.

