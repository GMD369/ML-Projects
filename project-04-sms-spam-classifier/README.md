# Project 04: SMS Spam Classifier

Build a natural-language-processing classifier that labels an SMS message as
`ham` (legitimate) or `spam`.

## What you will learn

- Turn raw text into numerical TF-IDF features
- Understand tokens, n-grams, document frequency, and vocabulary size
- Split imbalanced text data with stratification
- Compare a majority baseline with Multinomial Naive Bayes
- Evaluate spam precision, recall, F1, ROC-AUC, and a confusion matrix
- Inspect false positives and false negatives
- Save a complete text-to-prediction pipeline

## Structure

```text
project-04-sms-spam-classifier/
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
cd "F:\Machine Learning\project-04-sms-spam-classifier"
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

1. How many messages are in the dataset?
2. What percentage are spam?
3. Why would accuracy alone be misleading?
4. Are spam messages typically longer than legitimate messages?
5. What kinds of words do you expect to indicate spam?

## Stage 2: Train the classifier

```powershell
python src/train.py
```

The script compares a majority baseline with a TF-IDF + Multinomial Naive Bayes
pipeline. It saves:

- `models/spam_pipeline.joblib`;
- `reports/confusion_matrix.png`;
- `reports/misclassified_messages.csv`.

## Experiments

1. Interpret every cell in the confusion matrix.
2. Inspect the incorrectly classified messages for patterns.
3. Add bigrams with `ngram_range=(1, 2)` and compare results.
4. Change `min_df` from 2 to 1 and compare vocabulary size and metrics.
5. Tune the Naive Bayes `alpha` value using cross-validation.

Run the supplied cross-validation experiment with:

```powershell
python src/tune_alpha.py
```

For spam filtering, false positives can be especially costly: an important real
message may be hidden from the user.
