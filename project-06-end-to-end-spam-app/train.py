"""Train, evaluate, and package the spam model for the web application."""

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "sms_spam.csv"
MODEL_PATH = PROJECT_DIR / "models" / "spam_pipeline.joblib"
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RANDOM_STATE = 42
THRESHOLD = 0.50


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    print("Downloading the UCI SMS Spam Collection...")
    with urlopen(DATA_URL, timeout=60) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        with archive.open("SMSSpamCollection") as source:
            data = pd.read_csv(
                source,
                sep="\t",
                header=None,
                names=["label", "message"],
                encoding="utf-8",
            )
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False)
    return data


def main() -> None:
    data = load_data()
    duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    x = data["message"]
    y = data["label"].map({"ham": 0, "spam": 1})
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    min_df=2,
                    ngram_range=(1, 1),
                    sublinear_tf=True,
                ),
            ),
            ("classifier", MultinomialNB(alpha=0.1)),
        ]
    )
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    print(f"Removed duplicates: {duplicates:,}")
    print(f"Training messages: {len(x_train):,}")
    print(f"Test messages: {len(x_test):,}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.3f}")
    print(f"Precision: {precision_score(y_test, predictions):.3f}")
    print(f"Recall:    {recall_score(y_test, predictions):.3f}")
    print(f"F1:        {f1_score(y_test, predictions):.3f}")

    artifact = {
        "pipeline": pipeline,
        "threshold": THRESHOLD,
        "model_version": "1.0.0",
        "classes": {0: "ham", 1: "spam"},
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved deployable model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

