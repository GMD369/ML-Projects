"""Train and evaluate an SMS spam text-classification pipeline."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "sms_spam.csv"
MODEL_PATH = PROJECT_DIR / "models" / "spam_pipeline.joblib"
MATRIX_PATH = PROJECT_DIR / "reports" / "confusion_matrix.png"
ERRORS_PATH = PROJECT_DIR / "reports" / "misclassified_messages.csv"
RANDOM_STATE = 42


def evaluate(name: str, model: object, x_test: pd.Series, y_test: pd.Series) -> None:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    print(f"\n{name}")
    print(f"  Accuracy:       {accuracy_score(y_test, predictions):.3f}")
    print(
        f"  Spam precision: "
        f"{precision_score(y_test, predictions, zero_division=0):.3f}"
    )
    print(f"  Spam recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")
    print(f"  Spam F1:        {f1_score(y_test, predictions, zero_division=0):.3f}")
    print(f"  ROC-AUC:        {roc_auc_score(y_test, probabilities):.3f}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run: python src/download_data.py"
        )

    data = pd.read_csv(DATA_PATH)
    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    print(
        f"Removed {duplicate_count:,} exact duplicate rows before splitting "
        f"({len(data):,} unique messages remain)."
    )
    x = data["message"]
    y = data["label"].map({"ham": 0, "spam": 1})
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(x_train.to_frame(), y_train)
    evaluate("Majority baseline", baseline, x_test.to_frame(), y_test)

    model = Pipeline(
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
    model.fit(x_train, y_train)
    evaluate("TF-IDF + Multinomial Naive Bayes", model, x_test, y_test)

    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
    )
    plt.title("SMS Spam Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(MATRIX_PATH, dpi=150)
    plt.close()

    errors = pd.DataFrame(
        {
            "message": x_test,
            "actual": y_test.map({0: "ham", 1: "spam"}),
            "predicted": pd.Series(predictions, index=x_test.index).map(
                {0: "ham", 1: "spam"}
            ),
            "spam_probability": model.predict_proba(x_test)[:, 1],
        }
    )
    errors = errors[errors["actual"] != errors["predicted"]].sort_values(
        "spam_probability", ascending=False
    )
    errors.to_csv(ERRORS_PATH, index=False)

    vocabulary_size = len(model.named_steps["tfidf"].vocabulary_)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nVocabulary size: {vocabulary_size:,}")
    print(f"Misclassified messages: {len(errors):,}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved confusion matrix to: {MATRIX_PATH}")
    print(f"Saved error analysis to: {ERRORS_PATH}")
    print(f"Train rows: {len(x_train):,} | Test rows: {len(x_test):,}")


if __name__ == "__main__":
    main()
