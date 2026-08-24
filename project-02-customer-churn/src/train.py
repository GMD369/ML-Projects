"""Train and evaluate baseline and logistic-regression churn classifiers."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "telco_customer_churn.csv"
MODEL_PATH = PROJECT_DIR / "models" / "churn_pipeline.joblib"
MATRIX_PATH = PROJECT_DIR / "reports" / "confusion_matrix.png"
RANDOM_STATE = 42


def evaluate(name: str, model: object, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    print(f"\n{name}")
    print(f"  Accuracy:  {accuracy_score(y_test, predictions):.3f}")
    print(f"  Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")
    print(f"  F1:        {f1_score(y_test, predictions, zero_division=0):.3f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, probabilities):.3f}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run: python src/download_data.py"
        )

    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # An ID identifies a customer but describes no reusable customer behavior.
    x = data.drop(columns=["customerID", "Churn"])
    y = data["Churn"].map({"No": 0, "Yes": 1})

    # Stratification preserves the churn proportion in both partitions.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(x_train, y_train)
    evaluate("Majority-class baseline", baseline, x_test, y_test)

    numeric_columns = x_train.select_dtypes(include="number").columns
    categorical_columns = x_train.select_dtypes(exclude="number").columns

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=2000,
                                              class_weight="balanced",)),
        ]
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    print("\nThreshold comparison")
    for threshold in [0.30, 0.50, 0.70]:
        threshold_predictions = (probabilities >= threshold).astype(int)
        print(
            f"  Threshold {threshold:.2f} | "
            f"Precision: {precision_score(y_test, threshold_predictions):.3f} | "
            f"Recall: {recall_score(y_test, threshold_predictions):.3f} | "
            f"F1: {f1_score(y_test, threshold_predictions):.3f}"
        )

    evaluate("Logistic regression", model, x_test, y_test)

    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stay", "Churn"],
        yticklabels=["Stay", "Churn"],
    )
    plt.title("Churn Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(MATRIX_PATH, dpi=150)
    plt.close()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved confusion matrix to: {MATRIX_PATH}")
    print(f"Train rows: {len(x_train):,} | Test rows: {len(x_test):,}")


if __name__ == "__main__":
    main()
