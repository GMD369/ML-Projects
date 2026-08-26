"""Tune Multinomial Naive Bayes smoothing using training-fold CV."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "sms_spam.csv"
RANDOM_STATE = 42


def main() -> None:
    data = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    x = data["message"]
    y = data["label"].map({"ham": 0, "spam": 1})
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Restore the strongest vectorizer before tuning one model parameter.
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
            ("classifier", MultinomialNB()),
        ]
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"classifier__alpha": [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]},
        scoring="f1",
        cv=5,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(x_train, y_train)

    results = pd.DataFrame(search.cv_results_)[
        ["param_classifier__alpha", "mean_test_score", "std_test_score"]
    ].sort_values("param_classifier__alpha")
    print("Cross-validation results (training folds only):")
    print(results.round(3).to_string(index=False))
    print(f"\nBest alpha: {search.best_params_['classifier__alpha']}")
    print(f"Best mean CV F1: {search.best_score_:.3f}")

    predictions = search.best_estimator_.predict(x_test)
    print("\nHeld-out test result for the selected model:")
    print(f"  Precision: {precision_score(y_test, predictions):.3f}")
    print(f"  Recall:    {recall_score(y_test, predictions):.3f}")
    print(f"  F1:        {f1_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()
