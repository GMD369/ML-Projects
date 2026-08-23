"""Train and evaluate baseline and random-forest house-price models."""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "ames_housing.csv"
MODEL_PATH = PROJECT_DIR / "models" / "house_price_pipeline.joblib"
RANDOM_STATE = 42


def evaluate(name: str, model: object, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"\n{name}")
    print(f"  MAE:       ${mae:,.0f}")
    print(f"  RMSE:      ${rmse:,.0f}")
    print(f"  R-squared: {r2:.3f}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run: python src/download_data.py"
        )

    frame = pd.read_csv(DATA_PATH)
    x = frame.drop(columns="SalePrice")
    y = frame["SalePrice"]

    # Splitting before fitting preprocessing prevents information from the test
    # set leaking into imputation, encoding, or model training.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE
    )

    baseline = DummyRegressor(strategy="median")
    baseline.fit(x_train, y_train)
    evaluate("Median baseline", baseline, x_test, y_test)

    numeric_columns = x_train.select_dtypes(include="number").columns
    categorical_columns = x_train.select_dtypes(exclude="number").columns

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
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
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    started = perf_counter()
    model.fit(x_train, y_train)
    elapsed = perf_counter() - started
    evaluate("Random forest", model, x_test, y_test)
    print(f"  Fit time:   {elapsed:.2f} seconds")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")
    print(f"Train rows: {len(x_train):,} | Test rows: {len(x_test):,}")


if __name__ == "__main__":
    main()
