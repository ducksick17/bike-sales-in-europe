"""Train, compare and persist regression models that predict order profit."""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_PATH = "models/model.joblib"

CANDIDATE_MODELS = {
    "Linear Regression": LinearRegression(),
    # settings below keep training fast on ~112k rows (dense one-hot + histogram boosting)
    "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
}


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = []
    fitted = {}
    for name, estimator in CANDIDATE_MODELS.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        results.append({"model": name, "R2": r2, "RMSE": rmse, "MAE": mae})
        fitted[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    best_name = results_df.iloc[0]["model"]
    return results_df, fitted, best_name, X_test, y_test


def save_model(pipeline: Pipeline, path: str = MODEL_PATH):
    joblib.dump(pipeline, path)


def load_model(path: str = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict_single(pipeline: Pipeline, customer_age: int, customer_gender: str, country: str,
                   product_category: str, sub_category: str, order_quantity: int,
                   unit_cost: float, unit_price: float, month_number: int, year: int) -> float:
    row = pd.DataFrame([{
        "Customer_Age": customer_age,
        "Customer_Gender": customer_gender,
        "Country": country,
        "Product_Category": product_category,
        "Sub_Category": sub_category,
        "Order_Quantity": order_quantity,
        "Unit_Cost": unit_cost,
        "Unit_Price": unit_price,
        "Month_Number": month_number,
        "Year": year,
    }])
    return float(pipeline.predict(row)[0])
