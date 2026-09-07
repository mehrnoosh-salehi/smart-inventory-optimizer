"""Demand forecasting models, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

TARGET_COLUMN = "demand"

REQUIRED_CATEGORICAL_FEATURES = [
    "stock_code",
]

OPTIONAL_CATEGORICAL_FEATURES = [
    "category",
]

BASE_NUMERIC_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
    "rolling_std_4",
    "rolling_mean_8",
    "rolling_std_8",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "sin_week",
    "cos_week",
]

OPTIONAL_NUMERIC_FEATURES = [
    "price_to_cost_ratio_lag_1",
    "promotion_days_lag_1",
]


@dataclass
class ForecastingResult:
    """Store the trained forecasting model and its evaluation artifacts."""

    best_model_name: str
    best_model: Pipeline
    metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]


def temporal_train_test_split(
    df: pd.DataFrame,
    test_weeks: int = 8,
    min_train_weeks: int = 52,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split forecasting data chronologically into training and test sets.

    The most recent unique weeks are reserved for testing, while a minimum
    amount of historical data is required for model training.
    """

    required_columns = {
        "date",
        "stock_code",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if test_weeks <= 0:
        raise ValueError(
            "test_weeks must be greater than zero."
        )

    if min_train_weeks <= 0:
        raise ValueError(
            "min_train_weeks must be greater than zero."
        )

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data = (
        data
        .sort_values(["date", "stock_code"])
        .reset_index(drop=True)
    )

    unique_dates = (
        data["date"]
        .drop_duplicates()
        .sort_values()
        .to_list()
    )

    required_weeks = min_train_weeks + test_weeks

    if len(unique_dates) < required_weeks:
        raise ValueError(
            f"At least {required_weeks} unique weeks are required: "
            f"{min_train_weeks} for training and "
            f"{test_weeks} for testing. "
            f"Only {len(unique_dates)} unique weeks were provided."
        )

    cutoff_date = unique_dates[-test_weeks]

    train = (
        data
        .loc[data["date"] < cutoff_date]
        .copy()
    )

    test = (
        data
        .loc[data["date"] >= cutoff_date]
        .copy()
    )

    return train, test


def _resolve_model_features(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Resolve required and optional features available for modeling."""

    missing_numeric_features = set(
        BASE_NUMERIC_FEATURES
    ).difference(df.columns)

    if missing_numeric_features:
        raise ValueError(
            f"Missing required numeric features: "
            f"{sorted(missing_numeric_features)}"
        )

    missing_categorical_features = set(
        REQUIRED_CATEGORICAL_FEATURES
    ).difference(df.columns)

    if missing_categorical_features:
        raise ValueError(
            f"Missing required categorical features: "
            f"{sorted(missing_categorical_features)}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Input data must contain target column "
            f"'{TARGET_COLUMN}'."
        )

    numeric_features = BASE_NUMERIC_FEATURES.copy()

    numeric_features.extend(
        feature
        for feature in OPTIONAL_NUMERIC_FEATURES
        if feature in df.columns
    )

    categorical_features = REQUIRED_CATEGORICAL_FEATURES.copy()

    categorical_features.extend(
        feature
        for feature in OPTIONAL_CATEGORICAL_FEATURES
        if feature in df.columns
    )

    return numeric_features, categorical_features


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build preprocessing steps for numeric and categorical model features."""

    if scale_numeric:
        numeric_transformer = StandardScaler()
    else:
        numeric_transformer = "passthrough"

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def _build_model_pipelines(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = RANDOM_STATE,
) -> dict[str, Pipeline]:
    """Build the forecasting model pipelines."""

    ridge_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                _build_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    scale_numeric=True,
                ),
            ),
            (
                "model",
                Ridge(
                    alpha=1.0,
                ),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                _build_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    scale_numeric=False,
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    min_samples_leaf=3,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    gradient_boosting_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                _build_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    scale_numeric=False,
                ),
            ),
            (
                "model",
                GradientBoostingRegressor(
                    random_state=random_state,
                ),
            ),
        ]
    )

    models = {
        "ridge_regression": ridge_pipeline,
        "random_forest": random_forest_pipeline,
        "gradient_boosting": gradient_boosting_pipeline,
    }

    return models


def _mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate MAPE while excluding observations with zero actual demand."""

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    nonzero_mask = y_true != 0

    if not np.any(nonzero_mask):
        return float("nan")

    percentage_errors = np.abs(
        (
            y_true[nonzero_mask]
            - y_pred[nonzero_mask]
        )
        / y_true[nonzero_mask]
    )

    return float(
        np.mean(percentage_errors) * 100
    )


def _forecast_bias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate mean forecast bias as prediction minus actual demand."""

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    return float(
        np.mean(y_pred - y_true)
    )


def train_and_evaluate_models(
    supervised: pd.DataFrame,
    test_weeks: int = 8,
    min_train_weeks: int = 52,
    random_state: int = RANDOM_STATE,
) -> ForecastingResult:
    """Train and evaluate forecasting models using a temporal holdout."""

    train_df, test_df = temporal_train_test_split(
        supervised,
        test_weeks=test_weeks,
        min_train_weeks=min_train_weeks,
    )

    numeric_features, categorical_features = (
        _resolve_model_features(train_df)
    )

    model_features = [
        *numeric_features,
        *categorical_features,
    ]

    X_train = train_df[
        model_features
    ].copy()

    y_train = train_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float,
    )

    X_test = test_df[
        model_features
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float,
    )

    models = _build_model_pipelines(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
    )

    metric_rows: list[dict] = []

    test_predictions = test_df[
        [
            "date",
            "stock_code",
            TARGET_COLUMN,
        ]
    ].copy()

    baseline_predictions = test_df[
        "rolling_mean_4"
    ].to_numpy(
        dtype=float,
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    baseline_rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                baseline_predictions,
            )
        )
    )

    baseline_mape = _mape(
        y_test,
        baseline_predictions,
    )

    baseline_bias = _forecast_bias(
        y_test,
        baseline_predictions,
    )

    metric_rows.append(
        {
            "model": "moving_average_4",
            "MAE": baseline_mae,
            "RMSE": baseline_rmse,
            "MAPE_percent": baseline_mape,
            "bias": baseline_bias,
        }
    )

    test_predictions[
        "pred_moving_average_4"
    ] = baseline_predictions

    best_model_name: str | None = None
    best_model: Pipeline | None = None
    best_rmse = float("inf")

    for model_name, model in models.items():
        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        predictions = np.maximum(
            predictions,
            0.0,
        )

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )
        )

        mape = _mape(
            y_test,
            predictions,
        )

        bias = _forecast_bias(
            y_test,
            predictions,
        )

        metric_rows.append(
            {
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE_percent": mape,
                "bias": bias,
            }
        )

        test_predictions[
            f"pred_{model_name}"
        ] = predictions

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = model_name
            best_model = model

    metrics = (
        pd.DataFrame(metric_rows)
        .sort_values(
            by="RMSE",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    if best_model is None or best_model_name is None:
        raise RuntimeError(
            "No trainable forecasting model was successfully selected."
        )

    best_prediction_column = (
        f"pred_{best_model_name}"
    )

    test_predictions["best_model"] = (
        best_model_name
    )

    test_predictions["prediction"] = (
        test_predictions[
            best_prediction_column
        ]
    )

    # Refit the selected ML pipeline on all available supervised data.
    # The temporal holdout above is used only for unbiased model evaluation
    # and selection. Once the winning algorithm is selected, all known
    # observations can be used to train the final model for future inference.
    X_full = supervised[
        model_features
    ].copy()

    y_full = supervised[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float,
    )

    best_model.fit(
        X_full,
        y_full,
    )

    return ForecastingResult(
        best_model_name=best_model_name,
        best_model=best_model,
        metrics=metrics,
        test_predictions=test_predictions,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def predict_next_period(
    result: ForecastingResult,
    next_features: pd.DataFrame,
) -> pd.DataFrame:
    """Predict next-period demand using the selected forecasting pipeline."""

    model_features = [
        *result.numeric_features,
        *result.categorical_features,
    ]

    missing_features = set(
        model_features
    ).difference(next_features.columns)

    if missing_features:
        raise ValueError(
            f"Next-period data is missing model features: "
            f"{sorted(missing_features)}"
        )

    X_next = next_features[
        model_features
    ].copy()

    predictions = result.best_model.predict(
        X_next
    )

    predictions = np.maximum(
        predictions,
        0.0,
    )

    metadata_columns = [
        column
        for column in [
            "date",
            "stock_code",
            "description",
            "category",
            "unit_price",
            "unit_cost",
        ]
        if column in next_features.columns
    ]

    output = next_features[
        metadata_columns
    ].copy()

    output["predicted_demand"] = predictions

    output["model"] = (
        result.best_model_name
    )

    return (
        output
        .sort_values(
            "predicted_demand",
            ascending=False,
        )
        .reset_index(drop=True)
    )
