"""Forecast evaluation and diagnostic analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_EVALUATION_COLUMNS = {
    "date",
    "stock_code",
    "demand",
    "prediction",
}


def _validate_evaluation_data(
    test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and standardize forecast evaluation data."""

    if test_predictions.empty:
        raise ValueError(
            "Test predictions must not be empty."
        )

    missing_columns = (
        REQUIRED_EVALUATION_COLUMNS.difference(
            test_predictions.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Test predictions are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    data = test_predictions.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data["demand"] = pd.to_numeric(
        data["demand"],
        errors="raise",
    )

    data["prediction"] = pd.to_numeric(
        data["prediction"],
        errors="raise",
    )

    if data[
        sorted(REQUIRED_EVALUATION_COLUMNS)
    ].isna().any().any():
        raise ValueError(
            "Test predictions contain missing values "
            "in required columns."
        )

    if not np.isfinite(
        data["demand"]
    ).all():
        raise ValueError(
            "Actual demand must contain only finite values."
        )

    if not np.isfinite(
        data["prediction"]
    ).all():
        raise ValueError(
            "Predictions must contain only finite values."
        )

    if (
        data["demand"] < 0
    ).any():
        raise ValueError(
            "Actual demand must be non-negative."
        )

    if (
        data["prediction"] < 0
    ).any():
        raise ValueError(
            "Predictions must be non-negative."
        )

    return (
        data
        .sort_values(
            [
                "date",
                "stock_code",
            ]
        )
        .reset_index(drop=True)
    )


def build_error_table(
    test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build row-level forecast error diagnostics."""

    data = _validate_evaluation_data(
        test_predictions
    )

    data["error"] = (
        data["prediction"]
        - data["demand"]
    )

    data["absolute_error"] = (
        data["error"].abs()
    )

    data["squared_error"] = (
        data["error"] ** 2
    )

    data["absolute_percentage_error"] = np.nan

    nonzero_demand = (
        data["demand"] > 0
    )

    data.loc[
        nonzero_demand,
        "absolute_percentage_error",
    ] = (
        data.loc[
            nonzero_demand,
            "absolute_error",
        ]
        / data.loc[
            nonzero_demand,
            "demand",
        ]
        * 100.0
    )

    return data


def evaluate_by_product(
    test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate forecast performance metrics for each product."""

    error_table = build_error_table(
        test_predictions
    )

    product_metrics = (
        error_table
        .groupby(
            "stock_code",
            as_index=False,
        )
        .agg(
            observations=(
                "demand",
                "size",
            ),
            MAE=(
                "absolute_error",
                "mean",
            ),
            MSE=(
                "squared_error",
                "mean",
            ),
            MAPE_percent=(
                "absolute_percentage_error",
                "mean",
            ),
            bias=(
                "error",
                "mean",
            ),
        )
    )

    product_metrics["RMSE"] = np.sqrt(
        product_metrics["MSE"]
    )

    product_metrics = product_metrics.drop(
        columns="MSE"
    )

    return (
        product_metrics
        .sort_values(
            [
                "RMSE",
                "stock_code",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def summarize_evaluation(
    test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create a one-row summary of overall forecast performance."""

    error_table = build_error_table(
        test_predictions
    )

    observations = len(
        error_table
    )

    products = int(
        error_table[
            "stock_code"
        ].nunique()
    )

    start_date = error_table[
        "date"
    ].min()

    end_date = error_table[
        "date"
    ].max()

    mae = float(
        error_table[
            "absolute_error"
        ].mean()
    )

    rmse = float(
        np.sqrt(
            error_table[
                "squared_error"
            ].mean()
        )
    )

    mape_percent = float(
        error_table[
            "absolute_percentage_error"
        ].mean()
    )

    bias = float(
        error_table[
            "error"
        ].mean()
    )

    zero_demand_observations = int(
        (
            error_table["demand"] == 0
        ).sum()
    )

    error_values = error_table[
        "error"
    ].to_numpy(
        dtype=float
    )

    tolerance = 1e-9

    over_forecast_mask = (
        error_values > tolerance
    )

    under_forecast_mask = (
        error_values < -tolerance
    )

    exact_forecast_mask = (
        np.abs(error_values)
        <= tolerance
    )

    over_forecast_count = int(
        over_forecast_mask.sum()
    )

    under_forecast_count = int(
        under_forecast_mask.sum()
    )

    exact_forecast_count = int(
        exact_forecast_mask.sum()
    )

    over_forecast_rate = (
        over_forecast_count
        / observations
    )

    under_forecast_rate = (
        under_forecast_count
        / observations
    )

    exact_forecast_rate = (
        exact_forecast_count
        / observations
    )

    return pd.DataFrame(
        [
            {
                "observations":
                    observations,

                "products":
                    products,

                "start_date":
                    start_date,

                "end_date":
                    end_date,

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "MAPE_percent":
                    mape_percent,

                "bias":
                    bias,

                "zero_demand_observations":
                    zero_demand_observations,

                "over_forecast_count":
                    over_forecast_count,

                "under_forecast_count":
                    under_forecast_count,

                "exact_forecast_count":
                    exact_forecast_count,

                "over_forecast_rate":
                    over_forecast_rate,

                "under_forecast_rate":
                    under_forecast_rate,

                "exact_forecast_rate":
                    exact_forecast_rate,
            }
        ]
    )