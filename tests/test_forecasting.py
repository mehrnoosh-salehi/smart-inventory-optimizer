import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    _forecast_bias,
    _mape,
    _resolve_model_features,
    predict_next_period,
    temporal_train_test_split,
    train_and_evaluate_models,
)


def _make_supervised_data(
    n_weeks: int = 12,
) -> pd.DataFrame:
    """Create deterministic supervised forecasting data for unit tests."""

    dates = pd.date_range(
        start="2025-01-06",
        periods=n_weeks,
        freq="W-MON",
    )

    products = [
        ("SKU-001", "Home", 100.0),
        ("SKU-002", "Office", 180.0),
    ]

    rows = []

    for week_index, date in enumerate(
        dates,
        start=1,
    ):
        week_of_year = int(
            date.isocalendar().week
        )

        for stock_code, category, base_demand in products:
            demand = (
                base_demand
                + 2.0 * week_index
            )

            previous_4 = np.array(
                [
                    demand - 2.0,
                    demand - 4.0,
                    demand - 6.0,
                    demand - 8.0,
                ]
            )

            previous_8 = np.array(
                [
                    demand - 2.0 * lag
                    for lag in range(1, 9)
                ]
            )

            rows.append(
                {
                    "date": date,
                    "stock_code": stock_code,
                    "category": category,
                    "demand": demand,
                    "lag_1": demand - 2.0,
                    "lag_2": demand - 4.0,
                    "lag_4": demand - 8.0,
                    "lag_8": demand - 16.0,
                    "rolling_mean_4": float(
                        previous_4.mean()
                    ),
                    "rolling_std_4": float(
                        previous_4.std(ddof=1)
                    ),
                    "rolling_mean_8": float(
                        previous_8.mean()
                    ),
                    "rolling_std_8": float(
                        previous_8.std(ddof=1)
                    ),
                    "week_of_year": week_of_year,
                    "month": date.month,
                    "quarter": date.quarter,
                    "year": date.year,
                    "sin_week": np.sin(
                        2 * np.pi * week_of_year / 52
                    ),
                    "cos_week": np.cos(
                        2 * np.pi * week_of_year / 52
                    ),
                    "price_to_cost_ratio_lag_1": 2.0,
                    "promotion_days_lag_1": 0.0,
                }
            )

    return pd.DataFrame(rows)


def test_temporal_split_keeps_latest_weeks_in_test():
    dates = pd.date_range(
        start="2025-01-06",
        periods=8,
        freq="W-MON",
    )

    data = pd.DataFrame(
        {
            "date": dates,
            "stock_code": ["SKU-001"] * 8,
        }
    )

    train, test = temporal_train_test_split(
        data,
        test_weeks=2,
        min_train_weeks=6,
    )

    assert len(
        train["date"].unique()
    ) == 6

    assert len(
        test["date"].unique()
    ) == 2

    assert (
        train["date"].max()
        < test["date"].min()
    )

    assert set(
        test["date"].unique()
    ) == set(
        dates[-2:]
    )


def test_mape_ignores_zero_actual_demand():
    y_true = np.array(
        [
            100.0,
            0.0,
            50.0,
        ]
    )

    y_pred = np.array(
        [
            110.0,
            20.0,
            40.0,
        ]
    )

    result = _mape(
        y_true,
        y_pred,
    )

    assert np.isclose(
        result,
        15.0,
    )


def test_forecast_bias_sign_matches_over_and_under_forecasting():
    y_true = np.array(
        [
            100.0,
            200.0,
        ]
    )

    over_predictions = np.array(
        [
            110.0,
            220.0,
        ]
    )

    under_predictions = np.array(
        [
            90.0,
            180.0,
        ]
    )

    over_bias = _forecast_bias(
        y_true,
        over_predictions,
    )

    under_bias = _forecast_bias(
        y_true,
        under_predictions,
    )

    assert over_bias > 0
    assert under_bias < 0

    assert np.isclose(
        over_bias,
        15.0,
    )

    assert np.isclose(
        under_bias,
        -15.0,
    )


def test_resolve_model_features_includes_available_optional_features():
    data = _make_supervised_data()

    numeric_features, categorical_features = (
        _resolve_model_features(data)
    )

    assert "lag_1" in numeric_features

    assert (
        "rolling_mean_8"
        in numeric_features
    )

    assert (
        "price_to_cost_ratio_lag_1"
        in numeric_features
    )

    assert (
        "promotion_days_lag_1"
        in numeric_features
    )

    assert (
        "stock_code"
        in categorical_features
    )

    assert (
        "category"
        in categorical_features
    )

    assert "demand" not in numeric_features

    assert (
        "demand"
        not in categorical_features
    )


def test_train_and_evaluate_models_returns_expected_artifacts():
    data = _make_supervised_data(
        n_weeks=12,
    )

    result = train_and_evaluate_models(
        data,
        test_weeks=2,
        min_train_weeks=8,
        random_state=42,
    )

    expected_models = {
        "moving_average_4",
        "ridge_regression",
        "random_forest",
        "gradient_boosting",
    }

    assert set(
        result.metrics["model"]
    ) == expected_models

    assert result.best_model_name in {
        "ridge_regression",
        "random_forest",
        "gradient_boosting",
    }

    assert result.best_model is not None

    assert len(
        result.test_predictions[
            "date"
        ].unique()
    ) == 2

    assert (
        "prediction"
        in result.test_predictions.columns
    )

    assert (
        "best_model"
        in result.test_predictions.columns
    )

    assert (
        result.test_predictions[
            "prediction"
        ] >= 0
    ).all()

    assert {
        "MAE",
        "RMSE",
        "MAPE_percent",
        "bias",
    }.issubset(
        result.metrics.columns
    )


def test_predict_next_period_returns_valid_forecasts():
    data = _make_supervised_data(
        n_weeks=12,
    )

    result = train_and_evaluate_models(
        data,
        test_weeks=2,
        min_train_weeks=8,
        random_state=42,
    )

    last_date = data["date"].max()

    next_features = data.loc[
        data["date"] == last_date
    ].copy()

    next_features["date"] = (
        next_features["date"]
        + pd.Timedelta(days=7)
    )

    next_features = next_features.drop(
        columns=["demand"]
    )

    forecast = predict_next_period(
        result,
        next_features,
    )

    assert len(forecast) == 2

    assert {
        "date",
        "stock_code",
        "category",
        "predicted_demand",
        "model",
    }.issubset(
        forecast.columns
    )

    assert (
        forecast["predicted_demand"] >= 0
    ).all()

    assert (
        forecast["model"]
        == result.best_model_name
    ).all()

    assert set(
        forecast["stock_code"]
    ) == {
        "SKU-001",
        "SKU-002",
    }

    assert (
        forecast["predicted_demand"]
        .is_monotonic_decreasing
    )


def test_predict_next_period_raises_error_for_missing_model_feature():
    data = _make_supervised_data(
        n_weeks=12,
    )

    result = train_and_evaluate_models(
        data,
        test_weeks=2,
        min_train_weeks=8,
        random_state=42,
    )

    last_date = data["date"].max()

    next_features = data.loc[
        data["date"] == last_date
    ].copy()

    next_features["date"] = (
        next_features["date"]
        + pd.Timedelta(days=7)
    )

    next_features = next_features.drop(
        columns=[
            "demand",
            "lag_8",
        ]
    )

    with pytest.raises(
        ValueError,
        match="lag_8",
    ):
        predict_next_period(
            result,
            next_features,
        )