import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    _validate_evaluation_data,
    build_error_table,
    evaluate_by_product,
    summarize_evaluation,
)


def _make_evaluation_data():
    """Create deterministic forecast data for evaluation tests."""

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-08",
                    "2026-01-01",
                    "2026-01-08",
                ]
            ),
            "stock_code": [
                "SKU-001",
                "SKU-001",
                "SKU-002",
                "SKU-002",
            ],
            "demand": [
                100.0,
                50.0,
                0.0,
                20.0,
            ],
            "prediction": [
                110.0,
                40.0,
                0.0,
                30.0,
            ],
        }
    )


def test_validate_evaluation_data_sorts_and_preserves_rows():
    """Validation should standardize and sort evaluation data."""

    data = _make_evaluation_data()

    result = _validate_evaluation_data(
        data
    )

    assert len(result) == 4

    assert list(result["stock_code"]) == [
        "SKU-001",
        "SKU-002",
        "SKU-001",
        "SKU-002",
    ]

    assert pd.api.types.is_datetime64_any_dtype(
        result["date"]
    )


def test_build_error_table_calculates_expected_errors():
    """Row-level forecast diagnostics should be calculated correctly."""

    data = _make_evaluation_data()

    result = build_error_table(
        data
    )

    sku_001_first = result.loc[
        (
            result["stock_code"] == "SKU-001"
        )
        & (
            result["date"]
            == pd.Timestamp("2026-01-01")
        )
    ].iloc[0]

    assert (
        sku_001_first["error"]
        == pytest.approx(10.0)
    )

    assert (
        sku_001_first["absolute_error"]
        == pytest.approx(10.0)
    )

    assert (
        sku_001_first["squared_error"]
        == pytest.approx(100.0)
    )

    assert (
        sku_001_first[
            "absolute_percentage_error"
        ]
        == pytest.approx(10.0)
    )


def test_build_error_table_uses_nan_percentage_error_for_zero_demand():
    """Percentage error should be undefined when actual demand is zero."""

    data = _make_evaluation_data()

    result = build_error_table(
        data
    )

    zero_demand_row = result.loc[
        result["demand"] == 0
    ].iloc[0]

    assert np.isnan(
        zero_demand_row[
            "absolute_percentage_error"
        ]
    )


def test_evaluate_by_product_returns_expected_metrics():
    """Product-level metrics should match hand-calculated values."""

    data = _make_evaluation_data()

    result = evaluate_by_product(
        data
    )

    assert len(result) == 2

    sku_001 = result.loc[
        result["stock_code"] == "SKU-001"
    ].iloc[0]

    assert (
        sku_001["observations"]
        == 2
    )

    assert (
        sku_001["MAE"]
        == pytest.approx(10.0)
    )

    assert (
        sku_001["RMSE"]
        == pytest.approx(10.0)
    )

    assert (
        sku_001["MAPE_percent"]
        == pytest.approx(15.0)
    )

    assert (
        sku_001["bias"]
        == pytest.approx(0.0)
    )

    sku_002 = result.loc[
        result["stock_code"] == "SKU-002"
    ].iloc[0]

    assert (
        sku_002["observations"]
        == 2
    )

    assert (
        sku_002["MAE"]
        == pytest.approx(5.0)
    )

    assert (
        sku_002["RMSE"]
        == pytest.approx(
            np.sqrt(50.0)
        )
    )

    assert (
        sku_002["MAPE_percent"]
        == pytest.approx(50.0)
    )

    assert (
        sku_002["bias"]
        == pytest.approx(5.0)
    )


def test_evaluate_by_product_sorts_highest_rmse_first():
    """Products with larger RMSE should appear first."""

    data = _make_evaluation_data()

    result = evaluate_by_product(
        data
    )

    assert list(result["stock_code"]) == [
        "SKU-001",
        "SKU-002",
    ]


def test_summarize_evaluation_returns_expected_metrics():
    """Overall summary should match hand-calculated forecast metrics."""

    data = _make_evaluation_data()

    summary = summarize_evaluation(
        data
    )

    assert len(summary) == 1

    row = summary.iloc[0]

    assert (
        row["observations"]
        == 4
    )

    assert (
        row["products"]
        == 2
    )

    assert (
        row["start_date"]
        == pd.Timestamp("2026-01-01")
    )

    assert (
        row["end_date"]
        == pd.Timestamp("2026-01-08")
    )

    assert (
        row["MAE"]
        == pytest.approx(7.5)
    )

    assert (
        row["RMSE"]
        == pytest.approx(
            np.sqrt(75.0)
        )
    )

    assert (
        row["MAPE_percent"]
        == pytest.approx(
            80.0 / 3.0
        )
    )

    assert (
        row["bias"]
        == pytest.approx(2.5)
    )

    assert (
        row["zero_demand_observations"]
        == 1
    )

    assert (
        row["over_forecast_count"]
        == 2
    )

    assert (
        row["under_forecast_count"]
        == 1
    )

    assert (
        row["exact_forecast_count"]
        == 1
    )

    assert (
        row["over_forecast_rate"]
        == pytest.approx(0.5)
    )

    assert (
        row["under_forecast_rate"]
        == pytest.approx(0.25)
    )

    assert (
        row["exact_forecast_rate"]
        == pytest.approx(0.25)
    )


def test_validate_evaluation_data_rejects_empty_data():
    """Evaluation data must contain at least one observation."""

    empty_data = pd.DataFrame(
        columns=[
            "date",
            "stock_code",
            "demand",
            "prediction",
        ]
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        _validate_evaluation_data(
            empty_data
        )


def test_validate_evaluation_data_rejects_missing_columns():
    """All required evaluation columns must be present."""

    data = _make_evaluation_data()

    data = data.drop(
        columns="prediction"
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        _validate_evaluation_data(
            data
        )


def test_validate_evaluation_data_rejects_negative_prediction():
    """Forecast predictions must not be negative."""

    data = _make_evaluation_data()

    data.loc[
        0,
        "prediction",
    ] = -5.0

    with pytest.raises(
        ValueError,
        match="Predictions must be non-negative",
    ):
        _validate_evaluation_data(
            data
        )


def test_validate_evaluation_data_rejects_non_finite_demand():
    """Actual demand must not contain infinity."""

    data = _make_evaluation_data()

    data.loc[
        0,
        "demand",
    ] = np.inf

    with pytest.raises(
        ValueError,
        match="Actual demand must contain only finite values",
    ):
        _validate_evaluation_data(
            data
        )