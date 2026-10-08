import pandas as pd
import pytest

from src.visualization import (
    plot_forecast_vs_recommendation,
    plot_holdout_actual_vs_predicted,
    plot_model_rmse,
    plot_resource_utilization,
)


def test_plot_model_rmse_creates_file(
    tmp_path,
):
    """Create the model RMSE plot successfully."""

    metrics = pd.DataFrame(
        {
            "model": [
                "ridge_regression",
                "random_forest",
            ],
            "RMSE": [
                20.0,
                30.0,
            ],
        }
    )

    output_path = (
        tmp_path
        / "model_rmse.png"
    )

    plot_model_rmse(
        metrics,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_model_rmse_rejects_missing_columns(
    tmp_path,
):
    """Reject model metrics without required columns."""

    metrics = pd.DataFrame(
        {
            "model": [
                "ridge_regression",
            ],
        }
    )

    output_path = (
        tmp_path
        / "model_rmse.png"
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        plot_model_rmse(
            metrics,
            output_path,
        )


def test_plot_holdout_actual_vs_predicted_creates_file(
    tmp_path,
):
    """Create the holdout comparison plot successfully."""

    predictions = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-08",
                "2024-01-08",
            ],
            "demand": [
                100,
                200,
                120,
                180,
            ],
            "prediction": [
                110,
                190,
                125,
                170,
            ],
        }
    )

    output_path = (
        tmp_path
        / "holdout_actual_vs_predicted.png"
    )

    plot_holdout_actual_vs_predicted(
        predictions,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_holdout_rejects_invalid_dates(
    tmp_path,
):
    """Reject holdout data containing invalid dates."""

    predictions = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "not-a-date",
            ],
            "demand": [
                100,
                200,
            ],
            "prediction": [
                110,
                190,
            ],
        }
    )

    output_path = (
        tmp_path
        / "holdout_actual_vs_predicted.png"
    )

    with pytest.raises(
        ValueError,
        match="dates must be valid",
    ):
        plot_holdout_actual_vs_predicted(
            predictions,
            output_path,
        )


def test_plot_forecast_vs_recommendation_creates_file(
    tmp_path,
):
    """Create the forecast-versus-recommendation plot successfully."""

    optimization_result = pd.DataFrame(
        {
            "stock_code": [
                "SKU-001",
                "SKU-002",
                "SKU-003",
            ],
            "predicted_demand": [
                500.0,
                300.0,
                200.0,
            ],
            "recommended_quantity": [
                400.0,
                250.0,
                150.0,
            ],
        }
    )

    output_path = (
        tmp_path
        / "forecast_vs_recommendation.png"
    )

    plot_forecast_vs_recommendation(
        optimization_result,
        output_path,
        top_n=2,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.parametrize(
    "invalid_top_n",
    [
        0,
        -1,
        2.5,
        "10",
        True,
    ],
)
def test_plot_forecast_vs_recommendation_rejects_invalid_top_n(
    tmp_path,
    invalid_top_n,
):
    """Reject invalid values for top_n."""

    optimization_result = pd.DataFrame(
        {
            "stock_code": [
                "SKU-001",
            ],
            "predicted_demand": [
                500.0,
            ],
            "recommended_quantity": [
                400.0,
            ],
        }
    )

    output_path = (
        tmp_path
        / "forecast_vs_recommendation.png"
    )

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        plot_forecast_vs_recommendation(
            optimization_result,
            output_path,
            top_n=invalid_top_n,
        )


def test_plot_resource_utilization_creates_file(
    tmp_path,
):
    """Create the resource-utilization plot successfully."""

    optimization_summary = pd.DataFrame(
        {
            "budget_utilization": [
                0.80,
            ],
            "capacity_utilization": [
                0.65,
            ],
        }
    )

    output_path = (
        tmp_path
        / "resource_utilization.png"
    )

    plot_resource_utilization(
        optimization_summary,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_resource_utilization_rejects_multiple_rows(
    tmp_path,
):
    """Reject optimization summaries containing multiple rows."""

    optimization_summary = pd.DataFrame(
        {
            "budget_utilization": [
                0.80,
                0.90,
            ],
            "capacity_utilization": [
                0.65,
                0.75,
            ],
        }
    )

    output_path = (
        tmp_path
        / "resource_utilization.png"
    )

    with pytest.raises(
        ValueError,
        match="exactly one row",
    ):
        plot_resource_utilization(
            optimization_summary,
            output_path,
        )


def test_plot_resource_utilization_rejects_out_of_range_values(
    tmp_path,
):
    """Reject utilization values outside the valid range."""

    optimization_summary = pd.DataFrame(
        {
            "budget_utilization": [
                1.20,
            ],
            "capacity_utilization": [
                0.75,
            ],
        }
    )

    output_path = (
        tmp_path
        / "resource_utilization.png"
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        plot_resource_utilization(
            optimization_summary,
            output_path,
        )