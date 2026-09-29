import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    _build_linear_program,
    _prepare_inventory_input,
    optimize_inventory,
    summarize_optimization,
)


def _make_inventory_data():
    """Create deterministic forecast and history data for tests."""

    forecasts = pd.DataFrame(
        {
            "stock_code": [
                "SKU-001",
                "SKU-002",
                "SKU-003",
            ],
            "predicted_demand": [
                10.0,
                8.0,
                6.0,
            ],
        }
    )

    weekly_history = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-05",
                    "2026-01-12",
                    "2026-01-05",
                    "2026-01-12",
                    "2026-01-05",
                    "2026-01-12",
                ]
            ),
            "stock_code": [
                "SKU-001",
                "SKU-001",
                "SKU-002",
                "SKU-002",
                "SKU-003",
                "SKU-003",
            ],
            "unit_price": [
                10.0,
                12.0,
                15.0,
                16.0,
                9.0,
                10.0,
            ],
            "unit_cost": [
                5.0,
                6.0,
                8.0,
                8.0,
                4.0,
                5.0,
            ],
        }
    )

    return forecasts, weekly_history


def test_prepare_inventory_input_uses_latest_history():
    """The latest price and cost must be used for each product."""

    forecasts, weekly_history = _make_inventory_data()

    result = _prepare_inventory_input(
        forecasts=forecasts,
        weekly_history=weekly_history,
    )

    assert len(result) == 3

    sku_001 = result.loc[
        result["stock_code"] == "SKU-001"
    ].iloc[0]

    assert sku_001["unit_price"] == 12.0
    assert sku_001["unit_cost"] == 6.0
    assert sku_001["unit_margin"] == 6.0


def test_build_linear_program_creates_expected_coefficients():
    """The business inputs must be converted to correct LP arrays."""

    inventory_inputs = pd.DataFrame(
        {
            "predicted_demand": [
                10.0,
                8.0,
            ],
            "unit_cost": [
                6.0,
                8.0,
            ],
            "unit_margin": [
                6.0,
                8.0,
            ],
        }
    )

    (
        objective,
        constraint_matrix,
        constraint_limits,
        bounds,
    ) = _build_linear_program(
        inventory_inputs=inventory_inputs,
        budget=100.0,
        capacity_units=12.0,
    )

    np.testing.assert_array_equal(
        objective,
        np.array(
            [
                -6.0,
                -8.0,
            ]
        ),
    )

    np.testing.assert_array_equal(
        constraint_matrix,
        np.array(
            [
                [6.0, 8.0],
                [1.0, 1.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        constraint_limits,
        np.array(
            [
                100.0,
                12.0,
            ]
        ),
    )

    assert bounds == [
        (0.0, 10.0),
        (0.0, 8.0),
    ]


def test_optimize_inventory_respects_constraints():
    """The optimized quantities must satisfy all model constraints."""

    forecasts, weekly_history = _make_inventory_data()

    budget = 100.0
    capacity_units = 12.0

    result = optimize_inventory(
        forecasts=forecasts,
        weekly_history=weekly_history,
        budget=budget,
        capacity_units=capacity_units,
    )

    assert len(result) == 3

    assert (
        result["recommended_quantity"] >= 0
    ).all()

    assert (
        result["recommended_quantity"]
        <= result["predicted_demand"] + 1e-9
    ).all()

    assert (
        result["allocated_cost"].sum()
        <= budget + 1e-9
    )

    assert (
        result["recommended_quantity"].sum()
        <= capacity_units + 1e-9
    )

    np.testing.assert_allclose(
        result["allocated_cost"],
        result["recommended_quantity"]
        * result["unit_cost"],
    )

    np.testing.assert_allclose(
        result["expected_profit"],
        result["recommended_quantity"]
        * result["unit_margin"],
    )

    assert (
        result["expected_profit"].sum()
        == pytest.approx(88.0)
    )


def test_summarize_optimization_returns_expected_metrics():
    """The business summary must calculate expected aggregate metrics."""

    optimization_result = pd.DataFrame(
        {
            "predicted_demand": [
                10.0,
                8.0,
                6.0,
            ],
            "recommended_quantity": [
                4.0,
                8.0,
                0.0,
            ],
            "allocated_cost": [
                24.0,
                64.0,
                0.0,
            ],
            "expected_profit": [
                24.0,
                64.0,
                0.0,
            ],
        }
    )

    summary = summarize_optimization(
        optimization_result=optimization_result,
        budget=100.0,
        capacity_units=12.0,
    )

    assert len(summary) == 1

    row = summary.iloc[0]

    assert (
        row["total_predicted_demand"]
        == pytest.approx(24.0)
    )

    assert (
        row["total_recommended_quantity"]
        == pytest.approx(12.0)
    )

    assert (
        row["unmet_predicted_demand"]
        == pytest.approx(12.0)
    )

    assert (
        row["total_allocated_cost"]
        == pytest.approx(88.0)
    )

    assert (
        row["remaining_budget"]
        == pytest.approx(12.0)
    )

    assert (
        row["budget_utilization"]
        == pytest.approx(0.88)
    )

    assert (
        row["remaining_capacity"]
        == pytest.approx(0.0)
    )

    assert (
        row["capacity_utilization"]
        == pytest.approx(1.0)
    )

    assert (
        row["overall_demand_coverage"]
        == pytest.approx(0.5)
    )

    assert (
        row["total_expected_profit"]
        == pytest.approx(88.0)
    )


def test_optimize_inventory_rejects_duplicate_forecast_products():
    """Forecast data must contain exactly one row per product."""

    forecasts, weekly_history = _make_inventory_data()

    duplicate_row = forecasts.iloc[[0]]

    forecasts = pd.concat(
        [
            forecasts,
            duplicate_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="one row per product",
    ):
        optimize_inventory(
            forecasts=forecasts,
            weekly_history=weekly_history,
            budget=100.0,
            capacity_units=12.0,
        )


def test_optimize_inventory_rejects_non_positive_budget():
    """Budget must be strictly greater than zero."""

    forecasts, weekly_history = _make_inventory_data()

    with pytest.raises(
        ValueError,
        match="Budget must be greater than zero",
    ):
        optimize_inventory(
            forecasts=forecasts,
            weekly_history=weekly_history,
            budget=0.0,
            capacity_units=12.0,
        )