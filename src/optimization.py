"""Inventory optimization using linear programming."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog


REQUIRED_FORECAST_COLUMNS = {
    "stock_code",
    "predicted_demand",
}

REQUIRED_HISTORY_COLUMNS = {
    "date",
    "stock_code",
    "unit_price",
    "unit_cost",
}


def _validate_inventory_source_data(
    forecasts: pd.DataFrame,
    weekly_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and standardize forecast and weekly-history inputs."""

    if forecasts.empty:
        raise ValueError(
            "Forecast data must not be empty."
        )

    if weekly_history.empty:
        raise ValueError(
            "Weekly history must not be empty."
        )

    missing_forecast_columns = (
        REQUIRED_FORECAST_COLUMNS.difference(
            forecasts.columns
        )
    )

    if missing_forecast_columns:
        raise ValueError(
            "Forecast data is missing required columns: "
            f"{sorted(missing_forecast_columns)}"
        )

    missing_history_columns = (
        REQUIRED_HISTORY_COLUMNS.difference(
            weekly_history.columns
        )
    )

    if missing_history_columns:
        raise ValueError(
            "Weekly history is missing required columns: "
            f"{sorted(missing_history_columns)}"
        )

    forecast_data = forecasts.copy()
    history_data = weekly_history.copy()

    history_data["date"] = pd.to_datetime(
        history_data["date"],
        errors="raise",
    )

    forecast_data["predicted_demand"] = pd.to_numeric(
        forecast_data["predicted_demand"],
        errors="raise",
    )

    history_data["unit_price"] = pd.to_numeric(
        history_data["unit_price"],
        errors="raise",
    )

    history_data["unit_cost"] = pd.to_numeric(
        history_data["unit_cost"],
        errors="raise",
    )

    if forecast_data[
        sorted(REQUIRED_FORECAST_COLUMNS)
    ].isna().any().any():
        raise ValueError(
            "Forecast data contains missing values "
            "in required columns."
        )

    if history_data[
        sorted(REQUIRED_HISTORY_COLUMNS)
    ].isna().any().any():
        raise ValueError(
            "History data contains missing values "
            "in required columns."
        )

    duplicate_forecast_products = (
        forecast_data.loc[
            forecast_data["stock_code"].duplicated(
                keep=False
            ),
            "stock_code",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if duplicate_forecast_products:
        raise ValueError(
            "Forecast data must contain one row per product. "
            "Duplicate stock codes: "
            f"{sorted(duplicate_forecast_products)}"
        )

    if not np.isfinite(
        forecast_data["predicted_demand"]
    ).all():
        raise ValueError(
            "Predicted demand must contain only finite values."
        )

    if (
        forecast_data["predicted_demand"] < 0
    ).any():
        raise ValueError(
            "Predicted demand must be non-negative."
        )

    if not np.isfinite(
        history_data["unit_cost"]
    ).all():
        raise ValueError(
            "Unit cost must contain only finite values."
        )

    if not np.isfinite(
        history_data["unit_price"]
    ).all():
        raise ValueError(
            "Unit price must contain only finite values."
        )

    if (
        history_data["unit_cost"] <= 0
    ).any():
        raise ValueError(
            "Unit cost must be greater than zero."
        )

    if (
        history_data["unit_price"] <= 0
    ).any():
        raise ValueError(
            "Unit price must be greater than zero."
        )

    forecast_products = set(
        forecast_data["stock_code"]
    )

    history_products = set(
        history_data["stock_code"]
    )

    missing_history_products = (
        forecast_products - history_products
    )

    if missing_history_products:
        raise ValueError(
            "Forecast products are missing from weekly history: "
            f"{sorted(missing_history_products)}"
        )

    history_data = (
        history_data
        .sort_values(
            ["stock_code", "date"]
        )
        .reset_index(drop=True)
    )

    forecast_data = (
        forecast_data
        .reset_index(drop=True)
    )

    return forecast_data, history_data


def _prepare_inventory_input(
    forecasts: pd.DataFrame,
    weekly_history: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare one optimization input row per forecast product."""

    forecast_data, history_data = (
        _validate_inventory_source_data(
            forecasts=forecasts,
            weekly_history=weekly_history,
        )
    )

    latest_history = (
        history_data
        .sort_values(
            ["stock_code", "date"]
        )
        .groupby(
            "stock_code",
            as_index=False,
        )
        .tail(1)
        [
            [
                "stock_code",
                "date",
                "unit_price",
                "unit_cost",
            ]
        ]
        .rename(
            columns={
                "date": "history_date"
            }
        )
        .reset_index(drop=True)
    )

    forecast_columns_to_drop = [
        column
        for column in [
            "unit_price",
            "unit_cost",
        ]
        if column in forecast_data.columns
    ]

    forecast_base = forecast_data.drop(
        columns=forecast_columns_to_drop
    )

    inventory_inputs = forecast_base.merge(
        latest_history,
        on="stock_code",
        how="left",
        validate="one_to_one",
    )

    inventory_inputs["unit_margin"] = (
        inventory_inputs["unit_price"]
        - inventory_inputs["unit_cost"]
    )

    return (
        inventory_inputs
        .sort_values("stock_code")
        .reset_index(drop=True)
    )


def _validate_optimization_parameter(
    budget: float,
    capacity_units: float,
) -> tuple[float, float]:
    """Validate budget and capacity constraints for optimization."""

    try:
        budget_value = float(budget)
        capacity_value = float(capacity_units)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "budget and capacity_units must be numeric."
        ) from exc

    if not np.isfinite(budget_value):
        raise ValueError(
            "Budget must be finite."
        )

    if not np.isfinite(capacity_value):
        raise ValueError(
            "Capacity must be finite."
        )

    if budget_value <= 0:
        raise ValueError(
            "Budget must be greater than zero."
        )

    if capacity_value <= 0:
        raise ValueError(
            "Capacity must be greater than zero."
        )

    return budget_value, capacity_value


def _build_linear_program(
    inventory_inputs: pd.DataFrame,
    budget: float,
    capacity_units: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[float, float]],
]:
    """Build coefficient arrays for the inventory linear program."""

    required_columns = {
        "predicted_demand",
        "unit_cost",
        "unit_margin",
    }

    missing_columns = required_columns.difference(
        inventory_inputs.columns
    )

    if missing_columns:
        raise ValueError(
            "Optimization inputs are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if inventory_inputs.empty:
        raise ValueError(
            "Optimization inputs must not be empty."
        )

    budget_value, capacity_value = (
        _validate_optimization_parameter(
            budget=budget,
            capacity_units=capacity_units,
        )
    )

    predicted_demand = inventory_inputs[
        "predicted_demand"
    ].to_numpy(dtype=float)

    unit_costs = inventory_inputs[
        "unit_cost"
    ].to_numpy(dtype=float)

    unit_margins = inventory_inputs[
        "unit_margin"
    ].to_numpy(dtype=float)

    objective = -unit_margins

    budget_constraint = unit_costs

    capacity_constraint = np.ones(
        len(inventory_inputs),
        dtype=float,
    )

    constraint_matrix = np.vstack(
        [
            budget_constraint,
            capacity_constraint,
        ]
    )

    constraint_limits = np.array(
        [
            budget_value,
            capacity_value,
        ],
        dtype=float,
    )

    bounds = [
        (0.0, float(demand))
        for demand in predicted_demand
    ]

    return (
        objective,
        constraint_matrix,
        constraint_limits,
        bounds,
    )


def _solve_linear_program(
    objective: np.ndarray,
    constraint_matrix: np.ndarray,
    constraint_limits: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    """Solve the inventory linear program."""

    result = linprog(
        c=objective,
        A_ub=constraint_matrix,
        b_ub=constraint_limits,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            "Inventory optimization failed: "
            f"{result.message}"
        )

    return result.x


def optimize_inventory(
    forecasts: pd.DataFrame,
    weekly_history: pd.DataFrame,
    budget: float,
    capacity_units: float,
) -> pd.DataFrame:
    """Optimize inventory allocation under budget and capacity constraints."""

    inventory_inputs = _prepare_inventory_input(
        forecasts=forecasts,
        weekly_history=weekly_history,
    )

    (
        objective,
        constraint_matrix,
        constraint_limits,
        bounds,
    ) = _build_linear_program(
        inventory_inputs=inventory_inputs,
        budget=budget,
        capacity_units=capacity_units,
    )

    optimal_quantities = _solve_linear_program(
        objective=objective,
        constraint_matrix=constraint_matrix,
        constraint_limits=constraint_limits,
        bounds=bounds,
    )

    result = inventory_inputs.copy()

    result["recommended_quantity"] = (
        optimal_quantities
    )

    result["allocated_cost"] = (
        result["recommended_quantity"]
        * result["unit_cost"]
    )

    result["expected_profit"] = (
        result["recommended_quantity"]
        * result["unit_margin"]
    )

    positive_demand = (
        result["predicted_demand"] > 0
    )

    result["demand_coverage"] = 0.0

    result.loc[
        positive_demand,
        "demand_coverage",
    ] = (
        result.loc[
            positive_demand,
            "recommended_quantity",
        ]
        / result.loc[
            positive_demand,
            "predicted_demand",
        ]
    )

    return (
        result
        .sort_values(
            [
                "expected_profit",
                "stock_code",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def summarize_optimization(
    optimization_result: pd.DataFrame,
    budget: float,
    capacity_units: float,
) -> pd.DataFrame:
    """Create a one-row business summary of optimization results."""

    required_columns = {
        "predicted_demand",
        "recommended_quantity",
        "allocated_cost",
        "expected_profit",
    }

    missing_columns = required_columns.difference(
        optimization_result.columns
    )

    if missing_columns:
        raise ValueError(
            "Optimization result is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if optimization_result.empty:
        raise ValueError(
            "Optimization result must not be empty."
        )

    budget_value, capacity_value = (
        _validate_optimization_parameter(
            budget=budget,
            capacity_units=capacity_units,
        )
    )

    total_predicted_demand = float(
        optimization_result[
            "predicted_demand"
        ].sum()
    )

    total_recommended_quantity = float(
        optimization_result[
            "recommended_quantity"
        ].sum()
    )

    total_allocated_cost = float(
        optimization_result[
            "allocated_cost"
        ].sum()
    )

    total_expected_profit = float(
        optimization_result[
            "expected_profit"
        ].sum()
    )

    remaining_budget = max(
        budget_value
        - total_allocated_cost,
        0.0,
    )

    remaining_capacity = max(
        capacity_value
        - total_recommended_quantity,
        0.0,
    )

    unmet_predicted_demand = max(
        total_predicted_demand
        - total_recommended_quantity,
        0.0,
    )

    budget_utilization = (
        total_allocated_cost
        / budget_value
    )

    capacity_utilization = (
        total_recommended_quantity
        / capacity_value
    )

    if total_predicted_demand > 0:
        overall_demand_coverage = (
            total_recommended_quantity
            / total_predicted_demand
        )
    else:
        overall_demand_coverage = 0.0

    return pd.DataFrame(
        [
            {
                "total_predicted_demand":
                    total_predicted_demand,

                "total_recommended_quantity":
                    total_recommended_quantity,

                "unmet_predicted_demand":
                    unmet_predicted_demand,

                "total_allocated_cost":
                    total_allocated_cost,

                "remaining_budget":
                    remaining_budget,

                "budget_utilization":
                    budget_utilization,

                "remaining_capacity":
                    remaining_capacity,

                "capacity_utilization":
                    capacity_utilization,

                "overall_demand_coverage":
                    overall_demand_coverage,

                "total_expected_profit":
                    total_expected_profit,
            }
        ]
    )