"""Visualization utilities for forecasting and inventory results."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def plot_model_rmse(
        metrics: pd.DataFrame,
        output_path: Path
) -> None:

    required_columns = {
        "model",
        "RMSE"
    }

    missing_columns = required_columns.difference(
        metrics.columns
    )

    if missing_columns:
        raise ValueError(
            "Model metrics are missing required columns:"
            f"{sorted(missing_columns)}"
        )

    if metrics.empty:
        raise ValueError(
            "Model metrics must not be empty."
        )

    plot_data = (
        metrics[["model", "RMSE"]]
        .sort_values(
            "RMSE",
            ascending=True
        )
        .reset_index(
            drop=True
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(9,5)
    )

    ax.bar(
        plot_data["model"],
        plot_data["RMSE"]
    )

    ax.set_title(
        "Forcasting Model RMSE Comparison"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "RMSE"
    )

    ax.tick_params(
        axis="x",
        rotation=25
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches = "tight"
    )

    plt.close(fig)


def plot_holdout_actual_vs_predicted(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot actual and predicted total demand across holdout weeks."""

    required_columns = {
        "date",
        "demand",
        "prediction",
    }

    missing_columns = required_columns.difference(
        predictions.columns
    )

    if missing_columns:
        raise ValueError(
            "Forecast predictions are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if predictions.empty:
        raise ValueError(
            "Forecast predictions must not be empty."
        )

    plot_data = predictions[
        [
            "date",
            "demand",
            "prediction",
        ]
    ].copy()

    plot_data["date"] = pd.to_datetime(
        plot_data["date"],
        errors="coerce",
    )

    if plot_data["date"].isna().any():
        raise ValueError(
            "Forecast prediction dates must be valid."
        )

    for column in [
        "demand",
        "prediction",
    ]:
        plot_data[column] = pd.to_numeric(
            plot_data[column],
            errors="coerce",
        )

    if plot_data[
        [
            "demand",
            "prediction",
        ]
    ].isna().any().any():
        raise ValueError(
            "Demand and prediction values must be numeric."
        )

    weekly_data = (
        plot_data
        .groupby(
            "date",
            as_index=False,
        )[
            [
                "demand",
                "prediction",
            ]
        ]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        weekly_data["date"],
        weekly_data["demand"],
        marker="o",
        label="Actual Demand",
    )

    ax.plot(
        weekly_data["date"],
        weekly_data["prediction"],
        marker="o",
        label="Predicted Demand",
    )

    ax.set_title(
        "Holdout Actual vs Predicted Demand"
    )

    ax.set_xlabel(
        "Week"
    )

    ax.set_ylabel(
        "Total Demand"
    )

    ax.legend()

    fig.autofmt_xdate()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_forecast_vs_recommendation(
    optimization_result: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Plot predicted demand against recommended inventory quantities."""

    required_columns = {
        "stock_code",
        "predicted_demand",
        "recommended_quantity",
    }

    missing_columns = required_columns.difference(
        optimization_result.columns
    )

    if missing_columns:
        raise ValueError(
            "Optimization results are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if optimization_result.empty:
        raise ValueError(
            "Optimization results must not be empty."
        )

    if (
        not isinstance(top_n, int)
        or isinstance(top_n, bool)
        or top_n <= 0
    ):
        raise ValueError(
            "top_n must be a positive integer."
        )

    plot_data = optimization_result[
        [
            "stock_code",
            "predicted_demand",
            "recommended_quantity",
        ]
    ].copy()

    if plot_data["stock_code"].isna().any():
        raise ValueError(
            "Stock codes must not contain missing values."
        )

    plot_data["stock_code"] = (
        plot_data["stock_code"]
        .astype(str)
    )

    for column in [
        "predicted_demand",
        "recommended_quantity",
    ]:
        plot_data[column] = pd.to_numeric(
            plot_data[column],
            errors="coerce",
        )

    if plot_data[
        [
            "predicted_demand",
            "recommended_quantity",
        ]
    ].isna().any().any():
        raise ValueError(
            "Predicted demand and recommended quantities "
            "must be numeric."
        )

    if (
        plot_data[
            [
                "predicted_demand",
                "recommended_quantity",
            ]
        ]
        < 0
    ).any().any():
        raise ValueError(
            "Predicted demand and recommended quantities "
            "must be non-negative."
        )

    plot_data = (
        plot_data
        .sort_values(
            "predicted_demand",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    positions = list(
        range(len(plot_data))
    )

    bar_width = 0.4

    predicted_positions = [
        position - bar_width / 2
        for position in positions
    ]

    recommended_positions = [
        position + bar_width / 2
        for position in positions
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        predicted_positions,
        plot_data["predicted_demand"],
        width=bar_width,
        label="Predicted Demand",
    )

    ax.bar(
        recommended_positions,
        plot_data["recommended_quantity"],
        width=bar_width,
        label="Recommended Quantity",
    )

    ax.set_title(
        "Forecast Demand vs Recommended Inventory"
    )

    ax.set_xlabel(
        "Product"
    )

    ax.set_ylabel(
        "Units"
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        plot_data["stock_code"],
        rotation=45,
        ha="right",
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_resource_utilization(
    optimization_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot budget and capacity utilization percentages."""

    required_columns = {
        "budget_utilization",
        "capacity_utilization",
    }

    missing_columns = required_columns.difference(
        optimization_summary.columns
    )

    if missing_columns:
        raise ValueError(
            "Optimization summary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if optimization_summary.empty:
        raise ValueError(
            "Optimization summary must not be empty."
        )

    if len(optimization_summary) != 1:
        raise ValueError(
            "Optimization summary must contain exactly one row."
        )

    utilization_data = optimization_summary[
        [
            "budget_utilization",
            "capacity_utilization",
        ]
    ].copy()

    for column in [
        "budget_utilization",
        "capacity_utilization",
    ]:
        utilization_data[column] = pd.to_numeric(
            utilization_data[column],
            errors="coerce",
        )

    if utilization_data.isna().any().any():
        raise ValueError(
            "Utilization values must be numeric."
        )

    tolerance = 1e-9

    if (
        (
            utilization_data
            < -tolerance
        )
        | (
            utilization_data
            > 1 + tolerance
        )
    ).any().any():
        raise ValueError(
            "Utilization values must be between 0 and 1."
        )

    labels = [
        "Budget",
        "Capacity",
    ]

    utilization_percent = [
        float(
            utilization_data.iloc[0][
                "budget_utilization"
            ]
        )
        * 100,
        float(
            utilization_data.iloc[0][
                "capacity_utilization"
            ]
        )
        * 100,
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bars = ax.bar(
        labels,
        utilization_percent,
    )

    ax.set_title(
        "Resource Utilization"
    )

    ax.set_xlabel(
        "Resource"
    )

    ax.set_ylabel(
        "Utilization (%)"
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.bar_label(
        bars,
        fmt="%.1f%%",
        padding=3,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)
