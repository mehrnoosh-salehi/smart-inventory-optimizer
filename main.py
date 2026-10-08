"""Run the Smart Inventory Optimizer end-to-end pipeline."""

from pathlib import Path

import pandas as pd

from src.preprocessing import (
    aggregate_weekly_demand,
    clean_sales_data,
    load_sales_data,
)

from src.features import (
    create_next_period_features,
    create_supervised_features,
)

from src.forecasting import (
    ForecastingResult,
    predict_next_period,
    train_and_evaluate_models,
)

from src.evaluation import (
    evaluate_by_product,
    summarize_evaluation,
)

from src.optimization import (
    optimize_inventory,
    summarize_optimization,
)

from src.visualization import (
    plot_forecast_vs_recommendation,
    plot_holdout_actual_vs_predicted,
    plot_model_rmse,
    plot_resource_utilization,
)


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sample_retail_sales.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weekly_sales.csv"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

FIGURES_DIR = (
    REPORTS_DIR
    / "figures"
)

DEFAULT_BUDGET = 250_000.0
DEFAULT_CAPACITY_UNITS = 5_000.0


def run_preprocessing() -> pd.DataFrame:
    """Load, clean, aggregate, and save the sales data."""

    raw_data = load_sales_data(
        RAW_DATA_PATH
    )

    clean_data = clean_sales_data(
        raw_data
    )

    weekly_data = aggregate_weekly_demand(
        clean_data
    )

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weekly_data.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    return weekly_data


def run_feature_engineering(
    weekly_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create training and next-period forecasting features."""

    supervised_data = create_supervised_features(
        weekly_data
    )

    next_features = create_next_period_features(
        weekly_data
    )

    return supervised_data, next_features


def run_forecasting(
    supervised_data: pd.DataFrame,
    next_features: pd.DataFrame,
) -> tuple[ForecastingResult, pd.DataFrame]:
    """Train forecasting models and predict next-period demand."""

    forecasting_result = train_and_evaluate_models(
        supervised_data
    )

    next_forecasts = predict_next_period(
        result=forecasting_result,
        next_features=next_features,
    )

    return forecasting_result, next_forecasts


def run_evaluation(
    forecasting_result: ForecastingResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate holdout forecast performance."""

    product_evaluation = evaluate_by_product(
        forecasting_result.test_predictions
    )

    evaluation_summary = summarize_evaluation(
        forecasting_result.test_predictions
    )

    return product_evaluation, evaluation_summary


def run_optimization(
    weekly_data: pd.DataFrame,
    next_forecasts: pd.DataFrame,
    budget: float,
    capacity_units: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optimize inventory allocation and summarize the result."""

    optimization_result = optimize_inventory(
        forecasts=next_forecasts,
        weekly_history=weekly_data,
        budget=budget,
        capacity_units=capacity_units,
    )

    optimization_summary = summarize_optimization(
        optimization_result=optimization_result,
        budget=budget,
        capacity_units=capacity_units,
    )

    return optimization_result, optimization_summary


def run_visualizations(
    forecasting_result: ForecastingResult,
    optimization_result: pd.DataFrame,
    optimization_summary: pd.DataFrame,
) -> None:
    """Generate and save forecasting and optimization visualizations."""

    plot_model_rmse(
        metrics=forecasting_result.metrics,
        output_path=FIGURES_DIR / "model_rmse.png",
    )

    plot_holdout_actual_vs_predicted(
        predictions=forecasting_result.test_predictions,
        output_path=FIGURES_DIR / "holdout_actual_vs_predicted.png",
    )

    plot_forecast_vs_recommendation(
        optimization_result=optimization_result,
        output_path=FIGURES_DIR / "forecast_vs_recommendation.png",
    )

    plot_resource_utilization(
        optimization_summary=optimization_summary,
        output_path=FIGURES_DIR / "resource_utilization.png",
    )


def save_reports(
    forecasting_result: ForecastingResult,
    next_forecasts: pd.DataFrame,
    product_evaluation: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
    optimization_result: pd.DataFrame,
    optimization_summary: pd.DataFrame,
) -> None:
    """Save forecasting, evaluation, and optimization reports."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecasting_result.metrics.to_csv(
        REPORTS_DIR / "model_metrics.csv",
        index=False,
    )

    next_forecasts.to_csv(
        REPORTS_DIR / "next_period_forecasts.csv",
        index=False,
    )

    product_evaluation.to_csv(
        REPORTS_DIR / "product_evaluation.csv",
        index=False,
    )

    evaluation_summary.to_csv(
        REPORTS_DIR / "evaluation_summary.csv",
        index=False,
    )

    optimization_result.to_csv(
        REPORTS_DIR / "inventory_recommendations.csv",
        index=False,
    )

    optimization_summary.to_csv(
        REPORTS_DIR / "optimization_summary.csv",
        index=False,
    )


def run_pipeline(
    budget: float = DEFAULT_BUDGET,
    capacity_units: float = DEFAULT_CAPACITY_UNITS,
) -> None:
    """Run the complete smart inventory optimization pipeline."""

    weekly_data = run_preprocessing()

    supervised_data, next_features = run_feature_engineering(
        weekly_data
    )

    forecasting_result, next_forecasts = run_forecasting(
        supervised_data,
        next_features,
    )

    product_evaluation, evaluation_summary = run_evaluation(
        forecasting_result
    )

    optimization_result, optimization_summary = run_optimization(
        weekly_data=weekly_data,
        next_forecasts=next_forecasts,
        budget=budget,
        capacity_units=capacity_units,
    )

    run_visualizations(
        forecasting_result=forecasting_result,
        optimization_result=optimization_result,
        optimization_summary=optimization_summary,
    )

    save_reports(
        forecasting_result=forecasting_result,
        next_forecasts=next_forecasts,
        product_evaluation=product_evaluation,
        evaluation_summary=evaluation_summary,
        optimization_result=optimization_result,
        optimization_summary=optimization_summary,
    )

    print("Pipeline completed successfully.")
    print(
        "Best forecasting model:",
        forecasting_result.best_model_name,
    )
    print(
        "Reports saved to:",
        REPORTS_DIR,
    )


if __name__ == "__main__":
    run_pipeline()