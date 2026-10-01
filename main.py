"""Run the Smart Inventory Optimizer end-to-end pipeline."""

from pathlib import Path

import pandas as pd

from src.preprocessing import(
    load_sales_data, 
    clean_sales_data, 
    aggregate_weekly_demand
)

from src.features import(
    create_next_period_features,
    create_supervised_features
)

from src.forecasting import(
    train_and_evaluate_models,
    predict_next_period
)

from src.evaluation import(
    evaluate_by_product,
    summarize_evaluation
)

from src.optimization import(
    optimize_inventory,
    summarize_optimization
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

def run_preprocessing() -> pd.DataFrame:
    """Load, clean, aggregate, and save the sales data."""

    raw_data = load_sales_data(RAW_DATA_PATH)

    clean_data = clean_sales_data(raw_data)

    weekly_data = aggregate_weekly_demand(clean_data)

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    weekly_data.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    return weekly_data
