"""Data preprocessing pipeline for retail sales data."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

# Columns required for the forcasting pipeline to work
CORE_COLUMNS = {
    "date",
    "stock_code",
    "quantity"
}

# Columns that improve the model or support downstream optimization
OPTIONAL_COLUMNS = {
    "description",
    "category",
    "unit_price",
    "unit_cost",
    "promotion"
}

def load_sales_data(path: str | Path) -> pd.DataFrame:
    """Load raw sales data and validate the input schema."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Sales data file not found: {path}"
        )

    df = pd.read_csv(path)

    missing_columns = CORE_COLUMNS.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(df["date"])

    return df


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw retail sales data and create business metrics."""

    clean = df.copy()

    clean = clean.drop_duplicates()

    clean = clean.dropna(
        subset=[
            "date",
            "stock_code",
            "quantity",
            "unit_price",
            "unit_cost",
        ]
    )

    clean = clean[clean["quantity"] >= 0]

    clean = clean[clean["unit_price"] > 0]

    clean = clean[clean["unit_cost"] > 0]

    clean["revenue"] = (
        clean["quantity"] * clean["unit_price"]
    )

    clean["gross_profit"] = (
        clean["quantity"]
        * (clean["unit_price"] - clean["unit_cost"])
    )

    return (
        clean
        .sort_values(["stock_code", "date"])
        .reset_index(drop=True)
    )

def aggregate_weekly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level sales into weekly product demand"""
    data = df.copy()
    data["week_start"] = (
        data["date"]
        .dt.to_period("W-SUN")
        .apply(lambda period: period.start_time)

    )

    weekly = (
        data.groupby(
        ["stock_code",
         "description",
         "category",
         "week_start"],

        as_index=False
        )
    ).agg(
        demand = ("quantity", "sum"),
        unit_price = ("unit_price", "mean"),
        unit_cost = ("unit_cost", "mean"),
        revenue = ("revenue", "sum"),
        gross_profit = ("gross_profit", "sum"),
        promotion_days = ("promotion", "sum")
    )

    weekly =weekly.rename(
        columns={
            "week_start": "date"
        }
    )

    weekly["date"] = pd.to_datetime(weekly["date"])

    return weekly.sort_values(
        [
            "stock_code",
            "date"
        ]
    ).reset_index(drop=True)



