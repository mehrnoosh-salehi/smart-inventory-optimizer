"""feature engineering for weekly product-demand forcasting"""
from __future__ import annotations
import numpy as np
import pandas as pd

LAGS = [1, 2, 4, 8]
ROLLING_WINDOWS = [4, 8]

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and cyclical time features."""

    out = df.copy()

    isocalender = out["date"].dt.isocalendar()

    out["week_of_year"] = isocalender.week.astype(int)
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["year"] = out["date"].dt.year

    out["sin_week"] = np.sin(2 * np.pi * out["week_of_year"] / 52)
    out["cos_week"] = np.cos(2 * np.pi * out["week_of_year"] / 52)

    return out

def create_supervised_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe supervised features from weekly product demand."""

    df = (
        weekly
        .copy()
        .sort_values(["stock_code", "date"])
        .reset_index(drop=True)
    )


    feature_columns = []

    for lag in LAGS:
        column_name = f"lag_{lag}"
        df[column_name] = ( df
                           .groupby("stock_code")["demand"]
                           .shift(lag)
        )
        feature_columns.append(column_name)

    for window in ROLLING_WINDOWS:
        mean_column = f"rolling_mean_{window}"
        std_column = f"rolling_std_{window}"

        df[mean_column] = (
            df
            .groupby("stock_code")["demand"]
            .transform(lambda series: series.shift(1)
            .rolling(window)
            .mean())
        )

        df[std_column] = (
            df.groupby("stock_code")["demand"]
            .transform(lambda series: series
                       .shift(1)
                       .rolling(window)
                       .std()
                       )
        )

        feature_columns.extend([mean_column, std_column])

    df = _add_time_features(df)

    if {"unit_price", "unit_cost"}.issubset(df.columns):
        previous_price = (
            df
            .groupby("stock_code")["unit_price"]
            .shift(1)
        )

        previous_cost = (
            df
            .groupby("stock_code")["unit_cost"]
            .shift(1)
        )

        df["price_to_cost_ratio_lag_1"] = (
            previous_price / previous_cost
        )

        feature_columns.append("price_to_cost_ratio_lag_1")

    if "promotion_days" in df.columns:
        df["promotion_days_lag_1"] = (
            df
            .groupby("stock_code")["promotion_days"]
            .shift(1)
            )

        feature_columns.append("promotion_days_lag_1")

    df = (
        df
        .dropna(subset=feature_columns)
        .reset_index(drop=True)
        )

    return df


def create_next_period_features(
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    """Create one feature row per product for next-week forecasting."""

    data = (
        weekly
        .copy()
        .sort_values(["stock_code", "date"])
        .reset_index(drop=True)
    )

    required_history = max(
        max(LAGS),
        max(ROLLING_WINDOWS),
    )

    rows = []

    for stock_code, group in data.groupby(
        "stock_code",
        sort=False,
    ):
        group = group.sort_values("date")

        if len(group) < required_history:
            raise ValueError(
                f"Not enough history for product {stock_code}. "
                f"At least {required_history} weeks are required."
            )

        last_row = group.iloc[-1]
        demand_history = group["demand"]

        next_date = (
            pd.to_datetime(last_row["date"])
            + pd.Timedelta(weeks=1)
        )

        row = {
            "date": next_date,
            "stock_code": stock_code,
        }

        for column in [
            "description",
            "category",
            "unit_price",
            "unit_cost",
        ]:
            if column in group.columns:
                row[column] = last_row[column]

        for lag in LAGS:
            row[f"lag_{lag}"] = float(
                demand_history.iloc[-lag]
            )

        for window in ROLLING_WINDOWS:
            recent_demand = demand_history.iloc[-window:]

            row[f"rolling_mean_{window}"] = float(
                recent_demand.mean()
            )

            row[f"rolling_std_{window}"] = float(
                recent_demand.std()
            )

        if {
            "unit_price",
            "unit_cost",
        }.issubset(group.columns):
            row["price_to_cost_ratio_lag_1"] = float(
                last_row["unit_price"]
                / last_row["unit_cost"]
            )

        if "promotion_days" in group.columns:
            row["promotion_days_lag_1"] = float(
                last_row["promotion_days"]
            )

        rows.append(row)

    next_features = pd.DataFrame(rows)

    next_features = _add_time_features(
        next_features
    )

    return (
        next_features
        .sort_values("stock_code")
        .reset_index(drop=True)
    )