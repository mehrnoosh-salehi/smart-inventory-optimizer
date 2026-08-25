import pandas as pd

from src.preprocessing import (
    clean_sales_data,
    aggregate_weekly_demand
)

def test_clean_sales_data_removes_invalid_rows():
    data = pd.DataFrame(
        {
            "date": [
                "2022-01-01",
                "2022-01-02",
                "2022-01-03",
            ],
            "stock_code": [
                "SKU-001",
                "SKU-001",
                "SKU-001",
            ],
            "quantity": [
                10,
                -5,
                20,
            ],
            "unit_price": [
                100,
                100,
                100,
            ],
            "unit_cost": [
                60,
                60,
                60,
            ],
        }
    )

    cleaned = clean_sales_data(data)

    assert len(cleaned) == 2
    assert (cleaned["quantity"] >= 0).all()


def test_aggregate_weekly_demand_sums_daily_sales():
    data = pd.DataFrame(
        {
            "date": [
                "2022-01-03",
                "2022-01-04",
                "2022-01-05",
            ],
            "stock_code": [
                "SKU-001",
                "SKU-001",
                "SKU-001",
            ],
            "description": [
                "Product A",
                "Product A",
                "Product A",
            ],
            "category": [
                "Home",
                "Home",
                "Home",
            ],
            "quantity": [
                10,
                20,
                30,
            ],
            "unit_price": [
                100,
                100,
                100,
            ],
            "unit_cost": [
                60,
                60,
                60,
            ],
            "revenue": [
                1000,
                2000,
                3000,
            ],
            "gross_profit": [
                400,
                800,
                1200,
            ],
            "promotion": [
                0,
                1,
                0,
            ],
        }
    )

    data["date"] = pd.to_datetime(data["date"])
    weekly = aggregate_weekly_demand(data)

    assert len(weekly) == 1
    assert weekly["demand"].iloc[0] == 60