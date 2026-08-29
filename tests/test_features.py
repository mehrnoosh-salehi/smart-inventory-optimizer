import pandas as pd
import pytest

from src.features import (
    create_next_period_features,
    create_supervised_features,
)

def test_supervised_features_use_only_past_demand():
    weekly = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2022-01-03",
                periods=10,
                freq="W-MON",
            ),
            "stock_code": ["SKU-001"] * 10,
            "demand": [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                9999,
                100,
            ],
        }
    )

    features = create_supervised_features(weekly)

    first_row = features.iloc[0]

    assert len(features) == 2

    assert first_row["demand"] == 9999
    assert first_row["lag_1"] == 80
    assert first_row["lag_2"] == 70
    assert first_row["lag_4"] == 50
    assert first_row["lag_8"] == 10

    assert first_row["rolling_mean_4"] == 65
    assert first_row["rolling_mean_8"] == 45




def test_supervised_features_keep_product_histories_separate():
    product_a = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2022-01-03",
                periods=9,
                freq="W-MON",
            ),
            "stock_code": ["SKU-001"] * 9,
            "demand": [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
            ],
        }
    )

    product_b = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2022-01-03",
                periods=9,
                freq="W-MON",
            ),
            "stock_code": ["SKU-002"] * 9,
            "demand": [
                1000,
                2000,
                3000,
                4000,
                5000,
                6000,
                7000,
                8000,
                9000,
            ],
        }
    )

    weekly = pd.concat(
        [product_a, product_b],
        ignore_index=True,
    )

    features = create_supervised_features(weekly)

    row_a = features.loc[
        features["stock_code"] == "SKU-001"
    ].iloc[0]

    row_b = features.loc[
        features["stock_code"] == "SKU-002"
    ].iloc[0]

    assert len(features) == 2

    assert row_a["demand"] == 90
    assert row_a["lag_1"] == 80
    assert row_a["lag_8"] == 10

    assert row_b["demand"] == 9000
    assert row_b["lag_1"] == 8000
    assert row_b["lag_8"] == 1000


def test_next_period_features_use_latest_history():
    weekly = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2022-01-03",
                periods=8,
                freq="W-MON",
            ),
            "stock_code": ["SKU-001"] * 8,
            "demand": [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
            ],
        }
    )

    next_features = create_next_period_features(weekly)

    row = next_features.iloc[0]

    assert len(next_features) == 1

    assert row["date"] == pd.Timestamp("2022-02-28")

    assert row["lag_1"] == 80
    assert row["lag_2"] == 70
    assert row["lag_4"] == 50
    assert row["lag_8"] == 10

    assert row["rolling_mean_4"] == 65
    assert row["rolling_mean_8"] == 45


def test_next_period_features_require_enough_history():
    weekly = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2022-01-03",
                periods=4,
                freq="W-MON",
            ),
            "stock_code": ["SKU-001"] * 4,
            "demand": [
                10,
                20,
                30,
                40,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="At least 8 weeks are required",
    ):
        create_next_period_features(weekly)