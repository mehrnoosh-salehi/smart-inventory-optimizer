import pandas as pd

import main


def test_run_pipeline_creates_expected_outputs(
    tmp_path,
    monkeypatch,
):
    """Run the full pipeline and verify its generated outputs."""

    processed_path = (
        tmp_path
        / "data"
        / "processed"
        / "weekly_sales.csv"
    )

    reports_dir = (
        tmp_path
        / "reports"
    )

    figures_dir = (
        reports_dir
        / "figures"
    )

    monkeypatch.setattr(
        main,
        "PROCESSED_DATA_PATH",
        processed_path,
    )

    monkeypatch.setattr(
        main,
        "REPORTS_DIR",
        reports_dir,
    )

    monkeypatch.setattr(
        main,
        "FIGURES_DIR",
        figures_dir,
    )

    main.run_pipeline(
        budget=250_000,
        capacity_units=5_000,
    )

    assert processed_path.exists()
    assert processed_path.stat().st_size > 0

    expected_report_files = [
        "model_metrics.csv",
        "next_period_forecasts.csv",
        "product_evaluation.csv",
        "evaluation_summary.csv",
        "inventory_recommendations.csv",
        "optimization_summary.csv",
    ]

    for filename in expected_report_files:
        report_path = (
            reports_dir
            / filename
        )

        assert report_path.exists()
        assert report_path.stat().st_size > 0

    expected_figure_files = [
        "model_rmse.png",
        "holdout_actual_vs_predicted.png",
        "forecast_vs_recommendation.png",
        "resource_utilization.png",
    ]

    for filename in expected_figure_files:
        figure_path = (
            figures_dir
            / filename
        )

        assert figure_path.exists()
        assert figure_path.stat().st_size > 0

    optimization_summary = pd.read_csv(
        reports_dir
        / "optimization_summary.csv"
    )

    summary_row = optimization_summary.iloc[0]

    assert (
        summary_row["total_allocated_cost"]
        <= 250_000 + 1e-6
    )

    assert (
        summary_row["total_recommended_quantity"]
        <= 5_000 + 1e-6
    )