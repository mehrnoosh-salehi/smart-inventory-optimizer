# Smart Inventory Optimizer

An end-to-end machine learning and operations research project for demand forecasting and inventory allocation under business constraints.

## Project Overview

**Smart Inventory Optimizer** is a decision-support system that combines machine learning and mathematical optimization to improve inventory planning.

The system uses historical retail sales data to forecast next-period product demand, evaluates forecasting performance using a temporal holdout set, and converts future demand forecasts into inventory recommendations under limited budget and storage capacity.

The project is designed as a modular, testable, and reproducible pipeline that covers the complete workflow from raw sales data to business-oriented inventory decisions.

---

## Problem Statement

Inventory planning requires balancing two competing risks:

- **Overstocking:** ordering too much inventory can tie up capital and increase excess-stock risk.
- **Understocking:** ordering too little inventory can cause stockouts and missed sales.

Demand forecasting alone does not fully solve the inventory problem.

A forecasting model can estimate how much demand may occur, but a business must still decide how to allocate limited financial and storage resources across multiple products.

This project addresses both stages:

1. **Demand Forecasting** — estimate next-period demand for each product.
2. **Inventory Optimization** — determine how much of each product should be ordered while respecting business constraints.

---

## Project Goals

The project is designed to:

- preprocess and validate historical retail sales data
- aggregate daily transactions into weekly product demand
- engineer time-series features without data leakage
- train and compare multiple forecasting models
- evaluate forecasts using chronological holdout data
- generate next-period product demand forecasts
- optimize inventory allocation under budget and capacity constraints
- produce business-ready forecasting and optimization reports
- provide a fully tested end-to-end pipeline

---

## End-to-End Pipeline

```text
Raw Sales Data
      |
      v
Data Validation & Cleaning
      |
      v
Weekly Demand Aggregation
      |
      v
Time-Series Feature Engineering
      |
      v
Forecast Model Training
      |
      v
Temporal Holdout Evaluation
      |
      +--------------------------+
      |                          |
      v                          v
Forecast Evaluation       Next-Period Forecast
                                 |
                                 v
                        Inventory Optimization
                                 |
                                 v
                         Business Reports
```

The complete pipeline can be executed with:

```bash
python main.py
```

---

## Project Architecture

```text
smart-inventory-optimizer/
|
├── data/
│   ├── raw/
│   │   └── sample_retail_sales.csv
│   └── processed/
│       └── weekly_sales.csv
|
├── reports/
│   ├── model_metrics.csv
│   ├── next_period_forecasts.csv
│   ├── product_evaluation.csv
│   ├── evaluation_summary.csv
│   ├── inventory_recommendations.csv
│   └── optimization_summary.csv
|
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── forecasting.py
│   ├── evaluation.py
│   └── optimization.py
|
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_forecasting.py
│   ├── test_evaluation.py
│   ├── test_optimization.py
│   └── test_main.py
|
├── notebooks/
├── dashboard/
├── main.py
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

`data/processed/` and `reports/` contain pipeline-generated outputs and are excluded from Git tracking.

---

## Dataset

The project currently uses a sample retail sales dataset containing:

- **18,250 daily observations**
- **25 products**
- historical quantity sold
- product descriptions and categories
- selling prices
- unit costs
- promotion indicators

The raw dataset contains the following columns:

```text
date
stock_code
description
category
quantity
unit_price
unit_cost
promotion
```

The preprocessing pipeline converts the raw date column to a proper datetime type, validates the data, cleans invalid observations, calculates additional business variables, and aggregates daily transactions into weekly product-level demand.

---

## Data Preprocessing

The preprocessing module is implemented in:

```text
src/preprocessing.py
```

Its main workflow is:

```text
load_sales_data()
        |
        v
clean_sales_data()
        |
        v
aggregate_weekly_demand()
```

### Cleaning

The pipeline validates required columns and data types and calculates:

```text
revenue
gross_profit
```

After preprocessing, the daily dataset contains:

```text
18,250 rows
10 columns
```

### Weekly Aggregation

Daily transactions are aggregated into weekly product demand.

The resulting dataset contains:

```text
2,625 weekly product observations
10 columns
```

The weekly dataset includes:

```text
stock_code
description
category
date
demand
unit_price
unit_cost
revenue
gross_profit
promotion_days
```

---

## Feature Engineering

Time-series feature engineering is implemented in:

```text
src/features.py
```

Two feature datasets are created.

### Supervised Training Features

Historical features are created for model training and evaluation.

The current supervised dataset contains:

```text
2,425 rows
26 columns
```

Demand lag features include:

```text
lag_1
lag_2
lag_4
lag_8
```

Rolling demand statistics include:

```text
rolling_mean_4
rolling_std_4
rolling_mean_8
rolling_std_8
```

Calendar and seasonal features include:

```text
week_of_year
month
quarter
year
sin_week
cos_week
```

Additional features include:

```text
price_to_cost_ratio_lag_1
promotion_days_lag_1
```

Historical values are shifted before being used as predictors to prevent future information from leaking into the training data.

### Next-Period Features

A separate feature-generation function creates one feature row per eligible product for future forecasting.

The current pipeline generates:

```text
25 next-period product rows
22 feature columns
```

Future features do not contain the unknown future demand target.

---

## Demand Forecasting

Forecasting is implemented in:

```text
src/forecasting.py
```

The current pipeline compares the following approaches:

- Moving Average baseline
- Ridge Regression
- Random Forest Regression
- Gradient Boosting Regression

### Temporal Train/Test Strategy

Because this is a time-dependent forecasting problem, observations are not randomly shuffled.

The pipeline reserves the most recent weeks for testing:

```text
Minimum training history: 52 weeks
Temporal test period:       8 weeks
```

This preserves chronological ordering and reduces the risk of time-series data leakage.

For the current dataset:

```text
25 products × 8 test weeks = 200 holdout predictions
```

### Model Preprocessing

Numeric features are passed through model-specific preprocessing.

Categorical variables such as product identifiers and categories are handled using one-hot encoding.

Ridge Regression uses standardized numeric features, while tree-based models do not require numeric scaling.

### Model Selection

Models are evaluated using:

- MAE
- RMSE
- MAPE
- Forecast Bias

The selected machine learning model is the one with the lowest RMSE on the temporal holdout period.

After model selection, the winning model is refitted using all available supervised historical observations before generating the future forecast.

---

## Forecasting Results

On the current sample dataset, **Ridge Regression** achieved the best holdout RMSE among the trained machine learning models.

| Model | MAE | RMSE | MAPE (%) | Bias |
|---|---:|---:|---:|---:|
| Ridge Regression | 25.39 | 36.94 | 8.87 | -2.23 |
| Gradient Boosting | 27.15 | 37.61 | 9.24 | -5.41 |
| Moving Average 4 | 27.25 | 38.96 | 9.09 | -11.19 |
| Random Forest | 27.23 | 39.41 | 9.06 | -8.10 |

The moving-average model is used as a baseline benchmark.

For the selected Ridge model:

```text
MAE  ≈ 25.39 units
RMSE ≈ 36.94 units
MAPE ≈ 8.87%
Bias ≈ -2.23 units
```

Bias is calculated as:

```text
prediction - actual demand
```

Therefore, the negative bias indicates a small average tendency toward under-forecasting.

---

## Forecast Evaluation

Detailed forecast diagnostics are implemented in:

```text
src/evaluation.py
```

Evaluation is performed only on holdout observations for which both actual demand and model predictions are available.

The module produces:

### Product-Level Evaluation

Metrics are calculated separately for every product:

```text
stock_code
observations
MAE
MAPE_percent
bias
RMSE
```

This helps identify products for which the forecasting model performs relatively poorly even when overall system performance is acceptable.

### Overall Evaluation

The evaluation summary includes:

```text
observations
products
start_date
end_date
MAE
RMSE
MAPE_percent
bias
zero_demand_observations
over_forecast_count
under_forecast_count
exact_forecast_count
over_forecast_rate
under_forecast_rate
exact_forecast_rate
```

For the current temporal holdout:

```text
Observations:         200
Products:              25
Test period:   2023-11-06 to 2023-12-25
Over-forecast rate:   52%
Under-forecast rate:  48%
```

---

## Next-Period Forecast

After selecting and refitting the best model, the system predicts demand for the next available weekly period.

For the current dataset, the next forecast period is:

```text
2024-01-01
```

The forecasting module produces one prediction for each of the 25 products.

Example high-demand predictions from the current run include:

| Product | Predicted Demand |
|---|---:|
| SKU-005 | 707.50 |
| SKU-010 | 479.77 |
| SKU-004 | 455.85 |
| SKU-006 | 436.14 |
| SKU-022 | 409.89 |

Negative model predictions are clipped to zero because negative demand is not meaningful in the inventory context.

---

## Inventory Optimization

Inventory optimization is implemented in:

```text
src/optimization.py
```

The optimization stage converts forecasted demand into inventory purchasing recommendations.

For each product, the decision variable represents the recommended inventory quantity:

```text
x_i = recommended quantity for product i
```

### Objective

The objective is to maximize expected gross profit:

```text
Maximize:

Σ unit_margin_i × x_i
```

where:

```text
unit_margin = unit_price - unit_cost
```

SciPy's `linprog` solves minimization problems, so the implementation minimizes negative unit margins internally.

### Budget Constraint

```text
Σ unit_cost_i × x_i ≤ budget
```

### Capacity Constraint

```text
Σ x_i ≤ capacity_units
```

### Demand Constraint

For every product:

```text
0 ≤ x_i ≤ predicted_demand_i
```

This prevents the optimizer from recommending quantities greater than forecast demand.

The optimization problem is solved using:

```text
scipy.optimize.linprog
method = "highs"
```

---

## Optimization Scenario

The default pipeline currently uses:

```text
Budget:          250,000
Storage capacity: 5,000 units
```

For reference, the full next-period forecast is approximately:

```text
Total predicted demand: 8,168.83 units
Estimated full-demand procurement cost: 385,845.54
```

The default scenario therefore creates meaningful resource constraints and requires the optimizer to prioritize products.

---

## Optimization Results

For the current default scenario:

```text
Total predicted demand:       8,168.83 units
Recommended quantity:         5,000.00 units
Unmet predicted demand:       3,168.83 units

Total allocated cost:       250,000.00
Remaining budget:                  ~0
Budget utilization:              100%

Remaining capacity:                 0
Capacity utilization:            100%

Overall demand coverage:        61.21%
Expected gross profit:      180,426.90
```

Both the budget and capacity constraints are binding in the current scenario.

The optimizer allocates limited resources toward the combination of products that maximizes the defined expected-profit objective while respecting all constraints.

---

## Continuous Optimization

The current optimization model uses continuous Linear Programming.

As a result, recommendations may contain fractional quantities such as:

```text
707.50 units
```

This is mathematically valid for the current LP formulation.

For inventory systems where products must be ordered only in whole units, a future version can replace the continuous model with Mixed Integer Linear Programming (MILP).

---

## Generated Reports

Running:

```bash
python main.py
```

generates the processed dataset and the following analytical reports.

### `model_metrics.csv`

Comparison of forecasting models using MAE, RMSE, MAPE, and Bias.

### `next_period_forecasts.csv`

Future demand predictions for each product.

### `product_evaluation.csv`

Product-level forecasting performance.

### `evaluation_summary.csv`

Overall forecasting performance and forecast-direction diagnostics.

### `inventory_recommendations.csv`

Product-level inventory recommendations including:

```text
predicted_demand
unit_price
unit_cost
unit_margin
recommended_quantity
allocated_cost
expected_profit
demand_coverage
```

### `optimization_summary.csv`

Business-level optimization KPIs including:

```text
total_predicted_demand
total_recommended_quantity
unmet_predicted_demand
total_allocated_cost
remaining_budget
budget_utilization
remaining_capacity
capacity_utilization
overall_demand_coverage
total_expected_profit
```

Generated files inside `data/processed/` and `reports/` are excluded from Git tracking and can be reproduced by rerunning the pipeline.

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/mehrnoosh-salehi/smart-inventory-optimizer.git
```

```bash
cd smart-inventory-optimizer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

On Linux or WSL:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the full pipeline

```bash
python main.py
```

A successful run prints information similar to:

```text
Pipeline completed successfully.
Best forecasting model: ridge_regression
Reports saved to: .../smart-inventory-optimizer/reports
```

---

## Testing

The project contains unit tests for individual modules as well as an end-to-end integration test.

Run the complete test suite with:

```bash
pytest -q
```

Current test status:

```text
30 passed
```

The tests cover:

- preprocessing validation and aggregation
- feature engineering behavior
- forecasting and temporal evaluation
- prediction generation
- forecast diagnostic calculations
- optimization constraints and objective behavior
- optimization summaries
- duplicate and invalid input handling
- full end-to-end pipeline execution
- generated report validation
- budget and capacity constraint compliance

The integration test uses temporary directories so testing does not overwrite the project's real generated reports.

---

## Technology Stack

The project currently uses:

```text
Python
NumPy
Pandas
scikit-learn
SciPy
Matplotlib
Pytest
Git
GitHub
WSL
VS Code
```

Main machine learning and optimization tools include:

```text
Ridge Regression
Random Forest Regression
Gradient Boosting Regression
One-Hot Encoding
Standard Scaling
Temporal Holdout Validation
Linear Programming
SciPy HiGHS Solver
```

---

## Software Design

The project follows a modular architecture.

Each module has a focused responsibility:

```text
preprocessing.py
→ loading, validation, cleaning, and weekly aggregation

features.py
→ historical and future time-series feature engineering

forecasting.py
→ model preprocessing, training, evaluation, selection, and prediction

evaluation.py
→ forecast error diagnostics and performance summaries

optimization.py
→ mathematical optimization and business-level allocation summaries

main.py
→ end-to-end pipeline orchestration
```

Business logic is separated from pipeline orchestration so that individual components can be independently tested and reused.

---

## Reproducibility

Randomized machine learning components use a fixed random state:

```text
RANDOM_STATE = 42
```

The forecasting test split is chronological rather than random.

Generated pipeline outputs can be reproduced by running:

```bash
python main.py
```

---

## Current Limitations

The current version intentionally keeps several assumptions simple:

- optimization quantities are continuous rather than integer-valued
- the optimization objective uses unit margin and does not yet explicitly model holding costs
- safety stock is not currently included
- supplier lead times are not explicitly modeled
- supplier-specific minimum order quantities are not included
- stockout penalties are not explicitly modeled
- the current forecasting workflow uses a single temporal holdout rather than rolling cross-validation
- hyperparameter tuning is currently limited
- the dashboard layer has not yet been implemented

These limitations provide clear directions for future development.

---

## Future Improvements

Potential extensions include:

- Mixed Integer Linear Programming for whole-unit inventory decisions
- safety stock modeling
- holding and stockout costs
- supplier lead times
- minimum order quantities
- supplier and warehouse constraints
- rolling-origin time-series cross-validation
- automated hyperparameter tuning
- probabilistic demand forecasting
- prediction intervals and uncertainty-aware optimization
- model persistence
- configuration files for business parameters
- interactive dashboard development
- scenario analysis for multiple budget and capacity levels
- automated reporting and visualization

---

## Key Learning Outcomes

This project demonstrates the integration of multiple technical areas:

```text
Data Engineering
        +
Time-Series Feature Engineering
        +
Machine Learning
        +
Model Evaluation
        +
Operations Research
        +
Software Testing
        +
Git-Based Development
```

The central idea is that accurate prediction is only one part of a decision-support system.

The project connects:

```text
"What is likely to happen?"
```

with:

```text
"What should the business do given limited resources?"
```

---

## Project Status

The core end-to-end pipeline is operational.

Current completed components include:

```text
Data preprocessing            ✓
Weekly demand aggregation     ✓
Feature engineering           ✓
Forecast model comparison     ✓
Temporal holdout evaluation   ✓
Next-period forecasting       ✓
Forecast diagnostics          ✓
Inventory optimization        ✓
Business report generation    ✓
Unit testing                  ✓
Integration testing           ✓
End-to-end execution          ✓
```

The current automated test suite contains:

```text
30 passing tests
```

The next development stages can focus on visualization, dashboard development, model improvements, and more advanced inventory constraints.

---

## Author

**Mehrnoosh Salehi**

GitHub:

```text
https://github.com/mehrnoosh-salehi
```