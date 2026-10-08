# Smart Inventory Optimizer

An end-to-end machine learning and operations research project for demand forecasting, inventory optimization, visualization, and decision support under real-world business constraints.

## Project Overview

**Smart Inventory Optimizer** is a modular decision-support system that combines machine learning and mathematical optimization to improve inventory planning.

The system uses historical retail sales data to:

1. clean and validate raw sales data
2. aggregate daily transactions into weekly product demand
3. engineer time-series forecasting features
4. train and compare multiple forecasting approaches
5. evaluate model performance using chronological holdout data
6. forecast next-period product demand
7. optimize inventory allocation under budget and capacity constraints
8. generate analytical CSV reports
9. generate forecasting and optimization visualizations

The project is designed as a reproducible, testable, and modular end-to-end pipeline.

---

## Problem Statement

Inventory planning requires balancing two competing risks:

- **Overstocking:** ordering too much inventory can tie up capital and increase excess-stock risk.
- **Understocking:** ordering too little inventory can lead to stockouts, missed sales, and lost revenue.

Demand forecasting alone does not fully solve this problem.

A forecasting model can estimate how much demand may occur, but a business must still decide how to allocate limited resources across multiple products.

For example, forecasted demand may exceed what the company can afford to purchase or store.

This project therefore combines two stages:

1. **Demand Forecasting**
   - Estimate future product demand.

2. **Inventory Optimization**
   - Decide how much inventory should actually be allocated while respecting business constraints.

The project connects prediction with decision-making.

---

## Project Goals

The project is designed to:

- preprocess and validate historical retail sales data
- aggregate daily observations into weekly product demand
- engineer time-series features without future data leakage
- train and compare multiple forecasting models
- evaluate predictions on chronological holdout data
- generate next-period product demand forecasts
- optimize inventory allocation under financial and capacity constraints
- produce business-oriented forecasting and optimization reports
- generate presentation-ready analytical visualizations
- validate individual components with unit tests
- validate the complete system with end-to-end integration testing

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
                         Visualization Layer
                                 |
                                 v
                     CSV Reports + PNG Figures
```

The complete pipeline can be executed with:

```bash
python main.py
```

A single pipeline run performs preprocessing, feature engineering, forecasting, evaluation, inventory optimization, visualization, and report generation.

---

## Project Architecture

```text
smart-inventory-optimizer/
|
├── data/
│   ├── raw/
│   │   └── sample_retail_sales.csv
│   │
│   └── processed/
│       └── weekly_sales.csv
|
├── reports/
│   ├── figures/
│   │   ├── model_rmse.png
│   │   ├── holdout_actual_vs_predicted.png
│   │   ├── forecast_vs_recommendation.png
│   │   └── resource_utilization.png
│   │
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
│   ├── optimization.py
│   └── visualization.py
|
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_forecasting.py
│   ├── test_evaluation.py
│   ├── test_optimization.py
│   ├── test_visualization.py
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

`data/processed/`, `reports/`, and generated visualization files are pipeline outputs and are excluded from Git tracking.

They can be reproduced by rerunning the pipeline.

---

# Dataset

The project currently uses a sample retail sales dataset containing:

```text
18,250 daily observations
25 products
8 input columns
```

The raw dataset contains:

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

The dataset contains historical information about:

- transaction dates
- product identifiers
- product descriptions
- product categories
- quantity sold
- selling price
- unit cost
- promotion status

The preprocessing pipeline validates and transforms this raw dataset before it is used for forecasting.

---

# Data Preprocessing

Data preprocessing is implemented in:

```text
src/preprocessing.py
```

The main preprocessing workflow is:

```text
load_sales_data()
        |
        v
clean_sales_data()
        |
        v
aggregate_weekly_demand()
```

---

## Data Loading

The loading stage:

- verifies that the input file exists
- reads the CSV dataset
- checks required columns
- converts the date column to datetime

Core required columns include:

```text
date
stock_code
quantity
```

Additional supported columns include:

```text
description
category
unit_price
unit_cost
promotion
```

---

## Data Cleaning

The cleaning stage:

- removes duplicate rows
- removes observations with missing key values
- validates non-negative quantities
- validates positive prices and costs
- calculates additional business metrics
- sorts observations chronologically

Additional calculated variables include:

```text
revenue
gross_profit
```

where revenue is based on quantity and selling price, while gross profit is based on the difference between selling price and unit cost.

After preprocessing, the cleaned daily dataset contains approximately:

```text
18,250 rows
10 columns
```

---

## Weekly Demand Aggregation

Daily product observations are aggregated into weekly demand.

The resulting weekly dataset contains approximately:

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

Weekly aggregation provides a more appropriate time scale for demand forecasting and inventory planning.

---

# Feature Engineering

Time-series feature engineering is implemented in:

```text
src/features.py
```

The project generates two different feature datasets:

1. supervised historical features
2. next-period forecasting features

---

## Historical Supervised Features

The supervised feature dataset is used for:

```text
Model Training
+
Temporal Holdout Evaluation
```

The current supervised dataset contains approximately:

```text
2,425 rows
26 columns
```

---

## Demand Lag Features

Historical demand lags include:

```text
lag_1
lag_2
lag_4
lag_8
```

These variables represent demand from previous weeks.

For example:

```text
lag_1
```

represents demand from the previous week.

---

## Rolling Features

Rolling demand statistics include:

```text
rolling_mean_4
rolling_std_4
rolling_mean_8
rolling_std_8
```

These features summarize recent demand behavior.

---

## Calendar Features

Calendar and seasonal variables include:

```text
week_of_year
month
quarter
year
```

Seasonality is also represented using cyclical transformations:

```text
sin_week
cos_week
```

These variables help represent the cyclical nature of weekly seasonality.

---

## Additional Business Features

Additional features include:

```text
price_to_cost_ratio_lag_1
promotion_days_lag_1
```

These variables allow the forecasting models to incorporate recent pricing and promotion information.

---

## Data Leakage Prevention

Historical values are shifted before they are used as predictors.

This prevents future information from leaking into historical model training.

For example, rolling features are calculated using previously observed values rather than future demand.

This is especially important in time-series forecasting.

---

# Next-Period Feature Engineering

A separate feature-generation workflow is used for future forecasting.

The next-period dataset currently contains:

```text
25 product rows
22 feature columns
```

There is one row for each eligible product.

Unlike the supervised training dataset, the next-period feature table does not contain the unknown future demand target.

The goal is:

```text
Historical observations
        ↓
Next-period features
        ↓
Trained forecasting model
        ↓
Future demand prediction
```

---

# Demand Forecasting

Demand forecasting is implemented in:

```text
src/forecasting.py
```

The forecasting pipeline compares several approaches.

Current models include:

- Moving Average baseline
- Ridge Regression
- Random Forest Regression
- Gradient Boosting Regression

---

## Moving Average Baseline

A moving-average forecast is included as a baseline model.

The baseline provides a simple reference point against which machine learning models can be compared.

A more complex model should ideally outperform a reasonable baseline.

---

## Ridge Regression

Ridge Regression is a linear regression model with L2 regularization.

Regularization helps reduce sensitivity to highly correlated or noisy predictors.

The current dataset identifies Ridge Regression as the best-performing machine learning model based on temporal holdout RMSE.

---

## Random Forest Regression

Random Forest combines multiple decision trees and averages their predictions.

It can model nonlinear relationships without requiring explicit feature scaling.

---

## Gradient Boosting Regression

Gradient Boosting builds models sequentially, with later models attempting to correct errors made by earlier models.

It can capture nonlinear relationships between engineered time-series features and demand.

---

# Temporal Train/Test Strategy

Because this is a time-dependent forecasting problem, observations are not randomly shuffled before evaluation.

Instead, the latest observations are reserved as the test period.

Current configuration:

```text
Minimum training history: 52 weeks
Temporal test period:       8 weeks
```

For the current dataset:

```text
25 products × 8 test weeks
=
200 holdout predictions
```

This chronological split more closely represents a real forecasting scenario:

```text
Past
→ Train model

Future
→ Evaluate model
```

Random train/test splitting could leak future information into the training set and produce overly optimistic performance estimates.

---

# Model Preprocessing

The forecasting pipeline uses preprocessing pipelines for machine learning models.

Categorical features are encoded using:

```text
One-Hot Encoding
```

Numeric features used by Ridge Regression are standardized before training.

Tree-based models such as Random Forest and Gradient Boosting do not require numeric scaling in the same way.

Using machine learning pipelines helps ensure that preprocessing is applied consistently during both training and prediction.

---

# Forecast Evaluation Metrics

Forecasting models are evaluated using several metrics.

---

## Mean Absolute Error — MAE

MAE represents the average absolute difference between actual and predicted demand.

Lower MAE indicates better predictions.

---

## Root Mean Squared Error — RMSE

RMSE penalizes larger prediction errors more strongly than MAE.

The best machine learning model is selected using holdout RMSE.

Lower RMSE is better.

---

## Mean Absolute Percentage Error — MAPE

MAPE measures forecast error relative to actual demand.

It is useful for interpreting forecasting performance as a percentage.

---

## Forecast Bias

Forecast bias is calculated as:

```text
prediction - actual demand
```

Interpretation:

```text
Positive bias
→ tendency toward over-forecasting

Negative bias
→ tendency toward under-forecasting
```

---

# Forecasting Results

On the current sample dataset, **Ridge Regression** achieved the lowest temporal holdout RMSE among the trained machine learning models.

| Model | MAE | RMSE | MAPE (%) | Bias |
|---|---:|---:|---:|---:|
| Ridge Regression | 25.39 | 36.94 | 8.87 | -2.23 |
| Gradient Boosting | 27.15 | 37.61 | 9.24 | -5.41 |
| Moving Average 4 | 27.25 | 38.96 | 9.09 | -11.19 |
| Random Forest | 27.23 | 39.41 | 9.06 | -8.10 |

The Moving Average model is included as a baseline comparison.

For Ridge Regression:

```text
MAE  ≈ 25.39 units
RMSE ≈ 36.94 units
MAPE ≈ 8.87%
Bias ≈ -2.23 units
```

The small negative bias indicates a slight average tendency toward under-forecasting.

---

# Forecast Evaluation

Detailed forecast diagnostics are implemented in:

```text
src/evaluation.py
```

The evaluation layer analyzes model performance using the temporal holdout predictions.

---

## Product-Level Evaluation

Metrics are calculated separately for each product.

The resulting table includes:

```text
stock_code
observations
MAE
MAPE_percent
bias
RMSE
```

This makes it possible to identify products that are especially difficult to forecast.

A model may perform well overall while still producing relatively large errors for specific products.

---

## Overall Evaluation

The summary includes:

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

For the current dataset:

```text
Holdout observations: 200
Products:              25
Start date:    2023-11-06
End date:      2023-12-25
```

Forecast direction rates are approximately:

```text
Over-forecast rate:  52%
Under-forecast rate: 48%
```

---

# Next-Period Forecast

After model comparison, the best machine learning model is refitted using all available supervised historical observations.

The fitted model is then used to predict demand for the next available weekly period.

For the current dataset, the next forecast date is:

```text
2024-01-01
```

Predictions are generated for:

```text
25 products
```

The total forecast demand is approximately:

```text
8,168.83 units
```

Example high-demand predictions include:

| Product | Predicted Demand |
|---|---:|
| SKU-005 | 707.50 |
| SKU-010 | 479.77 |
| SKU-004 | 455.85 |
| SKU-006 | 436.14 |
| SKU-022 | 409.89 |

Negative model predictions are clipped to zero because negative product demand is not meaningful.

---

# Inventory Optimization

Inventory optimization is implemented in:

```text
src/optimization.py
```

The optimization layer converts future demand forecasts into inventory allocation recommendations.

This is the main decision-making stage of the project.

---

## Decision Variables

For each product:

```text
x_i
```

represents the recommended quantity to purchase or allocate.

---

## Unit Margin

Expected profit contribution per unit is calculated using:

```text
unit_margin
=
unit_price - unit_cost
```

Products with higher unit margins may contribute more strongly to the optimization objective.

---

## Optimization Objective

The objective is to maximize expected gross profit:

```text
Maximize:

Σ unit_margin_i × x_i
```

SciPy's `linprog` solves minimization problems.

Therefore, the implementation internally minimizes the negative of the profit objective.

---

## Budget Constraint

Total inventory cost cannot exceed the available budget:

```text
Σ unit_cost_i × x_i ≤ budget
```

---

## Capacity Constraint

Total inventory quantity cannot exceed available storage capacity:

```text
Σ x_i ≤ capacity_units
```

---

## Demand Constraint

Recommended inventory cannot exceed predicted demand:

```text
0 ≤ x_i ≤ predicted_demand_i
```

This prevents the optimizer from allocating inventory beyond forecasted requirements.

---

## Solver

The optimization problem is solved using:

```text
scipy.optimize.linprog
```

with:

```text
method = "highs"
```

The HiGHS solver is designed for efficient linear optimization.

---

# Default Optimization Scenario

The current pipeline uses the following default business constraints:

```text
Budget:           250,000
Capacity:           5,000 units
```

The total next-period forecast is approximately:

```text
Predicted demand:
8,168.83 units
```

Estimated cost of satisfying all forecast demand:

```text
385,845.54
```

Because the full-demand procurement cost exceeds the available budget and predicted demand exceeds storage capacity, the optimizer must prioritize inventory allocation.

This creates a meaningful constrained optimization problem.

---

# Optimization Results

For the current default scenario:

```text
Total predicted demand:       8,168.83 units

Recommended inventory:        5,000.00 units

Unmet predicted demand:       3,168.83 units

Allocated cost:             250,000.00

Remaining budget:                  ~0

Budget utilization:              100%

Remaining capacity:                 0

Capacity utilization:            100%

Overall demand coverage:         61.21%

Expected gross profit:       180,426.90
```

Both budget and capacity constraints are binding.

This means the optimizer uses essentially all available budget and storage capacity.

The optimizer selects a combination of product quantities that maximizes the defined expected-profit objective while respecting all constraints.

---

# Visualization Layer

Visualization is implemented in:

```text
src/visualization.py
```

The visualization layer converts forecasting and optimization outputs into presentation-ready PNG figures.

All figures are generated automatically when the end-to-end pipeline is executed:

```bash
python main.py
```

Generated figures are stored in:

```text
reports/figures/
```

The project currently produces four analytical visualizations.

---

## Model RMSE Comparison

Generated file:

```text
model_rmse.png
```

This figure compares the RMSE values of all evaluated forecasting approaches.

The current comparison includes:

- Ridge Regression
- Gradient Boosting Regression
- Random Forest Regression
- Moving Average baseline

Because lower RMSE represents better forecasting performance, this chart provides a quick visual comparison between models.

---

## Holdout Actual vs Predicted Demand

Generated file:

```text
holdout_actual_vs_predicted.png
```

This figure compares:

```text
Actual Demand
vs
Predicted Demand
```

across the temporal holdout weeks.

Product-level predictions are aggregated by week before visualization.

The chart helps identify:

- weeks with over-forecasting
- weeks with under-forecasting
- whether predictions follow overall demand movement
- periods where forecast errors increase

---

## Forecast Demand vs Recommended Inventory

Generated file:

```text
forecast_vs_recommendation.png
```

This grouped bar chart compares:

```text
Predicted Demand
        vs
Recommended Inventory Quantity
```

for high-demand products.

This figure highlights an important distinction between forecasting and optimization.

The forecasting model estimates how much customers may demand.

The optimizer determines how much inventory should actually be allocated after considering limited resources.

Therefore:

```text
Forecast
≠
Final Inventory Decision
```

when business constraints are active.

---

## Resource Utilization

Generated file:

```text
resource_utilization.png
```

This figure displays:

```text
Budget Utilization
Capacity Utilization
```

as percentages.

For the current default scenario:

```text
Budget utilization:   100%
Capacity utilization: 100%
```

Both constraints are therefore binding.

---

## Visualization Input Validation

The visualization module validates its input data before generating figures.

Validation includes checks for:

- required columns
- empty DataFrames
- valid dates
- numeric values
- non-negative quantities
- valid `top_n` values
- valid utilization ranges
- valid optimization-summary structure

This prevents invalid analytical inputs from silently producing misleading plots.

---

# Continuous Optimization

The current inventory model uses continuous Linear Programming.

As a result, recommendations may contain fractional values such as:

```text
707.50 units
```

This is mathematically valid for the current formulation.

For products that must only be ordered in whole units, a future implementation could use Mixed Integer Linear Programming.

That would constrain variables to integer quantities.

---

# Generated Reports

Running:

```bash
python main.py
```

generates both CSV reports and PNG visualizations.

---

## `model_metrics.csv`

Contains forecasting model comparison metrics including:

```text
model
MAE
RMSE
MAPE_percent
bias
```

---

## `next_period_forecasts.csv`

Contains next-period demand predictions for individual products.

Typical fields include:

```text
date
stock_code
description
category
unit_price
unit_cost
predicted_demand
model
```

---

## `product_evaluation.csv`

Contains product-level forecasting performance.

Example metrics include:

```text
observations
MAE
MAPE_percent
bias
RMSE
```

---

## `evaluation_summary.csv`

Contains overall holdout forecasting diagnostics.

---

## `inventory_recommendations.csv`

Contains product-level inventory optimization recommendations.

Important fields include:

```text
stock_code
predicted_demand
unit_price
unit_cost
unit_margin
recommended_quantity
allocated_cost
expected_profit
demand_coverage
```

---

## `optimization_summary.csv`

Contains business-level optimization KPIs.

Important fields include:

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

---

# Generated Figures

The visualization pipeline generates:

```text
reports/figures/model_rmse.png
reports/figures/holdout_actual_vs_predicted.png
reports/figures/forecast_vs_recommendation.png
reports/figures/resource_utilization.png
```

These visualizations provide graphical views of:

```text
forecasting model performance
holdout forecasting behavior
forecast-versus-allocation decisions
budget utilization
capacity utilization
```

Generated files inside `data/processed/`, `reports/`, and `reports/figures/` are excluded from Git tracking and can be reproduced by rerunning the pipeline.

---

# Running the Project

## 1. Clone the Repository

```bash
git clone https://github.com/mehrnoosh-salehi/smart-inventory-optimizer.git
```

Then:

```bash
cd smart-inventory-optimizer
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

---

## 3. Activate the Environment

On Linux or WSL:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Run the Complete Pipeline

```bash
python main.py
```

A successful execution prints output similar to:

```text
Pipeline completed successfully.
Best forecasting model: ridge_regression
Reports saved to: .../smart-inventory-optimizer/reports
```

The pipeline automatically generates:

```text
Processed weekly dataset
+
Forecasting reports
+
Evaluation reports
+
Optimization reports
+
Visualization figures
```

---

# Testing

The project contains unit tests for individual modules and an end-to-end integration test.

Run all tests using:

```bash
pytest -q
```

Current test status:

```text
43 passed
```

The test suite covers:

- preprocessing validation
- data cleaning
- weekly aggregation
- feature engineering
- historical lag generation
- next-period feature generation
- forecasting preprocessing
- forecasting model training
- temporal holdout behavior
- prediction generation
- forecast evaluation
- optimization input validation
- optimization constraints
- optimization summaries
- visualization input validation
- PNG figure generation
- invalid visualization parameter handling
- end-to-end visualization integration
- generated CSV validation
- generated PNG validation
- budget constraint compliance
- capacity constraint compliance
- complete pipeline execution

---

# Visualization Tests

Visualization tests are implemented in:

```text
tests/test_visualization.py
```

The visualization test suite currently contains:

```text
13 test cases
```

These tests verify:

- successful RMSE chart generation
- rejection of missing RMSE columns
- successful holdout chart generation
- rejection of invalid dates
- successful forecast-versus-recommendation chart generation
- validation of `top_n`
- rejection of invalid `top_n` values
- successful resource-utilization chart generation
- rejection of multi-row optimization summaries
- rejection of invalid utilization ranges
- creation of non-empty PNG files

Parameterized testing is used to test multiple invalid `top_n` inputs efficiently.

---

# End-to-End Integration Test

The end-to-end integration test is implemented in:

```text
tests/test_main.py
```

The test executes the complete pipeline using temporary output directories.

It verifies that the pipeline produces:

```text
weekly_sales.csv
```

plus six CSV reports and four PNG figures.

It also verifies that generated files are not empty.

The integration test checks two important business constraints:

```text
total_allocated_cost
≤
available budget
```

and:

```text
total_recommended_quantity
≤
available capacity
```

Temporary directories are used so automated tests do not overwrite the project's real generated outputs.

---

# Technology Stack

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

Main machine learning and optimization techniques include:

```text
Ridge Regression
Random Forest Regression
Gradient Boosting Regression
Moving Average Baseline
One-Hot Encoding
Feature Scaling
Time-Series Lag Features
Rolling Statistics
Temporal Holdout Validation
Linear Programming
SciPy HiGHS Solver
```

---

# Software Design

The project follows a modular architecture.

Each module has a focused responsibility.

```text
preprocessing.py
→ data loading, validation, cleaning, and weekly aggregation

features.py
→ historical and future time-series feature engineering

forecasting.py
→ preprocessing pipelines, model training, evaluation,
  selection, refitting, and future prediction

evaluation.py
→ forecast-error diagnostics and performance summaries

optimization.py
→ inventory allocation optimization and business summaries

visualization.py
→ forecasting and optimization visualization generation

main.py
→ end-to-end pipeline orchestration
```

This separation keeps business logic independent from pipeline orchestration.

Individual components can therefore be tested and reused independently.

---

# Main Pipeline Orchestration

`main.py` coordinates the major project stages:

```text
run_preprocessing()
        ↓
run_feature_engineering()
        ↓
run_forecasting()
        ↓
run_evaluation()
        ↓
run_optimization()
        ↓
run_visualizations()
        ↓
save_reports()
```

The `main.py` file acts as the high-level orchestrator rather than containing forecasting or optimization logic directly.

This improves readability and maintainability.

---

# Reproducibility

Randomized machine learning components use a fixed random state:

```text
42
```

The forecasting test split is chronological rather than random.

Generated analytical outputs can be reproduced by running:

```bash
python main.py
```

The project also uses automated tests to detect unintended behavior changes.

---

# Current Limitations

The current implementation intentionally keeps several assumptions relatively simple.

Current limitations include:

- inventory recommendations are continuous rather than integer-valued
- holding costs are not explicitly modeled
- stockout penalties are not explicitly modeled
- safety stock is not currently included
- supplier lead times are not explicitly modeled
- minimum order quantities are not included
- supplier-specific constraints are not included
- warehouse-specific capacity constraints are not included
- the forecasting workflow currently uses a single temporal holdout
- rolling-origin cross-validation has not yet been implemented
- hyperparameter tuning is limited
- forecast uncertainty is not yet incorporated into optimization
- the dashboard layer has not yet been implemented

These limitations provide clear directions for future development.

---

# Future Improvements

Potential future extensions include:

- Mixed Integer Linear Programming
- whole-unit inventory decisions
- supplier minimum order quantities
- safety stock modeling
- holding cost modeling
- stockout cost modeling
- supplier lead-time constraints
- multiple warehouse constraints
- service-level constraints
- rolling-origin time-series cross-validation
- automated hyperparameter tuning
- probabilistic demand forecasting
- prediction intervals
- uncertainty-aware optimization
- model persistence
- configurable business parameters
- scenario analysis
- automated report generation
- richer analytical visualizations
- interactive dashboard development

---

# Planned Dashboard

A future interactive dashboard can provide business users with direct access to forecasting and optimization results.

Potential dashboard components include:

```text
Forecasting KPIs
Model comparison
Actual vs predicted demand
Product-level forecasts
Inventory recommendations
Expected profit
Budget utilization
Capacity utilization
Demand coverage
Scenario controls
```

Potential user inputs may include:

```text
Available budget
Storage capacity
Number of products to display
Product filters
```

The dashboard layer is planned as a future development stage and is not yet part of the current production pipeline.

---

# Key Learning Outcomes

This project demonstrates the integration of multiple technical disciplines:

```text
Data Engineering
        +
Time-Series Feature Engineering
        +
Machine Learning
        +
Forecast Evaluation
        +
Operations Research
        +
Data Visualization
        +
Software Testing
        +
Git-Based Development
```

The project demonstrates that machine learning predictions are not necessarily final business decisions.

Forecasting answers:

```text
"What is likely to happen?"
```

Optimization answers:

```text
"What should the business do?"
```

Visualization helps answer:

```text
"How can the result be interpreted quickly?"
```

The complete system connects all three questions.

---

# Project Status

The core end-to-end pipeline is operational.

Current completed components include:

```text
Data preprocessing             ✓
Weekly demand aggregation      ✓
Feature engineering            ✓
Forecast model comparison      ✓
Temporal holdout evaluation    ✓
Next-period forecasting        ✓
Forecast diagnostics           ✓
Inventory optimization         ✓
Visualization module           ✓
Automated figure generation    ✓
Business report generation     ✓
Unit testing                   ✓
Visualization testing          ✓
Integration testing            ✓
End-to-end execution           ✓
```

The current automated test suite contains:

```text
43 passing tests
```

The next major development stage will focus on building an interactive dashboard and extending the system with more advanced analytical and inventory-management capabilities.

---

# Author

**Mehrnoosh Salehi**

GitHub:

```text
https://github.com/mehrnoosh-salehi
```