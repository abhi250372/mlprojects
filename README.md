# Ecommerce Demand Forecasting Project

This project demonstrates an end-to-end machine learning workflow for forecasting weekly sales in a synthetic ecommerce dataset. The goal is to predict `weekly_units_sold` using product attributes, product category, traffic, pricing, and inventory signals.

The project is designed as a learning exercise and interview-prep project for understanding:
- data generation
- EDA and feature analysis
- missing value treatment
- outlier handling
- categorical encoding
- model training and evaluation
- regularization and model comparison
- time-based validation for forecasting

---

## Project Overview

The dataset contains synthetic weekly ecommerce records for multiple products across several weeks. Each row represents a product-week observation.

Core fields include:
- `item_id`
- `week`
- `date`
- `category`
- `is_prime_eligible`
- `current_price`
- `page_views_30d`
- `in_stock_percentage`
- `weekly_units_sold`

The target variable is:
- `weekly_units_sold`

The main objective is to estimate how product features and market conditions influence future demand.

---

## Repository Structure

- `src/components/input_generator.py` — synthetic data generation
- `src/components/input_processor.py` — data loading logic
- `src/components/input_transformer.py` — cleaning, EDA utilities, feature engineering
- `src/components/linear_regression.py` — baseline linear model wrapper
- `src/main.py` — orchestration and model execution
- `src/input_data/synthetic_ecommerce_data.csv` — generated dataset

---

## Data Generation

The synthetic data is created in `input_generator.py` and follows a realistic ecommerce pattern:
- repeated product records across multiple weeks
- category assignments
- prime eligibility
- product-level price variation
- page view traffic
- inventory percentage
- a demand spike during a simulated promotional week
- synthetic noise added to weekly sales

The generator produces a panel-style dataset where each product has a sequence of weekly observations, which makes it suitable for exploratory analysis and forecasting-style experiments.

---

## Column Design and Business Meaning

### Identifier columns
- `item_id`: product identifier
- `week`: integer week index
- `date`: derived weekly timestamp

### Categorical features
- `category`: product category such as Electronics, Apparel, Home, or Grocery
- `is_prime_eligible`: flag indicating Prime eligibility

### Numeric features
- `current_price`: current selling price
- `page_views_30d`: recent page views over 30 days
- `in_stock_percentage`: inventory availability ratio
- `weekly_units_sold`: target sales volume

### Derived features
- `price_per_page_view`: engineered metric created to capture the relationship between pricing and traffic
- `year`, `month`, `quarter`: time-derived indicators from the date column

---

## Data Cleaning and Preprocessing

The project applies several cleaning steps before modeling.

### 1. Missing value treatment
We handle missing values systematically:
- `in_stock_percentage` is filled using a median value grouped by category and week
- `category` missing values are filled with a placeholder such as `Unknown`
- `is_prime_eligible` missing values are filled with `0`
- `page_views_30d` missing values are filled with `0`
- `current_price` is interpolated within each product over time and then forward/backward filled as needed

This is important because forecasting models cannot easily work with unresolved missing values.

### 2. Outlier handling
The project uses an IQR-based method at the product level to identify outlier sales values.

The workflow is:
- calculate Q1 and Q3 for weekly sales within each item
- compute IQR = Q3 - Q1
- compute lower and upper bounds
- remove values outside the bounds

This helps protect the model from extreme values that may reflect noise or unusual anomalies rather than regular business behavior.

---

## Exploratory Data Analysis (EDA)

The project includes several EDA steps to understand the structure of the data before modeling.

### Summary statistics
We compute summary statistics for:
- `weekly_units_sold`
- `in_stock_percentage`
- `current_price`
- `category`
- `item_id`

This helps answer questions like:
- What is the average weekly demand?
- Are sales heavily skewed?
- Are there unusual price or traffic ranges?

### Graphical analysis
The EDA includes:
- histogram of weekly units sold
- box plot of weekly units sold by category
- scatter plot of in-stock percentage vs weekly units sold

This helps identify:
- distribution shape
- category-level differences
- relationship between inventory and demand

### Correlation analysis
We analyze the relationship between:
- `weekly_units_sold`
- `current_price`
- `page_views_30d`
- `in_stock_percentage`

This helps detect whether key features are strongly related to demand and whether correlations may be masking other effects.

### Multicollinearity check
The project evaluates VIF (Variance Inflation Factor) for key numeric features.

This is particularly useful because features like:
- `page_views_30d`
- `in_stock_percentage`
- `current_price`

may be correlated with one another, which can destabilize linear coefficients.

---

## Feature Engineering

The project creates and transforms features to improve model performance.

### Encoded categorical variables
The dataset uses one-hot encoding for the `category` feature so that each category becomes a separate binary column.

This is appropriate because `category` is nominal and should not be mapped to an ordinal integer scale.

### Derived feature
A new feature is created:
- `price_per_page_view = current_price / (page_views_30d + 1)`

This attempts to capture the relationship between price and demand attractiveness relative to traffic volume.

### Time-based features
The date column is used to generate:
- `year`
- `month`
- `quarter`

These features make it easier to model seasonal trends and time effects.

---

## Modeling Strategy

The project starts with a baseline linear regression model to establish a reference performance level.

The core logic is:
- load data
- clean data
- remove outliers
- inspect EDA metrics
- transform features
- fit linear model
- evaluate on appropriate metrics

The codebase is designed to support extension to more models, including:
- Ridge regression
- Lasso regression
- logistic regression
- KNN
- tree-based models
- gradient boosting
- Prophet-style forecasting experiments

---

## Regularization and Model Selection

One important step in the workflow is understanding when ordinary linear regression is not sufficient.

### Why regularization matters
Regularization becomes useful when:
- features are highly correlated
- many coefficients are unstable
- the feature set is large or sparse
- we want to reduce overfitting

### Lasso regularization
Lasso adds an L1 penalty and can reduce weak or redundant feature coefficients to zero. This makes it useful for:
- feature selection
- reducing overfitting
- simplified models

### Ridge regularization
Ridge adds an L2 penalty and shrinks coefficients more smoothly without necessarily zeroing them out. This is often useful when there is multicollinearity but we still want to retain all features.

### Model decision logic
A good production-style approach is:
- start with linear regression
- assess multicollinearity and validation performance
- if the feature set is noisy or correlated, try Ridge or Lasso
- compare models on a validation set rather than raw training fit alone

---

## Time-Series Validation

For forecasting problems, the most important evaluation principle is time-based validation.

Instead of random train/test splitting, the better approach is:
- train on earlier weeks
- validate or test on later weeks
- preserve chronological order
- avoid leakage from future periods into the model

This is essential for any forecasting pipeline because time order carries information that random splitting destroys.

---

## Model Evaluation

The main metrics used in this project are:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R-squared (R2)

These metrics are useful for understanding:
- how far predictions are from reality
- whether the model explains enough variance
- whether the model is generalizing or just memorizing the training distribution

---

## Key Learnings from the Project

The project highlights several important ML and forecasting lessons:
- EDA is essential before choosing a model
- data cleaning decisions matter because they influence downstream signal
- category encoding should preserve the meaning of categorical variables
- high VIF values signal multicollinearity and may require regularization
- feature engineering can improve predictive performance substantially
- time-based validation is more important than raw training accuracy in forecasting
- a model that performs well on training data may still fail on future periods

---

## Current Project Status

The project currently contains:
- synthetic data generation
- preprocessing and cleaning logic
- EDA utilities
- feature engineering
- baseline linear regression training and evaluation
- a regularization-ready workflow for Lasso and Ridge experiments

The codebase is ready to be extended into a broader benchmarking pipeline with multiple model families.

---

## Suggested Next Steps

The next logical steps are:
1. Build proper time-based train/validation/test splits
2. Add lag and rolling features for forecasting
3. Compare linear regression, Ridge, and Lasso side by side
4. Add tree-based models such as Random Forest and Gradient Boosting
5. Compare model performance on future weeks
6. Document business interpretation of the final model

---

## How to Run the Project

From the project root:

```bash
python src/components/input_generator.py
python src/main.py
```

This will regenerate the synthetic dataset and run the model pipeline.

---

## Final Note

This project is a strong educational example of how data science work should be structured:
- generate data
- inspect it deeply
- clean it intentionally
- engineer relevant features
- train a baseline model
- validate carefully
- compare model families logically

This is exactly the way to prepare for real-world forecasting and interview-style machine learning questions.
