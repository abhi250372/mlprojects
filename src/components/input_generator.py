import numpy as np
import pandas as pd

np.random.seed(42)

n_items = 5000
n_weeks = 12
total_rows = n_items * n_weeks

# 1. Generate base item characteristics (repeated across weeks)
item_ids = np.repeat(np.arange(1, n_items + 1), n_weeks)
weeks = np.tile(np.arange(1, n_weeks + 1), n_items)
categories = np.repeat(np.random.choice(["Electronics", "Apparel", "Home", "Grocery"], size=n_items), n_weeks)
is_prime = np.repeat(np.random.choice([0, 1], size=n_items, p=[0.3, 0.7]), n_weeks)

# 2. Generate time-varying variables (change week-over-week)
# Base prices vary per item, with small weekly fluctuations
base_prices = np.repeat(np.random.uniform(10.0, 120.0, n_items), n_weeks)
weekly_price_fluctuation = np.random.uniform(0.95, 1.05, total_rows)
current_price = base_prices * weekly_price_fluctuation

# Generate sequential weekly web traffic exhibiting seasonal trends
page_views = np.random.negative_binomial(30, 0.01, total_rows)
# Inject a demand spike in Week 6 to simulate a promotional event (e.g., Prime Day)
page_views = np.where(weeks == 6, page_views * 1.8, page_views).astype(int)

in_stock_pct = np.random.beta(9, 1, total_rows)

df = pd.DataFrame({
    "item_id": item_ids,
    "week": weeks,
    "category": categories,
    "is_prime_eligible": is_prime,
    "current_price": current_price,
    "page_views_30d": page_views,
    "in_stock_percentage": in_stock_pct
})

# 3. Simulate sequential target variable: weekly_units_sold
# Demand relies heavily on historical trends, current traffic, and seasonal shocks
base_demand = 200
price_impact = -1.8 * df["current_price"]
traffic_impact = 0.8 * df["page_views_30d"]
promo_shock = np.where(df["week"] == 6, 150, 0) # Systematic structural spike

# 4. Add a date column for better interpretability
df["date"] = pd.to_datetime('2023-01-01') + pd.to_timedelta((df['week'] - 1)) * 7
df['year'] = df['date'].dt.year 
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter

# Base signal calculation
signal = base_demand + price_impact + traffic_impact + promo_shock
noise = np.random.normal(0, 30, total_rows)
df["weekly_units_sold"] = (signal + noise).clip(lower=0).astype(int)

print(f"Panel dataset successfully generated with shape: {df.shape}")
print(df.head(15)) # Look at item_id 1 across consecutive weeks

df.to_csv("src/input_data/synthetic_ecommerce_data.csv", index=False)