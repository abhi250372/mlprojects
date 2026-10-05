import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

class InputTransformer:
    def __init__(self):
        self.data = None

    def clean_data(self, data):
        # Handle missing values: Fill missing in stock percentage with median by category and week
        self.data = data.copy()
        self.data['median_in_stock_percentage'] = self.data.groupby(['category', 'week'])['in_stock_percentage'].transform('median')
        self.data['in_stock_percentage'] = self.data['in_stock_percentage'].fillna(self.data['median_in_stock_percentage'])
        self.data['category'] = self.data['category'].fillna('Unknown')
        self.data['is_prime_eligible'] = self.data['is_prime_eligible'].fillna(0)
        self.data['page_views_30d'] = self.data['page_views_30d'].fillna(0)
        self.data['current_price'] = self.data.groupby('item_id')['current_price'].transform(lambda x: x.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill'))
        return self.data.drop(columns=['median_in_stock_percentage'])
    
    def remove_outliers(self, data):
        # Remove data using IQR method for weekly units sold
        data['Q1'] = data.groupby(['item_id'])['weekly_units_sold'].transform(lambda x: x.quantile(0.25))
        data['Q3'] = data.groupby(['item_id'])['weekly_units_sold'].transform(lambda x: x.quantile(0.75))
        data['IQR'] = data['Q3'] - data['Q1']
        data['lower_bound'] = data['Q1'] - 1.5 * data['IQR']
        data['upper_bound'] = data['Q3'] + 1.5 * data['IQR']
        cleaned_data = data[(data['weekly_units_sold'] >= data['lower_bound']) & (data['weekly_units_sold'] <= data['upper_bound'])]
        return cleaned_data.drop(columns=['Q1', 'Q3', 'IQR', 'lower_bound', 'upper_bound'])

    def summary_statistics(self, data):
        summary = data[['item_id','weekly_units_sold','in_stock_percentage', 'current_price','category']].describe(include='all')
        return summary
    
    def graphical_analysis(self, data):

        plt.figure(figsize=(12, 6))
        sns.histplot(data['weekly_units_sold'], bins=30, kde=True)
        plt.title('Distribution of Weekly Units Sold')
        plt.xlabel('Weekly Units Sold')
        plt.ylabel('Frequency')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='category', y='weekly_units_sold', data=data)
        plt.title('Weekly Units Sold by Category')
        plt.xlabel('Category')
        plt.ylabel('Weekly Units Sold')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='in_stock_percentage', y='weekly_units_sold', data=data)
        plt.title('Weekly Units Sold vs In-Stock Percentage')
        plt.xlabel('In-Stock Percentage')
        plt.ylabel('Weekly Units Sold')
        plt.show()

        return plt, plt, plt  # Return the last plot objects for potential further use

    def correlational_analysis(self, data):
        correlation_matrix = data[['weekly_units_sold', 'current_price', 'page_views_30d', 'in_stock_percentage']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()
        return correlation_matrix
    
    def multicollinearity_check(self, data):

        X = data[['current_price', 'page_views_30d', 'in_stock_percentage']]
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data
    
    def trend_analysis(self, data):
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='week', y='weekly_units_sold', data=data.groupby('week')['weekly_units_sold'].mean().reset_index())
        plt.title('Average Weekly Units Sold Over Time')
        plt.xlabel('Week')
        plt.ylabel('Average Weekly Units Sold')
        plt.show()
    
    def transform_features(self, data):
        # Feature engineering: Create a new feature for price per page view
        data['price_per_page_view'] = data['current_price'] / (data['page_views_30d'] + 1)  # Avoid division by zero
        data = pd.get_dummies(data, columns=['category'], prefix='category')
        return data