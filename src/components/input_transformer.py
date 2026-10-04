import pandas as pd
import numpy as np


class InputTransformer:
    def __init__(self):
        self.data = None

    def clean_data(self, data):
        # Handle missing values: Fill missing in stock percentage with median by category and week
        self.data = data.copy()
        self.data['median_in_stock_percentage'] = self.data.groupby(['category', 'week'])['in_stock_percentage'].transform('median')
        self.data['in_stock_percentage'] = self.data['in_stock_percentage'].fillna(self.data['median_in_stock_percentage'])
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
    
    def transform_features(self, data):
        # Feature engineering: Create a new feature for price per page view
        data['price_per_page_view'] = data['current_price'] / (data['page_views_30d'] + 1)  # Avoid division by zero
        data = pd.get_dummies(data, columns=['category'], prefix='category')
        return data