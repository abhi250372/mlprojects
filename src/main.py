import pandas as pd
from pathlib import Path
from components.input_processor import InputProcessor
from components.input_transformer import InputTransformer
from components.linear_regression import LinearRegressionModel


def main():
    input_processor = InputProcessor()
    input_transformer = InputTransformer()
    data = input_processor.load_data('src/input_data/synthetic_ecommerce_data.csv')
    cleaned_data = input_transformer.clean_data(data)
    outlier_removed_data = input_transformer.remove_outliers(cleaned_data)
    transformed_data = input_transformer.transform_features(outlier_removed_data)
    linear_model = LinearRegressionModel()
    X = transformed_data.drop(columns=['weekly_units_sold'])
    y = transformed_data['weekly_units_sold']
    linear_model.fit(X, y)
    predictions = linear_model.predict(X)
    rmse = linear_model.root_mean_squared_error(y, predictions)
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    r2 = linear_model.accuracy(X, y)
    print(f"R-squared (R2): {r2:.4f}")
    coefficients = linear_model.get_coeffictients()
    intercept = linear_model.get_intercept()
    print(f"Model Coefficients: {coefficients}")
    print(f"Model Intercept: {intercept}")
    



if __name__ == "__main__":
    main()