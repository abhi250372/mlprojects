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
    summary_stats = input_transformer.summary_statistics(outlier_removed_data)
    print(f"Summary Statistics:\n{summary_stats}")
    #histogram, boxplot, scatterplot = input_transformer.graphical_analysis(outlier_removed_data)
    # correlation_matrix = input_transformer.correlational_analysis(outlier_removed_data)

    # print(f"Correlation Matrix:\n{correlation_matrix}")
    vif_data = input_transformer.multicollinearity_check(outlier_removed_data)

    print(f"VIF Data:\n{vif_data}")
    transformed_data = input_transformer.transform_features(outlier_removed_data)

    # trend_analysis_plot = input_transformer.trend_analysis(transformed_data)

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

    feature_importance_df = linear_model.feature_importance(X)
    print(f"Feature Importance:\n{feature_importance_df}")

    if vif_data['VIF'].max() > 10:
        print("Warning: High multicollinearity detected. Consider removing features with high VIF values.")
        lasso_model = linear_model.lasso_regression(X, y, alpha=0.1)
        lasso_predictions = linear_model.lasso_predict(X)
        lasso_rmse = linear_model.lasso_root_mean_squared_error(y, lasso_predictions)
        lasso_r2 = linear_model.lasso_accuracy(X, y)
        print(f"Lasso Regression RMSE: {lasso_rmse:.2f}")
        print(f"Lasso Regression R-squared (R2): {lasso_r2:.4f}")
        lasso_feature_importance_df = linear_model.lasso_feature_importance(X)
        print(f"Lasso Feature Importance:\n{lasso_feature_importance_df}")





if __name__ == "__main__":
    main()