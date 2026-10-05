from ast import For
from pyexpat import model
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import numpy as np

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.lasso_model = None  # Initialize lasso_model attribute

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_coeffictients(self):
        return self.model.coef_
    
    def get_intercept(self):  
        return self.model.intercept_
    
    def get_model(self):
        return self.model
    
    def accuracy(self, X, y):
        return self.model.score(X, y)
    
    def root_mean_squared_error(self, y, predictions):
        rmse = np.sqrt(mean_squared_error(y, predictions))
        return rmse
    
    def shape_of_coefficients(self):
        return self.model.coef_.shape
    
    def feature_importance(self, X):
        # Get coefficients
        importance = pd.DataFrame({
        "feature": X.columns,
        "importance": self.model.coef_
        }).sort_values("importance", ascending=False)
        return importance
    
    def lasso_regression(self, X, y, alpha=1.0):
        self.lasso_model = Lasso(alpha=alpha)
        self.lasso_model.fit(X, y)
        return self.lasso_model
    
    def lasso_predict(self, X):
        return self.lasso_model.predict(X)
    
    def lasso_coefficients(self):
        return self.lasso_model.coef_
    
    def lasso_intercept(self):
        return self.lasso_model.intercept_

    def lasso_accuracy(self, X, y):
        return self.lasso_model.score(X, y)
    
    def lasso_root_mean_squared_error(self, y, predictions):
        rmse = np.sqrt(mean_squared_error(y, predictions))
        return rmse
    
    def lasso_feature_importance(self, X):
        # Get coefficients
        importance = pd.DataFrame({
        "feature": X.columns,
        "importance": self.lasso_model.coef_
        }).sort_values("importance", ascending=False)
        return importance

    
