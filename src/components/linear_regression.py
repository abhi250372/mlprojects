from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

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