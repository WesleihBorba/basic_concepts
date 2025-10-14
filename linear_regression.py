# Goal:
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.datasets import make_regression
import patsy
import logging
import sys

# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class LinearRegression:
    def __init__(self):
        self.X, self.y = make_regression(
            n_samples=15000,
            n_features=1,
            noise=10,
            random_state=42)

    def fit_model(self):
        logger.info('Starting to fit our regression')

        model = sm.OLS.from_formula('score ~ hours_studied', data=students)
        results = model.fit()
        logger.info(results.params)

    def predict_model(self):
        fitted_values = results.predict(students)
        logger.info('')

    def summary(self):
        pass

    def plot_linear_regression(self):
        pass

# Introduction to Linear Regression with sklearn and gradient



residuals = students.score - fitted_values # Criar regra (def) para ver se a assumption residuals passa como um modelo bom usando também try and excpt

# Fazer a mesma coisa normality and homoscedasticity - TRY AND EXCEPT - TENHO NO NOTE DO TRABALHO
plt.hist(residuals)
plt.show()

plt.scatter(fitted_values, residuals)
plt.show()




# Linear Regression with a Categorical Predictor

y, X = patsy.dmatrices('rent ~ borough', rentals)  #  SE EU CONSIGO USAR ISSO

