# Goal: We will use logging to monitor the application of our cluster
import logging
import sys
from sklearn import datasets, linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


# Logger setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(message)s]')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


# Class cluster
class DiabeticCluster:
    def __init__(self):
        self.diabetes_X, self.diabetes_y = datasets.load_diabetes(return_X_y=True)
        self.y_train = []
        self.y_test = []
        self.X_train = []
        self.X_test = []
        self.regression = []

    def create_data(self):
        # Split the data into training/testing sets
        self.X_train = self.diabetes_X[:-20]
        self.X_test = self.diabetes_X[-20:]

        # Split the targets into training/testing sets
        self.y_train = self.diabetes_y[:-20]
        self.y_test = self.diabetes_y[-20:]

    def train(self):
        # Create linear regression object
        self.regression = linear_model.LinearRegression()

        # Train the model using the training sets
        self.regression.fit(self.X_train, self.y_train)

    def predict(regr, X_test):
        # Make predictions using the testing set
        diabetes_y_pred = regr.predict(X_test)

        return diabetes_y_pred

    def eval(y, y_hat):
        return mean_squared_error(y, y_hat)

# https://medium.com/@dKimothi/finding-best-features-for-predicting-diabetes-8656cf0d1185

# TRAINING LOGS


# EVALUATION LOGS
# PREDICTIONS LOGS
# SYSTEM LOGS (CPU AND MEMORY USAGE)

print(logger.addHandler(stream_handler))

logger.setLevel(logging.DEBUG)

logger.info("Converting from {from_country} to USD: {converted_to_usd}".format(from_country='from_country',
                                                                     converted_to_usd='converted_to_usd'))
logger.debug("Current rates: {exchange_rates}".format(exchange_rates='exchange_rates'))
logger.error("The TO country supplied is not a valid country.")
logger.log(logging.CRITICAL, 'Test critical')
logger.log(logging.INFO, "Test")
logger.warning('Values')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("my_program.log")
logger.addHandler(file_handler)

sys.exit(0)