# We will provoke errors using Logistic regression
import unittest
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CreateErrorInDatabase:
    @staticmethod
    def transform_y_class(data_y):
        data_y[data_y == 0] = 1
        data_replaced_y = data_y.copy()
        return data_replaced_y

    @staticmethod
    def add_nan_in_x(data_x):
        data_x[1, 1:4] = [np.nan, np.nan, np.nan]
        data_replaced_x = data_x.copy()
        return data_replaced_x

    @staticmethod
    def reshape_data_x(data_x):
        data_reshape_x = data_x.reshape(-1)
        return data_reshape_x

    @staticmethod
    def resize_data_x(data_x):
        data_resize = np.resize(data_x, (565, 25))
        return data_resize


# Creating a class to build logistic regression
class LogisticRegressionAnalysis(CreateErrorInDatabase):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def analysis_data(self):
        try:
            # Creating errors
            self.y_data = self.transform_y_class(self.y_data)
            self.x_data = self.add_nan_in_x(self.x_data)
            self.x_data = self.reshape_data_x(self.x_data)
            self.x_data = self.resize_data_x(self.x_data)

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                self.x_data, self.y_data, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = LogisticRegression()
            model.fit(X_train_scaled, y_train)

            y_prediction = model.predict(X_test_scaled)

            accuracy = accuracy_score(y_test, y_prediction)
            print(f'accuracy score: {accuracy}')
            print(f'Classification report : \n{classification_report(y_test, y_prediction)}')

            # Warning if accuracy is lower
            if accuracy < 0.5:
                warnings.warn("The accuracy is not good, try to change something!", UserWarning)

            return accuracy

        except ValueError as e:
            print(f"The error: {e}")
            raise  # Send to unittest


class TestingError(unittest.TestCase):
    def setUp(self):
        cancer_data = load_breast_cancer()
        self.X, self.y = cancer_data.data, cancer_data.target

    def test_logistic_regression_with_errors(self):
        analysis = LogisticRegressionAnalysis(self.X, self.y)

        with self.assertRaises(ValueError):  # intentional error
            analysis.analysis_data()

    def test_accuracy_warning(self):
        # Trying to run and find a bad accuracy
        X_small, y_small = self.X[:20], self.y[:20]  # Few data -> Bad performance
        analysis = LogisticRegressionAnalysis(X_small, y_small)

        try:
            acc = analysis.analysis_data()
            self.assertIsInstance(acc, float)
        except ValueError:
            pass


# Run Tests
if __name__ == "__main__":
    unittest.main()