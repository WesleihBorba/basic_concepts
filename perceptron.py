# Goal:
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import logging
import sys

# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class PerceptronClassifier:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        self.X, self.y = make_classification(
            n_samples=15000,
            n_features=2,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=2,
            random_state=42
        )

        self.data = pd.DataFrame(self.X, columns=["income", "credit_score"])
        self.data["approved"] = self.y

    def plot_decision_boundary(self):
        X = self.data[["income", "credit_score"]].values
        y = self.data["approved"].values

        model = Perceptron()
        model.fit(X, y)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.2, cmap="bwr")
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k")
        plt.xlabel("income")
        plt.ylabel("credit_score")
        plt.title("Perceptron Decision Boundary")
        plt.show()

    def train_test(self):
        logger.info("Divide train and test")
        X = self.data.drop(columns={'approved'})
        y = self.data['approved']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=0
        )
        logger.debug(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    def parameters(self):
        logger.info('Finding best hyperparameters for Perceptron')

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('percep', Perceptron(random_state=42))
        ])

        param_grid = {
            'percep__alpha': 10.0 ** -np.arange(1, 7),
            'percep__penalty': ['l2', 'l1', 'elasticnet'],
            'percep__eta0': [0.1, 0.5, 1.0]  # Common values, you can expand this
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        logger.info(f"Best parameters found: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.2f}")

        best_model = grid_search.best_estimator_
        test_score = best_model.score(self.X_test, self.y_test)
        logger.info(f"Test set accuracy with best model: {test_score:.2f}")




    def fitting(self):
        logger.info('Starting to fit our regression')
        try:
            model = sm.OLS.from_formula('income ~ education + working_time', data=self.train)
            self.fit_regression = model.fit()
            logger.info("Success!")
        except Exception as e:
            logger.error(f"Fail to training: {e}")

        logger.info(f"Coefficients: {self.fit_regression.params}")

    def predict_model(self):
        logger.info('Predict Test Data')
        self.predict_values = self.fit_regression.predict(self.test[['education', 'working_time']])
        self.resid = self.test['income'] - self.predict_values

    def evaluating_model(self):
        logger.info("Looking if our model is good to use")
        mse = mean_squared_error(self.test['income'], self.predict_values)
        r2 = r2_score(self.test['income'], self.predict_values)

        logger.info(f'Mean Squared Error: {mse}')
        logger.info(f'R-squared: {r2}')

    def plot_multiple_regression(self):
        sample = self.test.sample(50)  # 50 random points
        y_true = sample['income']
        y_predict = self.fit_regression.predict(sample[['education', 'working_time']])

        plt.scatter(y_true, y_predict, alpha=0.7)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect prediction')

        for i in range(len(sample)):
            plt.plot([y_true.iloc[i], y_true.iloc[i]], [y_true.iloc[i], y_predict.iloc[i]], 'gray', alpha=0.3)

        plt.xlabel('Real value (income)')
        plt.ylabel('Predict value (income)')
        plt.title('Comparison between actual and predicted values (with errors)')
        plt.legend()
        plt.show()


class_regression = PerceptronClassifier()
class_regression.plot_decision_boundary()
#class_regression.train_test()
#class_regression.fit_model()
#class_regression.predict_model()
#class_regression.evaluating_model()
#class_regression.plot_multiple_regression()