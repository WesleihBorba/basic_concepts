# Goal: Create a Perceptron, predict loan credit
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
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
        self.best_model, self.predict_values = None, None

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
        X = self.data[["income", "credit_score"]].values.copy()
        scale = StandardScaler()
        X = scale.fit_transform(X)
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

    def parameters_to_fitting(self):
        logger.info('Finding best hyperparameters for Perceptron')

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('percep', Perceptron(random_state=42))
        ])

        param_grid = {
            'percep__alpha': 10.0 ** -np.arange(1, 7),
            'percep__penalty': ['l2', 'l1', 'elasticnet'],
            'percep__eta0': [0.1, 0.5, 1.0]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        logger.info(f"Best parameters found: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.2f}")

        self.best_model = grid_search.best_estimator_
        test_score = self.best_model.score(self.X_test, self.y_test)
        logger.info(f"Test set accuracy with best model: {test_score:.2f}")

    def predict_model(self):
        logger.info('Predict Test Data')
        self.predict_values = self.best_model.predict(self.X_test)

    def evaluating_model(self):
        logger.info("Looking if our model is good to use")
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(self.y_test, self.predict_values),
            display_labels=['Denied', 'Approved']
        )
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix - Loan Credit")
        plt.show()

        logger.debug(f'Precision Score: {precision_score(self.y_test, self.predict_values)}')
        logger.debug(f'Recall score: {recall_score(self.y_test, self.predict_values)}')
        logger.debug(f'F1 Score: {f1_score(self.y_test, self.predict_values)}')


class_perceptron = PerceptronClassifier()
class_perceptron.plot_decision_boundary()
class_perceptron.train_test()
class_perceptron.parameters_to_fitting()
class_perceptron.predict_model()
class_perceptron.evaluating_model()