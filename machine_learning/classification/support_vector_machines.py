# Goal: Classify data using SVM polynomial kernel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import pandas as pd
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


class SupportVector:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.model, self.predict_values = None, None
        self.scale = None
        self.poly_data = self._create_synthetic_data()

    @staticmethod
    def _create_synthetic_data():
        logger.info('Creating our data to use with Polynomial')
        X, y = make_circles(n_samples=500, noise=0.1, factor=0.3, random_state=42)
        df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
        df["fraud_credit_risk"] = y
        return df

    def train_test(self):
        logger.info("Divide train and test")
        X = self.poly_data.drop(columns="fraud_credit_risk")
        y = self.poly_data["fraud_credit_risk"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True  # Independence assumption
        )

    def best_parameters(self):
        logger.info('Finding best parameters of our model')
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="poly", degree=3))
        ])

        param_grid = {
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", "auto"],
            "svc__coef0": [0, 1, 5]
        }

        grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
        grid.fit(self.X_train, self.y_train)

        self.model = grid.best_estimator_
        logger.debug(f"Best parameters: {grid.best_params_}")

    def predict_model(self):
        logger.info('Predict Test Data')
        self.predict_values = self.model.predict(self.X_test)

    def evaluating_model(self):
        logger.info("Looking if our model is good to use")

        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(self.y_test, self.predict_values),
            display_labels=['Ham', 'Spam']
        )
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix - Spam Classification")
        plt.show()

        logger.debug(classification_report(self.y_test, self.predict_values))

    def plot_svm(self):
        logger.info('Plotting SVM')
        h = .02  # Resolution
        x_min, x_max = self.poly_data["feature_1"].min() - 0.5, self.poly_data["feature_1"].max() + 0.5
        y_min, y_max = self.poly_data["feature_2"].min() - 0.5, self.poly_data["feature_2"].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=self.X_train.columns)

        Z = self.model.decision_function(grid_df)  # Equation of SVM, resulting in the coordinates
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))

        plt.contourf(xx, yy, Z, levels=50, cmap=plt.cm.RdBu, alpha=0.3)

        contours = plt.contour(xx, yy, Z, colors=['k', 'k', 'k'],
                               linestyles=['--', '-', '--'], levels=[-1, 0, 1])
        plt.clabel(contours, inline=1, fontsize=10)

        plt.scatter(self.X_test.iloc[:, 0], self.X_test.iloc[:, 1], c=self.y_test,
                    edgecolors='k', cmap=plt.cm.RdBu)

        plt.title("SVM: hyperplane (-) e Margins Upper/Lower (--)")
        plt.show()


class_svm = SupportVector()
class_svm.train_test()
class_svm.best_parameters()
class_svm.predict_model()
class_svm.evaluating_model()
class_svm.plot_svm()