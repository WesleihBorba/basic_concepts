# Goal: Predict data with k-nearest neighbors regression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
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


class NearstNeighborsRegression:

    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.best_k, self.cv_scores = None, None
        self.model, self.y_predict, self.y_prob = None, None, None

        np.random.seed(42)
        n = 10_000

        amount = np.random.lognormal(mean=4, sigma=1, size=n)
        avg_amount = np.random.lognormal(mean=2, sigma=1, size=n)
        amount_ratio = amount / (avg_amount + 1)

        number_transaction_24h = np.random.poisson(lam=2, size=n)
        new_device = np.random.binomial(1, 0.4, n)
        new_location = np.random.binomial(1, 0.4, n)
        merchant_risk = np.random.uniform(0, 1, n)

        # Normalize some parts to build continuous target
        amount_ratio_norm = (amount_ratio - amount_ratio.min()) / (amount_ratio.max() - amount_ratio.min())
        tx_norm = number_transaction_24h / number_transaction_24h.max()

        fraud_risk_score = (
            0.4 * amount_ratio_norm +
            0.3 * tx_norm +
            0.2 * merchant_risk +
            0.1 * ((new_device == 1) | (new_location == 1))
        )

        fraud_risk_score += np.random.normal(0, 0.05, n)
        fraud_risk_score = np.clip(fraud_risk_score, 0, 1)

        self.data = pd.DataFrame({
            "amount": amount,
            "avg_amount_30d": avg_amount,
            "amount_ratio": amount_ratio,
            "number_transaction_24h": number_transaction_24h,
            "new_device": new_device,
            "new_location": new_location,
            "merchant_risk": merchant_risk,
            "fraud_risk_score": fraud_risk_score
        })

    def train_test(self):
        logger.info("Divide train and test")
        X = self.data.drop(columns="fraud_risk_score")
        y = self.data["fraud_risk_score"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def cross_validation(self):
        logger.info("Cross validation for KNN Regressor")

        k_values = range(1, 21)
        scores = []
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        for k in k_values:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(
                    n_neighbors=k,
                    weights="distance"
                ))
            ])

            cv_score = cross_val_score(
                pipe,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring="neg_mean_squared_error"
            )

            scores.append(-cv_score.mean())

        self.best_k = k_values[np.argmin(scores)]
        self.cv_scores = scores

        logger.info(f"Best K found: {self.best_k}")

    def regressor_data(self):
        logger.info("Training final KNN Regressor")

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(
                n_neighbors=self.best_k,
                weights="distance"
            ))
        ])

        self.model.fit(self.X_train, self.y_train)
        self.y_predict = self.model.predict(self.X_test)

    def plot_cv(self):
        plt.figure()
        plt.plot(range(1, 21), self.cv_scores, marker='o')
        plt.xlabel("Number of Neighbors (K)")
        plt.ylabel("Mean Squared Error")
        plt.title("KNN Regression - Cross Validation")
        plt.show()

    def validation_model(self):
        logger.info(f"MSE: {mean_squared_error(self.y_test, self.y_predict):.4f}")
        logger.info(f"R2 Score: {r2_score(self.y_test, self.y_predict):.4f}")

    def single_predict(self):
        logger.info("Predicting fraud risk score for a new transaction")

        new_transaction = np.array([[
            1200,
            300,
            4.0,
            5,
            1,
            0,
            0.8
        ]])

        transaction_df = pd.DataFrame(
            new_transaction,
            columns=self.X_train.columns
        )

        risk_score = self.model.predict(transaction_df)[0]
        logger.info(f"Predicted fraud risk score: {risk_score:.4f}")


class_neighbor = NearstNeighborsRegression()
class_neighbor.train_test()
class_neighbor.cross_validation()
class_neighbor.regressor_data()
class_neighbor.plot_cv()
class_neighbor.validation_model()
class_neighbor.single_predict()