# Goal: Find best K of K-Nearst and predict data
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
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


class NearstNeighbors:

    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.best_k, self.cv_scores = None, None
        self.y_predict, self.y_prob = None, None

        np.random.seed(42)
        n = 10_000
        amount = np.random.lognormal(mean=4, sigma=1, size=n)
        avg_amount = np.random.lognormal(mean=2, sigma=1, size=n)
        amount_ratio = amount / (avg_amount + 1)

        number_transaction_24h = np.random.poisson(lam=2, size=n)

        new_device = np.random.binomial(1, 0.4, n)
        new_location = np.random.binomial(1, 0.4, n)
        merchant_risk = np.random.uniform(0, 1, n)  # merchant score

        fraud = (
                (amount_ratio > 2.5) &
                (number_transaction_24h > 3) &
                ((new_device == 1) | (new_location == 1)) &
                (merchant_risk > 0.6)
                )
        fraud = [int(elem) for elem in fraud]

        self.data = pd.DataFrame({
            "amount": amount,
            "avg_amount_30d": avg_amount,
            "amount_ratio": amount_ratio,
            "number_transaction_24h": number_transaction_24h,
            "new_device": new_device,
            "new_location": new_location,
            "merchant_risk": merchant_risk,
            "fraud": fraud
        })

    def train_test(self):
        logger.info("Divide train and test")
        X = self.data.drop(columns={'fraud'})
        y = self.data['fraud']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    def cross_validation(self):
        logger.info("Cross validation for KNN")

        k_values = range(1, 21)
        scores = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for k in k_values:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(
                    n_neighbors=k,
                    weights="distance"
                ))
            ])

            cv_score = cross_val_score(
                pipe,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring="roc_auc"
            )

            scores.append(cv_score.mean())

        best_k = k_values[np.argmax(scores)]
        self.best_k = best_k
        self.cv_scores = scores
        logger.info(f"Best K found: {best_k}")

    def classifier_data(self):
        logger.info("Training final KNN model")

        data_predict = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=self.best_k,
                weights="distance"
            ))
        ])

        data_predict.fit(self.X_train, self.y_train)

        self.y_predict = data_predict.predict(self.X_test)
        self.y_prob = data_predict.predict_proba(self.X_test)[:, 1]
        print(self.y_predict)
        print(self.y_prob)

    def validation_model(self):
        print(classification_report(self.y_test, y_pred))
        print("ROC AUC:", roc_auc_score(self.y_test, y_proba))

        cm = confusion_matrix(y_test, y_pred)


class_neighbor = NearstNeighbors()
class_neighbor.train_test()
class_neighbor.cross_validation()
class_neighbor.classifier_data()
