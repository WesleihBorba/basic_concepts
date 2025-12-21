# Goal: Find best K of K-Nearst and predict data
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
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
        np.random.seed(42)
        n = 10_000
        amount = np.random.lognormal(mean=3, sigma=1, size=n)
        avg_amount = np.random.lognormal(mean=3, sigma=0.5, size=n)
        amount_ratio = amount / (avg_amount + 1)

        hour = np.random.randint(0, 24, n)
        number_transaction_24h = np.random.poisson(lam=2, size=n)

        new_merchant = np.random.binomial(1, 0.1, n)
        new_device = np.random.binomial(1, 0.05, n)
        new_location = np.random.binomial(1, 0.03, n)
        merchant_risk = np.random.uniform(0, 1, n)  # merchant score

        fraud = (
                (amount_ratio > 4) &
                (number_transaction_24h > 5) &
                ((new_device == 1) | (new_location == 1)) &
                (merchant_risk > 0.7)).astype(int)

        self.data = pd.DataFrame({
            "amount": amount,
            "avg_amount_30d": avg_amount,
            "amount_ratio": amount_ratio,
            "hour": hour,
            "number_transaction_24h": number_transaction_24h,
            "new_merchant": new_merchant,
            "new_device": new_device,
            "new_location": new_location,
            "merchant_risk": merchant_risk,
            "fraud": fraud
        })

        print(self.data)


    def train_test(self):
        logger.info()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    def normalization(self):
        pass
        # PRECISA NORMALIZAR AS INFORMAÇÕES

    def cross_validation(self):
        # Usar cross validation
        k_values = [i for i in range(1, 21)]
        scores = []

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn, X, y, cv=5)
            scores.append(np.mean(score))

        sns.lineplot(x=k_values, y=scores, marker='o')
        plt.xlabel("K Values")
        plt.ylabel("Accuracy Score")

        best_index = np.argmax(scores)
        best_k = k_values[best_index]

        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # Descobrir como encontrar o K corretamente e colocar em um DEF e mostrar gráficamente com dados reais para ver como fica a mudança em forma de circulo, como feito aqui: K-Nearest Neighbors Classifier

    def validation_model(self):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)

        cm = confusion_matrix(y_test, y_pred)


class_neighbor = NearstNeighbors()
