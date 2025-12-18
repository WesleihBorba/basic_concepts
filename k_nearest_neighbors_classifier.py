# Goal: Find best K of K-Nearst and predict data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
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
        pass

        # Fazer um dataset com 3 grupos (pensar que grupos usar - sobre o que vai ser)

    def train_test(self):
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

    def validation_model(self):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)

        cm = confusion_matrix(y_test, y_pred)




# Descobrir como encontrar o K corretamente e colocar em um DEF e mostrar gráficamente com dados reais para ver como fica a mudança em forma de circulo, como feito aqui: K-Nearest Neighbors Classifier

# An important thing about this algorithm is that we have to standardize each feature to have a mean zero and a variance one
# Olhar as assumptions tipo a anterior
# Usar normal distribution assumption
# Correlation (Maybe)









