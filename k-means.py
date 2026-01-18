# Goal: Segmentation Supermarket mall customers. Understanding more our customers and discover "target customers'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


class KMeansClassifier:
    def __init__(self):
        self.train, self.test = None, None
        self.normalization_train = None
        self.model, self.predict_values = None, None

        self.data = pd.read_csv('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\Mall_Customers.csv')

    def transform_data(self):
        self.data.drop(columns={'CustomerID'}, inplace=True)

        logger.info('Transforming Gender in 1 and 0')
        map_gender = {'Male': 1, 'Female': 0}
        self.data['Gender'] = self.data['Gender'].map(map_gender)
        logger.debug(f'Changes of values, Mall Customers: gender{map_gender}')

        self.data.rename(columns={'Annual Income (k$)': 'annual_income', 'Spending Score (1-100)': 'spending_score'},
                         inplace=True)

    def train_test(self):
        logger.info("Divide train and test")
        self.train, self.test = train_test_split(self.data, test_size=0.2, random_state=0)
        logger.debug(f"Train: {self.train.shape}, Test: {self.test.shape}")

    def normalization(self):
        logger.info('Normalization values')
        zscore = StandardScaler()
        self.normalization_train = pd.DataFrame(zscore.fit_transform(self.train), columns=self.train.columns)

    def elbow_silhouette_method(self):
        logger.info('Elbow: how close the points are within the same cluster')
        logger.info('Silhouette: how far apart the clusters are from each other')
        inertia = []
        sil_scores = []
        k_range = range(2, 11)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(self.train)
            inertia.append(kmeans.inertia_)
            sil_scores.append(silhouette_score(self.train, kmeans.labels_))

        fig, ax1 = plt.subplots(figsize=(6, 4))

        # Primary y-axis: Inertia (Elbow Method)
        ax1.plot(k_range, inertia, 'bo-', label='Inertia (Elbow)')
        ax1.set_xlabel('Number of Clusters k')
        ax1.set_ylabel('Inertia', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Secondary y-axis: Silhouette Score
        ax2 = ax1.twinx()
        ax2.plot(k_range, sil_scores, 'go-', label='Silhouette Score')
        ax2.set_ylabel('Silhouette Score', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        # Title and layout
        plt.title('Elbow Method vs. Silhouette Score')
        fig.tight_layout()
        plt.show()

    def fitting_data(self):
        logger.info('Fitting with KMeans ++')
        self.model = KMeans(
            n_clusters=5,
            init='k-means++',
            random_state=42
        ).fit(self.train)


    def plot_groups(self):
        plt.figure(figsize=(12, 7))

        plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Temkinli')
        plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cimri')
        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Standart')
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Savurgan')
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Hedef Kitle')

        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker='X', c='orange',
                    label='Merkezler')

        plt.title('Müşteri Segmentleri')
        plt.xlabel('Yıllık Gelir (k$)')
        plt.ylabel('Harcama Skoru (1-100)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()


    def predict_model(self):
        pass

    def evaluating_model(self):
        pass


classification = KMeansClassifier()
classification.transform_data()
classification.train_test()
classification.normalization()
classification.elbow_silhouette_method()
exit()

centroids = np.array(list(zip(centroids_x, centroids_y)))

model = KMeans(init="k-means++", n_clusters=2)
# Initial centroids
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=100)

results = model.fit_predict(values)

plt.scatter(x, y, c=results, alpha=1)

# Cluster centers
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='v', s=100)

ax = plt.subplot()
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_yticks([0, 1, 2, 3, 4])

plt.title('K-Means++ Initialization')
plt.show()
print("The model's inertia is " + str(model.inertia_))


