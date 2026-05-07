# Goal: Segmentation Supermarket mall customers. Understanding more our customers and discover "target customers'
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
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
        self.normalization_data, self.scaler = None, None
        self.model, self.predict_values = None, None

        self.data = pd.read_csv('files\\Mall_Customers.csv')

    def transform_data(self):
        logger.info('Transforming Gender in 1 and 0')
        map_gender = {'Male': 1, 'Female': 0}
        self.data['Gender'] = self.data['Gender'].map(map_gender)
        logger.debug(f'Changes of values, Mall Customers: gender{map_gender}')

        self.data.drop(columns={'CustomerID', 'Gender'}, inplace=True)
        self.data.rename(columns={'Annual Income (k$)': 'annual_income', 'Spending Score (1-100)': 'spending_score'},
                         inplace=True)

    def normalization(self):
        logger.info('Normalization values')
        self.scaler = StandardScaler()
        self.normalization_data = pd.DataFrame(self.scaler.fit_transform(self.data), columns=self.data.columns)

    def elbow_silhouette_method(self):
        logger.info('Elbow: how close the points are within the same cluster')
        logger.info('Silhouette: how far apart the clusters are from each other')
        inertia = []
        sil_scores = []
        k_range = range(2, 11)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(self.normalization_data)
            inertia.append(kmeans.inertia_)
            sil_scores.append(silhouette_score(self.normalization_data, kmeans.labels_))

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
        ).fit(self.normalization_data)

    def plot_groups(self):
        logger.info('Plotting our groups with scatter')
        X = self.normalization_data.values
        labels = self.model.labels_

        plt.figure(figsize=(8, 6))

        for cluster in range(self.model.n_clusters):
            plt.scatter(
                X[labels == cluster, 0],
                X[labels == cluster, 1],
                label=f'Cluster {cluster}',
                s=60
            )

        centers = self.model.cluster_centers_
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            c='black',
            s=200,
            marker='X',
            label='Centroids'
        )

        plt.xlabel('Annual Income (scaled)')
        plt.ylabel('Spending Score (scaled)')
        plt.title('Customer Segmentation - KMeans')
        plt.legend()
        plt.show()

    def interpretation(self):
        self.data['cluster'] = self.model.labels_
        summary = self.data.groupby('cluster')[['annual_income', 'spending_score']].mean()
        logger.info(f'Mean of each group: {summary}')


classification = KMeansClassifier()
classification.transform_data()
classification.normalization()
classification.elbow_silhouette_method()
classification.fitting_data()
classification.plot_groups()
classification.interpretation()