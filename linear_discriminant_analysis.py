# Goal: Reduce the number of features in dataset to classify
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
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


class LinearDiscriminant:

    def __init__(self):
        self.X_class, self.y_class = make_classification(
            n_samples=20000,
            n_features=12,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=2,
            random_state=42
        )
        columns_classification = ["income", "credit_score", "debt_to_income_ratio", "employment_history",
                                  "loan_amount", "loan_term", "previous_defaults", "savings_balance", "property_value",
                                  "number_of_open_credits", "age", "residence_type"]
        self.classification_data = pd.DataFrame(self.X_class, columns=columns_classification)
        self.classification_data["approved"] = self.y_class

        self.components, self.lda_data = None, None
        self.scaler, self.normalization_classification = None, None

    def normalization(self):
        logger.info('Adjust data to use some methods')
        logger.info("If you will use this code, we need to divide in train test before normalization"
                    "Don't normalization all dataset, i'm using to show filters methods examples (not real life)")
        self.scaler = StandardScaler()

        X = self.classification_data.drop(columns={'approved'})
        y = self.classification_data['approved']
        self.normalization_classification = pd.DataFrame(self.scaler.fit_transform(X),
                                                         columns=X.columns)
        self.normalization_classification['approved'] = y.values

    def pca_model(self):
        logger.info('Using PCA model and creating dataframe')
        X = self.normalization_classification.drop(columns={'approved'})
        y = self.normalization_classification['approved']
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda_array = lda.fit_transform(X)

    def plot_pca(self):
        logger.info('Create graph if have less than 3 dimensional')
        if self.components is None or self.pca_data is None:
            logger.debug("Run model or fix it")
            return

        if self.components == 2:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=self.pca_data, x='PC1', y='PC2', hue='approved', alpha=0.6)
            plt.title('PCA - 2 Dimensional')
            plt.show()
        elif self.components == 1:
            plt.figure(figsize=(10, 4))
            sns.scatterplot(data=self.pca_data, x='PC1', y=[0]*len(self.pca_data), hue='approved', alpha=0.5)
            plt.title('PCA - 1 Dimensional')
            plt.show()
        else:
            logger.debug(f"Won't have plot because have more than 2 components.")


pca_class = PrincipalComponentAnalysis()
pca_class.normalization()
pca_class.number_of_components()
pca_class.pca_model()
pca_class.plot_pca()