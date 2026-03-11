# Goal: Reduce the number of features in dataset to classify
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PowerTransformer
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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

    def test_multicollinearity(self):
        logger.info("Test of Multicollinearity")
        X = self.classification_data.drop(columns={'approved'})

        new_X_test = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["variable"] = new_X_test.columns
        vif_data["VIF"] = [variance_inflation_factor(new_X_test.values, i) for i in range(new_X_test.shape[1])]

        # Drop constant
        vif_data = vif_data[vif_data["variable"] != "const"]

        for _, row in vif_data.iterrows():
            logger.debug(f"Variable: {row['variable']}, VIF: {row['VIF']:.4f}")

        # Attention of something wrong
        high_vif = vif_data[vif_data["VIF"] > 5]
        if not high_vif.empty:
            logger.warning(f"High multicollinearity detected:\n{high_vif}")
        else:
            logger.info("No significant multicollinearity detected.")

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

    def assumptions(self):
        logger.info('Fixing Normality, transforming in normal variables')
        X = self.classification_data.drop(columns={'approved'})
        y = self.classification_data['approved']

        columns = X.columns
        for col in columns:
            stat, p = shapiro(self.classification_data[col])
            if p < 0.05:
                logger.debug(f'The variable {col} is NOT normally distributed.')
                pt = PowerTransformer(method='yeo-johnson')
                self.normalization_classification = pd.DataFrame(pt.fit_transform(X), columns=X.columns)
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