# Goal: Reduce the number of features in dataset to classify
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PowerTransformer
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import levene
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
        self.columns = ["income", "credit_score", "debt_to_income_ratio", "employment_history",
                        "loan_amount", "loan_term", "previous_defaults", "savings_balance", "property_value",
                        "number_of_open_credits", "age", "residence_type"]
        self.classification_data = pd.DataFrame(self.X_class, columns=self.columns)
        self.classification_data["approved"] = self.y_class

        self.components, self.lda_data, self.X_lda = None, None, None
        self.scaler, self.prepared_data = None, None

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

    def run_assumptions_and_fix(self):
        X = self.classification_data.drop(columns={'approved'})
        y = self.classification_data['approved']

        logger.info('Fixing Normality, transforming in normal variables')
        skewness = X.skew()
        if any(abs(skewness) > 0.75):
            logger.debug("High skewness detected. Applying PowerTransformer (Yeo-Johnson)...")
            pt = PowerTransformer(method='yeo-johnson')
            X_transformed = pt.fit_transform(X)
            self.prepared_data = pd.DataFrame(X_transformed, columns=self.columns)
        else:
            logger.debug("Data is fairly symmetric. No PowerTransformer needed.")
            self.prepared_data = X.copy()

        self.prepared_data['approved'] = y.values

        logger.info("Testing Homoscedasticity (Levene's Test)")
        for col in self.columns:
            group0 = self.prepared_data[self.prepared_data['approved'] == 0][col]
            group1 = self.prepared_data[self.prepared_data['approved'] == 1][col]
            stat, p = levene(group0, group1)

            if p < 0.05:
                logger.warning(f"Variable {col} has heterogeneous variance (p={p:.4f})")
            else:
                logger.debug(f"Variable {col} has equal variance (p={p:.4f})")

    def fit_lda(self):
        logger.info('Running LDA')

        lda = LinearDiscriminantAnalysis()
        X = self.prepared_data.drop(columns=['approved'])
        y = self.prepared_data['approved']

        self.X_lda = lda.fit_transform(X, y)
        logger.info("LDA fitted. Explained variance ratio: %s", lda.explained_variance_ratio_)

    def plot_lda(self):
        logger.info('Create graph after LDA transformation')
        plt.figure(figsize=(10, 6))

        df_plot = pd.DataFrame({'LDA_Component': self.X_lda.flatten(), 'Target': self.prepared_data['approved']})

        sns.kdeplot(data=df_plot, x='LDA_Component', hue='Target', fill=True)
        plt.title('Separation of our classes')
        plt.xlabel('Linear Components as 1')
        plt.ylabel('Density')
        plt.show()


lda_class = LinearDiscriminant()
lda_class.test_multicollinearity()
lda_class.run_assumptions_and_fix()
lda_class.fit_lda()
lda_class.plot_lda()