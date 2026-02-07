# Goal: To learn some filtering methods for regression or classification models
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
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


class FilterMethods:
    def __init__(self, reg_data, class_data):
        self.regression_data = reg_data
        self.classification_data = class_data
        self.scaler, self.normalization_regression, self.normalization_classification = None, None, None

    def normalization(self):
        logger.info('Adjust data to use some methods')
        self.scaler = StandardScaler()
        self.normalization_regression = pd.DataFrame(self.scaler.fit_transform(self.regression_data),
                                                     columns=self.regression_data .columns)
        self.normalization_classification = pd.DataFrame(self.scaler.fit_transform(self.regression_data),
                                                         columns=self.normalization_classification .columns)

    def variance_threshold(self):
        logger.info('Use variance to select features, drop low variance, will not have changes in our models')
        selector = VarianceThreshold(threshold=0.2)
        df_reduced = selector.fit_transform(self.normalization_regression)
        selected_cols = self.normalization_regression.columns[selector.get_support()]
        logger.debug(f"Origin columns: {list(self.normalization_regression.columns)}")
        logger.debug(f"columns that remained: {list(selected_cols)}")
        return df_reduced

    def correlation(self):
        logger.info('Look correlation of each dependency variables in regression (multicollinearity)')
        corr_matrix = self.regression_data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
        df_reduced = self.regression_data.drop(columns=to_drop)
        logger.debug(f"Removing columns with 0.85 of correlation in regression: {to_drop}")
        logger.info("You could use correlation to compare with target variable, then you'll keep this variables")
        return df_reduced


# Criar os dados
regression_data = None
#self.X, self.y = make_regression( # COLOCAR UMA VARIÁVEL COM POUCA VARIÂNCIA OU APENAS UM VALOR COMO 1 OU 2 FIXO (PARA VARIANCIA)
#    n_samples=15000,
#    n_features=2,
#    noise=10,
#    random_state=42)
#self.data = pd.DataFrame(self.X, columns=["education", "working_time"])
#self.data['income'] = self.y
classification_data = None
#self.X, self.y = make_classification(
#    n_samples=15000,
#    n_features=2,
#    n_redundant=0,
#    n_clusters_per_class=1,
#    class_sep=2,
#    random_state=42
#)
#self.data = pd.DataFrame(self.X, columns=["income", "credit_score"])
#self.data["approved"] = self.y
class_filter = FilterMethods()
#class_filter.variance_threshold()
