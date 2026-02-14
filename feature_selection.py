# Goal: To learn some filtering methods for regression or classification models
from sklearn.feature_selection import (VarianceThreshold, f_regression, SelectKBest, mutual_info_classif,
                                       mutual_info_regression, RFE, SequentialFeatureSelector)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
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
        logger.info("If you will use this code, we need to divide in train test before normalization"
                    "Don't normalization all dataset, i'm using to show filters methods examples (not real life)")
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

    def test_f_regression(self):
        logger.info('Use to know relation between dependence variable and independence variable in regression')
        selector = SelectKBest(score_func=f_regression, k=5)
        X = self.normalization_regression.drop(columns=['target'])
        y = self.normalization_regression['target']

        select_columns = X.columns[selector.get_support()]
        df_reduced = X[select_columns].copy()
        df_reduced['target'] = y.values

        scores = pd.DataFrame({
            'Feature': self.regression_data.columns,
            'F-Score': selector.scores_,
            'P-Value': selector.pvalues_
        }).sort_values(by='F-Score', ascending=False)
        logger.debug(f'Important columns: {scores}')
        return df_reduced

    def mutual_information(self):
        logger.info('Measuring the statistical dependence between two variables')
        logger.info('Looking classification')
        X = self.normalization_classification.drop(columns=['target'])
        y = self.normalization_classification['target']
        scores = mutual_info_classif(X, y)

        mi_scores = pd.Series(scores, index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)

        selector = SelectKBest(mutual_info_classif, k=5)
        X_new = selector.fit_transform(X, y)
        print(X_new)
        print(mi_scores)

        select_columns = X.columns[selector.get_support()]
        df_reduced_classify = X[select_columns].copy()
        df_reduced_classify['target'] = y.values

        logger.info('Looking Regression')
        X = self.normalization_regression.drop(columns=['target'])
        y = self.normalization_regression['target']
        scores = mutual_info_classif(X, y)

        mi_scores = pd.Series(scores, index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)

        selector = SelectKBest(mutual_info_regression, k=5)
        X_new = selector.fit_transform(X, y)
        print(X_new)
        print(mi_scores)

        select_columns = X.columns[selector.get_support()]
        df_reduced_regression = X[select_columns].copy()
        df_reduced_regression['target'] = y.values
        return df_reduced_classify, df_reduced_regression

    def recursive_feature_elimination(self):
        logger.info('RFE working with Random Forest for classification')
        logger.info('We could use RFECV to discover how much features we will keep')
        model_classification = RandomForestClassifier(n_estimators=10, random_state=42)
        selector = RFE(estimator=model_classification, n_features_to_select=3, step=1)

        X = self.normalization_classification.drop(columns=['target'])
        y = self.normalization_classification['target']

        selector.fit(X, y)

        cols_keep = self.classification_data.columns[selector.support_]
        df_reduced_classify = self.classification_data[cols_keep].copy()

        logger.debug(f"Features kept classification: {list(cols_keep)}")
        logger.debug(f"Ranking features classification (1 is the best): {selector.ranking_}")

        logger.info('RFE working with Multiple Regression for Regression')
        model_regression = LinearRegression()
        selector = RFE(estimator=model_regression, n_features_to_select=3, step=1)

        X = self.normalization_regression.drop(columns=['target'])
        y = self.normalization_regression['target']

        selector.fit(X, y)

        cols_keep = self.normalization_regression.columns[selector.support_]
        df_reduced_regression = self.normalization_regression[cols_keep].copy()

        logger.debug(f"Features kept Regression: {list(cols_keep)}")
        logger.debug(f"Ranking features Regression (1 is the best): {selector.ranking_}")
        return df_reduced_classify, df_reduced_regression

    def sequential_feature_selection(self):
        logger.info('SFE working with Random Forest for classification')
        model_classification = RandomForestClassifier(n_estimators=10, random_state=42)
        sfs_forward = SequentialFeatureSelector(
            model_classification,
            n_features_to_select=2,
            direction='forward',
            scoring='accuracy',
            cv=5
        )

        X = self.normalization_classification.drop(columns=['target'])
        y = self.normalization_classification['target']

        sfs_forward.fit(X, y)

        cols_keep = self.classification_data.columns[sfs_forward.support_]
        df_reduced = self.classification_data[cols_keep].copy()
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
