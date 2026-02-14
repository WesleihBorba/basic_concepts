# Goal: To learn some filtering methods for regression or classification models
from sklearn.feature_selection import (VarianceThreshold, f_regression, SelectKBest, mutual_info_classif,
                                       mutual_info_regression, RFE, SequentialFeatureSelector)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_regression, make_classification
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
    def __init__(self):
        logger.info("It's an example, not represent data in real life")
        self.X_reg, self.y_reg = make_regression(
            n_samples=20000,
            n_features=12,
            noise=10,
            random_state=42)
        columns_regression = ["education", "working_time", "age", "experience_years", "job_level", "company_size",
                              "city_tier", "technical_skills", "soft_skills", "industry_sector", "previous_salary",
                              "english_level"]
        self.regression_data = pd.DataFrame(self.X_reg, columns=columns_regression)
        self.regression_data['no_variation'] = 50
        self.regression_data['income'] = self.y_reg

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

        self.scaler, self.normalization_regression, self.normalization_classification = None, None, None

    def normalization(self):
        logger.info('Adjust data to use some methods')
        logger.info("If you will use this code, we need to divide in train test before normalization"
                    "Don't normalization all dataset, i'm using to show filters methods examples (not real life)")
        self.scaler = StandardScaler()
        self.normalization_regression = pd.DataFrame(self.scaler.fit_transform(self.regression_data),
                                                     columns=self.regression_data.columns)
        self.normalization_classification = pd.DataFrame(self.scaler.fit_transform(self.classification_data),
                                                         columns=self.classification_data.columns)

    def variance_threshold(self):
        logger.info('Use variance to select features, drop low variance, will not have changes in our models')
        X = self.normalization_regression.drop(columns=['income'])
        y = self.normalization_regression['income']
        selector = VarianceThreshold(threshold=0.8)

        selector.fit(X)
        selected_cols = X.columns[selector.get_support()]
        df_reduced = X[selected_cols].copy()
        logger.debug(f"Origin columns: {list(self.normalization_regression.columns)}")
        logger.debug(f"columns that remained: {list(selected_cols)}")

    def correlation(self, target_column='income', threshold=0.3):
        logger.info(f'Filtering features based on correlation with {target_column}')

        corr_matrix = self.regression_data.corr()
        target_corr = corr_matrix[target_column].abs()
        low_corr_features = target_corr[target_corr >= threshold].index.tolist()

        logger.info('Drop features with high correlation (Multicollinearity)')
        abs_corr_matrix = corr_matrix.abs()
        upper = abs_corr_matrix.where(np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(bool))
        redundant_features = [column for column in upper.columns if any(upper[column] > 0.85)]

        to_drop = list(set(low_corr_features + redundant_features))
        if target_column in to_drop:
            to_drop.remove(target_column)

        df_reduced = self.regression_data.drop(columns=to_drop)
        logger.debug(f"Origin columns: {list(self.regression_data.columns)}")
        logger.debug(f"columns that remained: {list(df_reduced.columns)}")

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

    def feature_importance_trees(self):
        logger.info('Selecting the best features using Decision Trees such as the Gini with criteria.')
        X = self.normalization_classification.drop(columns=['target'])
        y = self.normalization_classification['target']
        y = [int(label) for label in y]

        clf = DecisionTreeClassifier(criterion='gini')
        clf = clf.fit(X, y)
        feature_important = clf.feature_importances_
        print(feature_important) # Trocar para o dataframe correto


class_filter = FilterMethods()
class_filter.normalization()
class_filter.variance_threshold()
class_filter.correlation()