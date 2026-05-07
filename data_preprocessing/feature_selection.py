# Goal: To learn some filtering methods for regression or classification models
from sklearn.feature_selection import (VarianceThreshold, f_regression, f_classif, SelectKBest, mutual_info_classif,
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

        X_reg = self.regression_data.drop(columns={'income'})
        y_reg = self.regression_data['income']
        self.normalization_regression = pd.DataFrame(self.scaler.fit_transform(X_reg),
                                                     columns=X_reg.columns)
        self.normalization_regression['income'] = y_reg.values

        X_class = self.classification_data.drop(columns={'approved'})
        y_class = self.classification_data['approved']
        self.normalization_classification = pd.DataFrame(self.scaler.fit_transform(X_class),
                                                         columns=X_class.columns)
        self.normalization_classification['approved'] = y_class.values

    def variance_threshold(self):
        logger.info('Use variance to select features, drop low variance, will not have changes in our models')
        X = self.normalization_regression.drop(columns=['income'])
        selector = VarianceThreshold(threshold=0.8)

        selector.fit(X)
        selected_cols = X.columns[selector.get_support()]
        df_reduced = X[selected_cols].copy()
        columns_dropped = list(set(df_reduced.columns) - set(selected_cols))
        logger.debug(f"Columns Dropped: {columns_dropped}")

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
        logger.debug(f"Columns to Drop: {to_drop}")
        logger.debug(f'Dataframe: {df_reduced}')

    def test_f_regression(self):
        logger.info('Use to know relation between dependence variable and independence variable in regression')
        selector = SelectKBest(score_func=f_regression, k=5)
        X = self.normalization_regression.drop(columns=['income'])
        y = self.normalization_regression['income']

        selector.fit(X, y)
        select_columns = X.columns[selector.get_support()]
        df_reduced = X[select_columns].copy()
        df_reduced['income'] = y.values
        columns_dropped = list(set(X.columns) - set(select_columns))

        scores = pd.DataFrame({
            'Feature': X.columns,
            'F-Score': selector.scores_,
            'P-Value': selector.pvalues_
        }).sort_values(by='F-Score', ascending=False)

        logger.debug(f'Important columns: {scores}')
        logger.debug(f"Columns to Drop: {columns_dropped}")

    def test_f_classify(self, k_features=5):
        logger.info('Use ANOVA (f_classify) to find relation between numerical features and categorical target')

        X = self.normalization_classification.drop(columns=['approved'])
        y = self.normalization_classification['approved']

        selector = SelectKBest(score_func=f_classif, k=k_features)
        selector.fit(X, y)

        select_columns = X.columns[selector.get_support()]
        df_reduced = X[select_columns].copy()
        df_reduced['approved'] = y.values
        columns_dropped = list(set(X.columns) - set(select_columns))

        scores = pd.DataFrame({
            'Feature': X.columns,
            'F-Score': selector.scores_,
            'P-Value': selector.pvalues_
        }).sort_values(by='F-Score', ascending=False)

        logger.debug(f'Important classification columns: \n{scores}')
        logger.debug(f"Columns to Drop (Classify): {columns_dropped}")

        return df_reduced

    def mutual_information(self):
        logger.info('Measuring the statistical dependence between two variables')
        logger.info('Looking classification')
        X_class = self.normalization_classification.drop(columns=['approved'])
        y_class = self.normalization_classification['approved']
        scores = mutual_info_classif(X_class, y_class)

        mi_scores = pd.Series(scores, index=X_class.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        logger.debug(f'values of scores of classification {mi_scores}')

        selector = SelectKBest(mutual_info_classif, k=5)
        selector.fit(X_class, y_class)

        select_columns = X_class.columns[selector.get_support()]
        df_reduced_classify = X_class[select_columns].copy()
        df_reduced_classify['approved'] = y_class.values
        columns_dropped_class = list(set(X_class.columns) - set(select_columns))

        logger.debug(f"Columns to Drop Classification: {columns_dropped_class}")

        logger.info('Looking Regression')
        X_reg = self.normalization_regression.drop(columns=['income'])
        y_reg = self.normalization_regression['income']
        scores = mutual_info_regression(X_reg, y_reg)

        mi_scores = pd.Series(scores, index=X_reg.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        logger.debug(f'values of scores of Regression {mi_scores}')

        selector = SelectKBest(mutual_info_regression, k=5)
        selector.fit(X_reg, y_reg)

        select_columns = X_reg.columns[selector.get_support()]
        df_reduced_regression = X_reg[select_columns].copy()
        df_reduced_regression['income'] = y_reg.values
        columns_dropped_regression = list(set(X_reg.columns) - set(select_columns))
        logger.debug(f"Columns to Drop regression: {columns_dropped_regression}")

    def recursive_feature_elimination(self):
        logger.info('RFE working with Random Forest for classification')
        logger.info('We could use RFECV to discover how much features we will keep')
        model_classification = RandomForestClassifier(n_estimators=10, random_state=42)
        selector_class = RFE(estimator=model_classification, n_features_to_select=4, step=1)

        X_class = self.normalization_classification.drop(columns=['approved'])
        y_class = self.normalization_classification['approved']

        selector_class.fit(X_class, y_class)

        cols_keep = X_class.columns[selector_class.support_]
        df_reduced_classify = X_class[cols_keep].copy()
        logger.debug(f'Columns of new dataframe classify: {df_reduced_classify.columns}')

        columns_dropped_classify = list(set(X_class.columns) - set(cols_keep))
        logger.debug(f"Columns to Drop classify: {columns_dropped_classify}")
        logger.debug(f"Ranking features classification (1 is the best): {selector_class.ranking_}")

        logger.info('RFE working with Multiple Regression for Regression')
        model_regression = LinearRegression()
        selector_reg = RFE(estimator=model_regression, n_features_to_select=3, step=1)

        X_reg = self.normalization_regression.drop(columns=['income'])
        y_reg = self.normalization_regression['income']

        selector_reg.fit(X_reg, y_reg)

        cols_keep = X_reg.columns[selector_reg.support_]
        df_reduced_regression = self.normalization_regression[cols_keep].copy()
        logger.debug(f'Columns of new dataframe Regression: {df_reduced_regression.columns}')

        columns_dropped_regression = list(set(X_reg.columns) - set(cols_keep))
        logger.debug(f"Columns to Drop Regression: {columns_dropped_regression}")
        logger.debug(f"Ranking features Regression (1 is the best): {selector_reg.ranking_}")

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

        X = self.normalization_classification.drop(columns=['approved'])
        y = self.normalization_classification['approved']

        sfs_forward.fit(X, y)

        cols_keep = X.columns[sfs_forward.support_]
        df_reduced = X[cols_keep].copy()
        columns_dropped = list(set(X.columns) - set(cols_keep))
        logger.debug(f"Columns of new dataframe: {df_reduced}")
        logger.debug(f"Columns to Drop: {columns_dropped}")

    def feature_importance_trees(self):
        logger.info('Selecting the best features using Decision Trees such as the Gini with criteria.')
        X = self.normalization_classification.drop(columns=['approved'])
        y = self.normalization_classification['approved']
        y = [int(label) for label in y]

        clf = DecisionTreeClassifier(criterion='gini')
        clf = clf.fit(X, y)
        importance = dict(zip(X.columns, clf.feature_importances_))

        threshold = 0.001
        cols_to_remove = [col for col, imp in importance.items() if imp < threshold]

        df_reduced = X.drop(columns=cols_to_remove)

        logger.debug(f'Columns to remove: {cols_to_remove}')
        logger.debug(f'Columns of new dataframe: {df_reduced}')


class_filter = FilterMethods()
class_filter.normalization()
class_filter.variance_threshold()
class_filter.correlation()
class_filter.test_f_regression()
class_filter.test_f_classify()
class_filter.mutual_information()
class_filter.recursive_feature_elimination()
class_filter.sequential_feature_selection()
class_filter.feature_importance_trees()