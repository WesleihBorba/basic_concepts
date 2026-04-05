# Goal: Testing ensemble methods with Decision Tree, analyzing each method.
from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier,
                              RandomForestClassifier)
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
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


class EnsembleMethods:

    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

        (self.model_bagging, self.model_adaboost, self.model_xgboost, self.model_gradient_boost,
         self.model_stacking) = [None] * 5

        self.cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=0
        )

        self.parameters_bagging = {  # Reduction variance
            'n_estimators': [50, 100, 200],
            'max_samples': [0.7, 0.8, 1.0],
            'max_features': [0.7, 0.8, 1.0],
            'estimator__max_depth': [None, 10, 20],
            'estimator__min_samples_leaf': [1, 2]
        }

        self.parameters_adaboost = {  # shallow trees
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 1.0],
            'estimator__max_depth': [1, 2, 3]
        }

        self.parameters_gradient = {  # Balance between Bias and Variance
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 8],
            'subsample': [0.8, 1.0],
            'min_samples_leaf': [1, 5, 10]
        }

        self.parameters_xgb = {  # Regularization and overfitting
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 8],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0, 1, 10],
            'subsample': [0.8, 1.0]
        }

        self.algorithm = DecisionTreeClassifier(random_state=0, criterion='gini')

        self.data = pd.read_csv('files\\bankloan.csv')

    def train_test(self):
        logger.info("Divide train and test")
        self.data.drop(columns=["ID"], inplace=True)
        X = self.data.drop(columns={'Personal.Loan'})
        y = self.data['Personal.Loan']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=0
        )
        logger.debug(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    def run_all_ensembles(self):
        logger.info('Running RandomizedSearch for Bagging')
        self.model_bagging = RandomizedSearchCV(
            estimator=BaggingClassifier(estimator=self.algorithm, random_state=0),
            param_distributions=self.parameters_bagging,
            scoring='recall', cv=self.cv, n_jobs=-1, n_iter=10
        )
        self.model_bagging.fit(self.X_train, self.y_train)
        logger.debug(f'Best Bagging Params: {self.model_bagging.best_params_}')

        logger.info('Running RandomizedSearch for AdaBoost')
        self.model_adaboost = RandomizedSearchCV(
            estimator=AdaBoostClassifier(estimator=self.algorithm, random_state=0),
            param_distributions=self.parameters_adaboost,
            scoring='recall', cv=self.cv, n_jobs=-1, n_iter=10
        )
        self.model_adaboost.fit(self.X_train, self.y_train)
        logger.debug(f'Best AdaBoost Params: {self.model_adaboost.best_params_}')

        logger.info('Running RandomizedSearch for Gradient Boosting')
        self.model_gradient_boost = RandomizedSearchCV(
            estimator=GradientBoostingClassifier(random_state=0),
            param_distributions=self.parameters_gradient,
            scoring='recall', cv=self.cv, n_jobs=-1, n_iter=10
        )
        self.model_gradient_boost.fit(self.X_train, self.y_train)
        logger.debug(f'Best GradientBoost Params: {self.model_gradient_boost.best_params_}')

        logger.info('Running RandomizedSearch for XGBoost')
        self.model_xgboost = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(random_state=0, eval_metric='logloss'),
            param_distributions=self.parameters_xgb,
            scoring='recall', cv=self.cv, n_jobs=-1, n_iter=10
        )
        self.model_xgboost.fit(self.X_train, self.y_train)
        logger.debug(f'Best XGBoost Params: {self.model_xgboost.best_params_}')

        logger.info('Running Stacking Ensemble')
        level0_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=0)),
            ('svc', SVC(probability=True, random_state=0)),  # Important to use "probability" to run with stacking
            ('dt', DecisionTreeClassifier(max_depth=5, random_state=0))
        ]
        meta_model = LogisticRegression()  # Level 1 to know weight
        self.model_stacking = StackingClassifier(
            estimators=level0_models,
            final_estimator=meta_model,
            cv=self.cv,
            n_jobs=-1
        )
        self.model_stacking.fit(self.X_train, self.y_train)

    def evaluating_model(self):
        logger.info("Looking if our model is good to use")

        list_models = {
            'Bagging': self.model_bagging,
            'AdaBoost': self.model_adaboost,
            'Gradient Boost': self.model_gradient_boost,
            'Xgboost': self.model_xgboost,
            'stacking': self.model_stacking}

        for name, lists in list_models.items():
            predict_values = lists.predict(self.X_test)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=confusion_matrix(self.y_test, predict_values),
                display_labels=['Approved', 'Denied']
            )
            disp.plot(cmap='Blues')
            plt.title(f"Confusion Matrix - Loan Credit - Using {name}")
            plt.show()

            logger.debug(f'Precision Score - using {name}: {precision_score(self.y_test, predict_values)}')
            logger.debug(f'Recall score - using {name}: {recall_score(self.y_test, predict_values)}')
            logger.debug(f'F1 Score - using {name}: {f1_score(self.y_test, predict_values)}')


class_ensemble = EnsembleMethods()
class_ensemble.train_test()
class_ensemble.run_all_ensembles()
class_ensemble.evaluating_model()