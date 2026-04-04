# Goal: Testing ensemble methods with Decision Tree, analyzing each method.
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
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
        self.best_model, self.predict_bagging = [None] * 2
        self.number_estimator = 100
        self.cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=0
        )

        self.parameters = {
            'ccp_alpha': [0.0, 0.01, 0.1],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['sqrt', 'log2'],
            'class_weight': [None, 'balanced']
        }

        self.algorithm = DecisionTreeClassifier(random_state=0, criterion='gini')

        self.data = pd.read_csv('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\bankloan.csv')

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

    def random_search(self):
        logger.info('Finding best hyperparameters with Random Search')

        grid = RandomizedSearchCV(
            estimator=self.algorithm,
            param_distributions=self.parameters,
            scoring='recall',
            cv=self.cv,
            n_jobs=-1
        )

        grid.fit(self.X_train, self.y_train)

        logger.info(f'Best parameters: {grid.best_params_}')
        logger.info(f'Best CV score: {grid.best_score_:.4f}')

        self.best_model = grid.best_estimator_

    def bagging_method(self):
        logger.info('Starting with bagging but its very similar of Random Forest. Better use Random Forest')
        bagging_model = BaggingClassifier(estimator=self.best_model, n_estimators=self.number_estimator,
                                          random_state=42, bootstrap=True, n_jobs=-1)

        bagging_model.fit(self.X_train, self.y_train)
        self.predict_bagging = bagging_model.predict(self.X_test)

    def boosting_method(self):
        logger.info('Running Boosting methods')