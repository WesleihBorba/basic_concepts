# Goals: Ways to discover the best hyperparameters and consequently good models
import pandas as pd
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
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


class Hyperparameters:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.model, self.predict_values = [None] * 2

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

    def grid_search(self):
        logger.info('Finding best hyperparameters with Grid Search')

        start_time = time.perf_counter()
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['sqrt', 'log2'],
            'class_weight': [None, 'balanced']
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=0
        )

        grid = GridSearchCV(
            estimator=RandomForestClassifier(random_state=0, n_jobs=-1, criterion='gini', bootstrap=True),
            param_grid=param_grid,
            scoring='recall',
            cv=cv,
            n_jobs=-1
        )

        grid.fit(self.X_train, self.y_train)

        logger.info(f'Best parameters: {grid.best_params_}')
        logger.info(f'Best CV score: {grid.best_score_:.4f}')

        self.model = grid.best_estimator_
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        logger.debug(f"Program executed in: {execution_time:.5f} seconds")
        logger.debug(f'Best Estimator of Grid Search: {self.model}')

    def random_search(self):
        logger.info('Finding best hyperparameters with Random Search')

        start_time = time.perf_counter()
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['sqrt', 'log2'],
            'class_weight': [None, 'balanced']
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=0
        )

        grid = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=0, n_jobs=-1, criterion='gini', bootstrap=True),
            param_distributions=param_grid,
            scoring='recall',
            cv=cv,
            n_jobs=-1
        )

        grid.fit(self.X_train, self.y_train)

        logger.info(f'Best parameters: {grid.best_params_}')
        logger.info(f'Best CV score: {grid.best_score_:.4f}')

        self.model = grid.best_estimator_
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        logger.debug(f"Program executed in: {execution_time:.5f} seconds")
        logger.info(f'Best Estimator of Random Search: {self.model}')

    def bayesian_optimization(self):
        search_space = {
            'n_estimators': Integer(50, 200),
            'max_depth': Integer(1, 20),
            'criterion': Categorical(['gini', 'entropy']),
        }

        opt = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=50,  # Number of hyperparameter combinations to try
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1,  # Use all available cores
            random_state=42,
        )


hyperparameters_class = Hyperparameters()
hyperparameters_class.train_test()
#hyperparameters_class.grid_search()
#hyperparameters_class.random_search()