# Goals: Ways to discover the best hyperparameters and consequently good models
import pandas as pd
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.model_selection import cross_val_score
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
        self.model_grid, self.model_random, self.model_bayesian, self.model_genetic = [None] * 4
        self.cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=0
        )

        self.parameters = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['sqrt', 'log2'],
            'class_weight': [None, 'balanced']
        }

        self.algorithm = RandomForestClassifier(random_state=0, n_jobs=-1, criterion='gini', bootstrap=True)

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

        grid = GridSearchCV(
            estimator=self.algorithm,
            param_grid=self.parameters,
            scoring='recall',
            cv=self.cv,
            n_jobs=-1
        )

        grid.fit(self.X_train, self.y_train)

        logger.info(f'Best parameters: {grid.best_params_}')
        logger.info(f'Best CV score: {grid.best_score_:.4f}')

        self.model_grid = grid.best_estimator_
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        logger.debug(f"Program executed in: {execution_time:.5f} seconds")
        logger.debug(f'Best Estimator of Grid Search: {self.model_grid}')

    def random_search(self):
        logger.info('Finding best hyperparameters with Random Search')

        start_time = time.perf_counter()

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

        self.model_random = grid.best_estimator_
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        logger.debug(f"Program executed in: {execution_time:.5f} seconds")
        logger.debug(f'Best Estimator of Random Search: {self.model_random}')

    def bayesian_optimization(self):
        logger.info('Finding best hyperparameters with Bayesian optimization')

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            }

            clf = RandomForestClassifier(**params, random_state=0, n_jobs=-1, criterion='gini', bootstrap=True)

            score = cross_val_score(clf, self.X_train, self.y_train,
                                    cv=self.cv, scoring='recall', n_jobs=-1).mean()
            return score

        start_time = time.perf_counter()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        logger.info(f'Best parameters: {study.best_params}')
        logger.info(f'Best CV score: {study.best_value:.4f}')

        self.model_bayesian = RandomForestClassifier(**study.best_params, random_state=0, n_jobs=-1, criterion='gini',
                                                     bootstrap=True)
        self.model_bayesian.fit(self.X_train, self.y_train)

        end_time = time.perf_counter()

        execution_time = end_time - start_time
        logger.debug(f"Program executed in: {execution_time:.5f} seconds")
        logger.debug(f'Best Estimator of Random Search: {self.model_bayesian}')


    def genetic_algorithm(self):
        pass
    
    def evaluating_model(self):
        pass

hyperparameters_class = Hyperparameters()
hyperparameters_class.train_test()
#hyperparameters_class.grid_search()
#hyperparameters_class.random_search()
hyperparameters_class.bayesian_optimization()