# Goals: Ways to discover the best hyperparameters and consequently good models
import pandas as pd
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import optuna
import pygad
import pygad.kerasga
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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

    def genetic_algorithm(self):
        logger.info('Finding best hyperparameters with Genetic Algorithm (PyGAD)')
        start_time = time.perf_counter()

        def fitness_func(ga_instance_func, solution_func, solution_idx_func):
            params = {
                'n_estimators': int(solution_func[0]),
                'max_depth': int(solution_func[1]),
                'min_samples_split': int(solution_func[2]),
                'min_samples_leaf': int(solution_func[3]),
                'max_features': ['sqrt', 'log2'][int(solution_func[4])],
                'class_weight': [None, 'balanced'][int(solution_func[5])],
            }

            clf = RandomForestClassifier(**params, random_state=0, n_jobs=-1, criterion='gini', bootstrap=True)
            score = cross_val_score(clf, self.X_train, self.y_train,
                                    cv=self.cv, scoring='recall', n_jobs=-1).mean()
            return score

        gene_space = [
            {'low': 50, 'high': 200},  # n_estimators
            {'low': 1, 'high': 20},  # max_depth
            {'low': 2, 'high': 10},  # min_samples_split
            {'low': 1, 'high': 5},  # min_samples_leaf
            [0, 1],  # max_features
            [0, 1]  # class_weight
        ]

        ga_instance = pygad.GA(
            num_generations=10,  # Population evolution
            num_parents_mating=5,  # How many parents reproduce per generation
            fitness_func=fitness_func,
            sol_per_pop=10,  # Size of population
            num_genes=len(gene_space),
            gene_space=gene_space,
            parent_selection_type="sss",  # Steady State Selection
            crossover_type="single_point",
            mutation_type="random",
            mutation_probability=0.1
        )

        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        best_params = {
            'n_estimators': int(solution[0]),
            'max_depth': int(solution[1]),
            'min_samples_split': int(solution[2]),
            'min_samples_leaf': int(solution[3]),
            'max_features': ['sqrt', 'log2'][int(solution[4])],
            'class_weight': [None, 'balanced'][int(solution[5])]
        }

        logger.info(f'Best parameters: {best_params}')
        logger.info(f'Best CV score: {solution_fitness:.4f}')

        self.model_genetic = RandomForestClassifier(**best_params, random_state=0, n_jobs=-1, criterion='gini',
                                                    bootstrap=True)
        self.model_genetic.fit(self.X_train, self.y_train)

        execution_time = time.perf_counter() - start_time
        logger.debug(f"Program executed in: {execution_time:.5f} seconds")

    def evaluating_model(self):
        logger.info("Looking if our model is good to use")

        list_models = {
            'Grid Search': self.model_grid,
            'Random Search': self.model_random,
            'Bayesian Optimizer': self.model_bayesian,
            'Genetic Algorithm': self.model_genetic}

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


hyperparameters_class = Hyperparameters()
hyperparameters_class.train_test()
hyperparameters_class.grid_search()
hyperparameters_class.random_search()
hyperparameters_class.bayesian_optimization()
hyperparameters_class.genetic_algorithm()
hyperparameters_class.evaluating_model()