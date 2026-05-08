# Goal: Classify Bank churn (customer continues with their account or closes) using ANN models of Neural Network
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import f_classif, SelectKBest, mutual_info_classif
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
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


class NeuralNetworkANN:
    def __init__(self):
        self.data = pd.read_csv('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\ANN_train.csv')
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.columns_to_exclude_mutual, self.columns_to_exclude_anova = [None] * 2
        self.best_model = None

        self.cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=0
        )

    def exploratory_analyses(self):
        logger.info('Null data?')
        null_group_train = self.data.isnull().sum().sum()
        if null_group_train > 0:
            logger.warning(f"Null data in any group: {null_group_train}")
        else:
            logger.debug('Not Null')

        logger.info('Duplicated data?')
        duplicated_group_train = self.data.duplicated().sum()
        if duplicated_group_train > 0:
            logger.warning(f"Duplicated data in any group: {duplicated_group_train}")
        else:
            logger.debug('Not Duplicated')

        logger.info('Deleting unimportant columns')
        columns_to_delete = ['id', 'CustomerId', 'Surname']
        self.data.drop(columns=columns_to_delete, inplace=True)

        logger.info('Converting Gender in binary')
        self.data['Gender'] = self.data['Gender'].map({'Female': 0, 'Male': 1})

        logger.info('Transforming Geography data')
        encoder = LabelEncoder()
        self.data['Geography'] = encoder.fit_transform(self.data['Geography'])

    def best_variables(self):
        logger.info('Use ANOVA (f_classify) to find relation between numerical features and categorical target')

        X = self.data.drop(columns=['Exited'])
        y = self.data['Exited']

        selector = SelectKBest(score_func=f_classif, k=5)
        selector.fit(X, y)

        select_columns = X.columns[selector.get_support()]
        df_reduced = X[select_columns].copy()
        df_reduced['Exited'] = y.values
        self.columns_to_exclude_anova = list(set(X.columns) - set(select_columns))

        scores = pd.DataFrame({
            'Feature': X.columns,
            'F-Score': selector.scores_,
            'P-Value': selector.pvalues_
        }).sort_values(by='F-Score', ascending=False)

        logger.debug(f'Important classification columns: \n{scores}')
        logger.debug(f"Columns to Drop with f_classify: {self.columns_to_exclude_anova}")

        logger.info('Measuring the statistical dependence between two variables')
        scores = mutual_info_classif(X, y)

        mi_scores = pd.Series(scores, index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        logger.debug(f'values of scores of classification {mi_scores}')

        selector = SelectKBest(mutual_info_classif, k=5)
        selector.fit(X, y)

        select_columns = X.columns[selector.get_support()]
        df_reduced_classify = X[select_columns].copy()
        df_reduced_classify['Exited'] = y.values
        self.columns_to_exclude_mutual = list(set(X.columns) - set(select_columns))

        logger.debug(f"Columns to Drop with mutual information: {self.columns_to_exclude_mutual}")

    def divide_train_test(self):
        logger.info('Excluding other columns based on the previous model')
        self.data.drop(columns=self.columns_to_exclude_anova, inplace=True)

        logger.info('Dividing in train and test')
        X = self.data.drop(columns='Exited')
        y = self.data['Exited']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        logger.info('Scaling our data')
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    @staticmethod
    def model_ann(learning_rate=0.01, num_units=64, meta=None):
        n_features = meta['n_features_in_'] if meta else 5
        model = Sequential([
            InputLayer(input_shape=(n_features,)),
            Dense(num_units, activation='relu'),  # Number of columns less Y
            Dropout(0.2),  # Overfitting in training
            Dense(num_units // 2, activation='relu'),  # Half of first (32)
            Dense(1, activation='sigmoid')
        ])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def hyperparameters_ann(self):
        logger.info('Finding best parameters with Random Search')  # Needs to be a function in "build_fn"
        model = KerasClassifier(model=self.model_ann, verbose=0, learning_rate=0.01, num_units=64)

        param_dist = {
            'model__learning_rate': [0.001, 0.01],
            'model__num_units': [32, 64],
            'batch_size': [32],
            'epochs': [10, 20]
        }

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            scoring='recall',
            cv=self.cv,
            n_iter=5,
            n_jobs=1
        )
        search.fit(self.X_train, self.y_train)

        logger.info(f'Best parameters: {search.best_params_}')
        logger.info(f'Best CV score: {search.best_score_:.4f}')

        self.best_model = search.best_estimator_
        print(self.best_model)

    def evaluate_model(self):
        logger.info('Evaluating the best model on test data')
        y_predict = self.best_model.predict(self.X_test)
        y_probs = self.best_model.predict_proba(self.X_test)[:, 1]

        cm = confusion_matrix(self.y_test, y_predict)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()

        logger.info("\n--- Classification Report ---")
        logger.info(f'{classification_report(self.y_test, y_predict)}')

        # 4. AUC Score
        auc = roc_auc_score(self.y_test, y_probs)
        logger.info(f"AUC Score: {auc:.4f}")


class_ann = NeuralNetworkANN()
class_ann.exploratory_analyses()
class_ann.best_variables()
class_ann.divide_train_test()
class_ann.hyperparameters_ann()
