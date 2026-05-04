# Goal: Classify Bank churn using ANN models of Neural Network
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif, SelectKBest, mutual_info_classif
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


# https://www.kaggle.com/competitions/playground-series-s4e1/data

class NeuralNetworkANN:
    def __init__(self):
        self.data = pd.read_csv('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\ANN_train.csv')
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

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

    def divide_train_test(self):
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

    def best_variables(self):
        logger.info('Use ANOVA (f_classify) to find relation between numerical features and categorical target')

        X = self.data.drop(columns=['Exited'])
        y = self.data['Exited']

        selector = SelectKBest(score_func=f_classif, k=5)
        selector.fit(X, y)

        select_columns = X.columns[selector.get_support()]
        df_reduced = X[select_columns].copy()
        df_reduced['Exited'] = y.values
        columns_dropped = list(set(X.columns) - set(select_columns))

        scores = pd.DataFrame({
            'Feature': X.columns,
            'F-Score': selector.scores_,
            'P-Value': selector.pvalues_
        }).sort_values(by='F-Score', ascending=False)

        logger.debug(f'Important classification columns: \n{scores}')
        logger.debug(f"Columns to Drop with f_classify: {columns_dropped}")

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
        columns_dropped_class = list(set(X.columns) - set(select_columns))

        logger.debug(f"Columns to Drop with mutual information: {columns_dropped_class}")

    def hyperparameters_ann(self):
        pass


class_ann = NeuralNetworkANN()
class_ann.exploratory_analyses()
class_ann.divide_train_test()
class_ann.best_variables()