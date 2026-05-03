# Goal: Classify Bank churn using ANN models of Neural Network
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
        self.data_train = pd.read_csv('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\ANN_train.csv')
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def exploratory_analyses(self):
        logger.info('Null data?')
        null_group_train = self.data_train.isnull().sum().sum()
        if null_group_train > 0:
            logger.warning(f"Null data in any group: {null_group_train}")
        else:
            logger.debug('Not Null')

        logger.info('Duplicated data?')
        duplicated_group_train = self.data_train.duplicated().sum()
        if duplicated_group_train > 0:
            logger.warning(f"Duplicated data in any group: {duplicated_group_train}")
        else:
            logger.debug('Not Duplicated')

        logger.info('Deleting unimportant columns')
        columns_to_delete = ['id', 'CustomerId', 'Surname']
        self.data_train.drop(columns=columns_to_delete, inplace=True)

        logger.info('Converting Gender in binary')
        self.data_train['Gender'] = self.data_train['Gender'].map({'Female': 0, 'Male': 1})

        logger.info('Transforming Geography data')
        encoder = LabelEncoder()
        self.data_train['Geography'] = encoder.fit_transform(self.data_train['Geography'])

    def divide_train_test(self):
        logger.info('Dividing in train and test')
        X = self.data_train.drop(columns='Exited')
        y = self.data_train['Exited']
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
        pass # USAR f_classif E Mutual Information

    def hyperparameters_ann(self):
        pass


class_ann = NeuralNetworkANN()
class_ann.exploratory_analyses()