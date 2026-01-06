# Goal: Create a Decision Trees, predict loan credit and plot our tree
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
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


class DecisionTreesClassification:

    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model, self.predict_values = None, None

        self.data = pd.read_csv('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\'
                                'Fases da vida\\Fase I\\Repository Projects\\files\\loan.csv') # Deixar apenas o arquivo

    def assumptions_tree(self):
        logger.info('Transforming string in numeric')
        map_loan_status = {'Approved': 1, 'Denied': 0}
        map_marital_status = {'Married': 1, 'Single': 0}
        map_gender = {'Male': 1, 'Female': 0}

        encoder_ordinal = OrdinalEncoder(categories=[["High School", "Associate's", "Bachelor's",
                                                      "Master's", "Doctoral"]])

        encoder = LabelEncoder()

        logger.debug(f'Changes of values, loan: {map_loan_status}, marital: {map_marital_status}, gender{map_gender},'
                     f'Education Level: {encoder_ordinal}')

        self.data['gender'] = self.data['gender'].map(map_gender)
        self.data['marital_status'] = self.data['marital_status'].map(map_marital_status)
        self.data['loan_status'] = self.data['loan_status'].map(map_loan_status)

        self.data['occupation'] = encoder.fit_transform(self.data['occupation'])
        self.data[['education_level']] = encoder_ordinal.fit_transform(self.data[['education_level']])

    def train_test(self):
        logger.info("Divide train and test")
        X = self.data.drop(columns={'loan_status'})
        y = self.data['loan_status']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=0
        )
        logger.debug(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    def fitting_data(self):
        logger.info('Finding best hyperparameters for Decision Tree')

        param_grid = {
            'max_depth': [None, 2, 3, 4, 5, 6, 8, 10],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'criterion': ['gini', 'entropy'],
            'ccp_alpha': [0.0, 0.001, 0.005, 0.01, 0.02]
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=0
        )

        grid = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=0),
            param_grid=param_grid,
            scoring='recall',
            cv=cv,
            n_jobs=-1
        )

        grid.fit(self.X_train, self.y_train)

        logger.info(f'Best parameters: {grid.best_params_}')
        logger.info(f'Best CV score: {grid.best_score_:.4f}')

        self.model = grid.best_estimator_
        print(self.model)

    def feature_importance_analysis(self):
        if self.model is None:
            raise ValueError("Model not trained yet.")

        importance = self.model.feature_importances_
        features = self.X_train.columns

        feature_importance_df = (
            pd.DataFrame({
                'feature': features,
                'importance': importance
            })
            .sort_values(by='importance', ascending=False)
            .reset_index(drop=True)
        )

        logger.info(f"Feature importance calculated: {feature_importance_df}")

    def predict_model(self):
        logger.info('Predict Test Data')
        self.predict_values = self.model.predict(self.X_test)

    def evaluating_model(self):
        logger.info("Looking if our model is good to use")
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(self.y_test, self.predict_values),
            display_labels=['Approved', 'Denied']
        )
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix - Loan Credit")
        plt.show()

        logger.debug(f'Precision Score: {precision_score(self.y_test, self.predict_values)}')
        logger.debug(f'Recall score: {recall_score(self.y_test, self.predict_values)}')
        logger.debug(f'F1 Score: {f1_score(self.y_test, self.predict_values)}')

    def plot_tree(self, max_depth=None):
        logger.info("Plotting Tree")
        plt.figure(figsize=(20, 10))

        plot_tree(
            self.model,
            feature_names=self.X_train.columns,
            class_names=['Denied', 'Approved'],
            filled=True,
            rounded=True,
            max_depth=max_depth,
            fontsize=10
        )

        plt.title(f"Decision Tree (max_depth visualized = {max_depth})")
        plt.show()


classification = DecisionTreesClassification()
classification.assumptions_tree()
classification.train_test()
classification.fitting_data()
classification.feature_importance_analysis()
classification.predict_model()
classification.evaluating_model()
classification.plot_tree()

exit()


# Entender como é feito a separação de cada item nos nó. Por exemplo 2 na esquerda, 3 na direita
# Entender como interpretar decision tree
# Quais modelos de evaluation usar para o tipo de dados que eu usar
# Entender o que é o Depth max
# Entender e usar o pruning
# Como interpretar o resultado da Decision Trees, colocar isso no arquivo também


## https://www.kaggle.com/datasets/sujithmandala/simple-loan-classification-dataset/data
## https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling