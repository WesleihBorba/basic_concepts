# Goal:
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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


class RandomForestClassification:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model, self.predict_values = None, None

        self.data = pd.read_csv('files\\loan.csv')





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
        logger.info(f'Best Estimator: {self.model}')

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

# https://www.kaggle.com/code/prashant111/random-forest-classifier-tutorial


classifier = RandomForestClassifier(n_estimators = 100)