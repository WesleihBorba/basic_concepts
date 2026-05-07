# Goal: Create a Random Forest, predict loan credit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.inspection import PartialDependenceDisplay
import pandas as pd
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


class RandomForestClassification:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model, self.predict_values = None, None

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

    def fitting_data(self):
        logger.info('Finding best hyperparameters for Random Forest')

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

    def partial_dependence_plot(self):
        if self.model is None:
            raise ValueError("Model not trained yet.")

        features = ['Income', 'CCAvg']
        logger.info(f"Generating Partial Dependence Plot for features: {features}")

        PartialDependenceDisplay.from_estimator(
            estimator=self.model,
            X=self.X_train,
            features=features,
            kind='average',
            grid_resolution=50
        )

        plt.suptitle("Partial Dependence Plot - Loan Credit", fontsize=14)
        plt.tight_layout()
        plt.show()


classification = RandomForestClassification()
classification.train_test()
classification.fitting_data()
classification.feature_importance_analysis()
classification.predict_model()
classification.evaluating_model()
classification.partial_dependence_plot()