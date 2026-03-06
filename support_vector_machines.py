# Goal: Classify data using SVM polynomial kernel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_circles
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


class SupportVector:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.model, self.predict_values = None, None
        self.scale = None
        self.poly_data = self._create_synthetic_data()

    @staticmethod
    def _create_synthetic_data():
        logger.info('Creating our data to use with Polynomial')
        X, y = make_circles(n_samples=500, noise=0.1, factor=0.3, random_state=42)
        df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
        df["fraud_risk_score"] = y
        return df

    def train_test(self):
        logger.info("Divide train and test")
        X = self.poly_data.drop(columns="fraud_risk_score")
        y = self.poly_data["fraud_risk_score"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

    def assumptions(self):
        self.scale = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="poly", degree=3, C=5, coef0=1, gamma="scale"))
        ])
        # Independência: Os dados devem ser independentes e identicamente distribuídos
        pass

    def best_parameters(self):
        # Testar coef0 entre [0,10]
        # degree para poly
        # gamma
        # C
        pass

    def fitting_data(self):
        logger.info('Fitting with Bayes MultinomialNB')

        self.model = MultinomialNB(alpha=1.0)  # Smoothing

        self.model = self.model.fit(self.X_train, self.y_train)

    def predict_model(self):
        logger.info('Predict Test Data')
        self.predict_values = self.model.predict(self.X_test)

    def evaluating_model(self):
        logger.info("Looking if our model is good to use")

        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(self.y_test, self.predict_values),
            display_labels=['Ham', 'Spam']
        )
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix - Spam Classification")
        plt.show()

        logger.debug(f'Precision Score: {precision_score(self.y_test, self.predict_values)}')
        logger.debug(f'Recall score: {recall_score(self.y_test, self.predict_values)}')

    def plot_svm(self):
        pass
