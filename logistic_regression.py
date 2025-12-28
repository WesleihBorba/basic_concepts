# Goal: Classifying e-mail spam
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
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


class LogisticRegressionModel:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model, self.predict_values = None, None

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_df=0.95,
            min_df=2,
            ngram_range=(1, 2)
        )

        self.data = pd.read_csv('spam.csv', encoding="latin-1")

    def clean_data(self):
        logger.info('Cleaning data to handle')

        def clean_text(t):
            t = re.sub(r'http\S+', '', t)
            t = re.sub(r'[^a-z0-9\s]', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            return t

        self.data = self.data[['v1', 'v2']]
        self.data.columns = ['label', 'text']

        self.data['label_num'] = self.data['label'].map({'ham': 0, 'spam': 1})
        self.data['text'] = self.data['text'].apply(clean_text)

    def train_test(self):
        logger.info('Vectorization text with TF-IDF')
        X = self.vectorizer.fit_transform(self.data['text'])

        logger.info("Divide train and test")
        y = self.data['label_num']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        logger.debug(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    def fitting_data(self):
        logger.info('Fitting with Logistic Regression')
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            penalty='l2',
            solver='liblinear'
        )

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
        logger.debug(f'F1 Score: {f1_score(self.y_test, self.predict_values)}')


class_regression = LogisticRegressionModel()
class_regression.clean_data()
class_regression.train_test()
class_regression.fitting_data()
class_regression.predict_model()
class_regression.evaluating_model()

# Assumptions don't need to test with this kind of data but important in Logistic Regression:
# 1 - Independent observations
# 2 - Large enough sample size - NLP has a lot of words then was enough
# 3 - Features linearly related to log odds - This is already being applied with NLP.
# 4 - Multicollinearity - This issue is solved with penalty='l2' in regression