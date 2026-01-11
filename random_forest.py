# Goal:
from sklearn.ensemble import RandomForestClassifier
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


# https://www.kaggle.com/code/prashant111/random-forest-classifier-tutorial


classifier = RandomForestClassifier(n_estimators = 100)