# Goal: We will use logging to monitor the application of our cluster
import logging
import sys
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import psutil


# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Handler to file
file_handler = logging.FileHandler("ml_pipeline.log")
file_handler.setLevel(logging.DEBUG)  # File will register important information
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Class cluster
class DiabeticCluster:
    def __init__(self):
        logger.info("Load the data")
        self.diabetes_X, self.diabetes_y = datasets.load_diabetes(return_X_y=True)
        self.y_train, self.y_test = None, None
        self.X_train, self.X_test = None, None
        self.regression = None
        self.predict_y = None

    def create_data(self):
        logger.info("Divide train and test")
        self.X_train = self.diabetes_X[:-20]
        self.X_test = self.diabetes_X[-20:]

        self.y_train = self.diabetes_y[:-20]
        self.y_test = self.diabetes_y[-20:]
        logger.debug(f"Shapes - X_train: {self.X_train.shape}, X_test: {self.X_test.shape}")

    def train(self):
        logger.info("Starting train to model")
        try:
            self.regression = linear_model.LinearRegression()
            self.regression.fit(self.X_train, self.y_train)
            logger.info("Success!")
        except Exception as e:
            logger.error(f"Fail to training: {e}")

    def predict(self):
        logger.info("Predict X test")

        if self.regression is None:
            logger.error("Model wasn't training!")
            return
        self.predict_y = self.regression.predict(self.X_test)
        logger.debug(f"Predictions: {self.predict_y[:5]}")

    def eval(self):
        logger.info("Evaluating Model...")
        if self.predict_y is None:
            logger.error("Any predictions find to evaluating!")
            return
        mse = mean_squared_error(self.y_test, self.predict_y)
        logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
        return mse

    @staticmethod
    def system_logs():
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        logger.info(f"System used: CPU={cpu_usage}%, RAM={memory_usage}%")


if __name__ == "__main__":
    class_diabetic = DiabeticCluster()
    class_diabetic.create_data()
    class_diabetic.train()
    class_diabetic.predict()
    class_diabetic.eval()
    class_diabetic.system_logs()
