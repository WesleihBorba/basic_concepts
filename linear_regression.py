# Goal:
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import logging
import sys
import matplotlib.pyplot as plt

# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class LinearRegression:
    def __init__(self):
        self.train, self.test = None, None
        self.predict_values, self.fit_regression = None, None
        self.resid = None

        self.X, self.y = make_regression(
            n_samples=15000,
            n_features=1,
            noise=10,
            random_state=42)

        self.data = pd.DataFrame({
            "education": self.X.flatten(),
            "income": self.y
        })

    def linearity_assumption(self):
        logger.info("A single predictor variable need to have straight-line relationship with the dependent variable")
        plt.scatter(self.X, self.y)
        plt.xlabel("Education")
        plt.ylabel("Income")
        plt.title("Linearity Assumption")
        plt.show()

    def train_test(self):
        logger.info("Divide train and test")
        self.train, self.test = train_test_split(self.data, test_size=0.3,
                                                 random_state=42)
        logger.debug(f"Shapes - test: {self.test.shape}, train: {self.train.shape}")

    def fit_model(self):
        logger.info('Starting to fit our regression')
        try:
            model = sm.OLS.from_formula('income ~ education', data=self.train)
            self.fit_regression = model.fit()
            logger.info("Success!")
        except Exception as e:
            logger.error(f"Fail to training: {e}")

        logger.info(f"Coefficients: {self.fit_regression.params}")

    def predict_model(self):
        logger.info('Predict Test Data')
        self.predict_values = self.fit_regression.predict(self.test)
        self.resid = self.test['income'] - self.predict_values

    def homoscedasticity(self):
        logger.info("Homoscedasticity assumption")

        exog = sm.add_constant(self.test[['education']])
        bp_test = het_breuschpagan(self.resid, exog)
        labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
        result = dict(zip(labels, bp_test))

        logger.info('Plot of resid')
        plt.hist(self.resid)
        plt.show()

        if result['LM-Test p-value'] >= 0.05:
            logger.debug('the dispersion of data around the mean is similar in all groups or for all values of the '
                         'predictor variable')
        else:
            logger.error("the model contains Heteroscedasticity")
            return

    def normality_of_residuals(self):
        logger.info("Resid of our model need to follow a normal distribution")

        stat, p_value = shapiro(self.resid)
        logger.info(f"Stats: {stat:.4f}, p-valor: {p_value:.4f}")

        if p_value > 0.05:
            logger.debug("Our model follow a normal distribution")
        else:
            logger.debug("We will need to adjust our model")
            return

    def summary(self):
        pass

    def plot_linear_regression(self):
        pass


class_regression = LinearRegression()
# LinearRegression().linearity_assumption()
class_regression.train_test()
class_regression.fit_model()
class_regression.predict_model()
class_regression.homoscedasticity()

# VER COM BASE NO EMAIL QUE EU ENVIEI SE TEM ALGO A MAIS PARA SER USADO


exit()


