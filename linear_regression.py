# Goal: Understanding Assumption of regression and predict X test
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error, r2_score
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
        self.predict_values = self.fit_regression.predict(self.test['education'])
        self.resid = self.test['income'] - self.predict_values

    def homoscedasticity(self):
        logger.info("Homoscedasticity assumption")

        resid = self.fit_regression.resid
        exog = sm.add_constant(self.train[['education']])

        bp_test = het_breuschpagan(resid, exog)
        labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
        result = dict(zip(labels, bp_test))

        plt.scatter(self.fit_regression.fittedvalues, resid, alpha=0.5)
        plt.axhline(0, linestyle='--', color='red')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted Values')
        plt.show()

        if result['LM-Test p-value'] >= 0.05:
            logger.debug(f"Homoscedasticity confirmed (p-value={result['LM-Test p-value']:.4f})")
        else:
            logger.warning(f"Heteroscedasticity detected (p-value={result['LM-Test p-value']:.4f})")

    def normality_of_residuals(self):
        logger.info("Resid of our model need to follow a normal distribution")

        stat, p_value = shapiro(self.resid)
        logger.info(f"Stats: {stat:.4f}, p-valor: {p_value:.4f}")

        if p_value > 0.05:
            logger.debug("Our model follow a normal distribution")
        else:
            logger.debug("We will need to adjust our model")
            return

    def evaluating_model(self):
        logger.info("Looking if our model is good to use")
        mse = mean_squared_error(self.test['income'], self.predict_values)
        r2 = r2_score(self.test['income'], self.predict_values)

        logger.info(f'Mean Squared Error: {mse}')
        logger.info(f'R-squared: {r2}')

    def plot_linear_regression(self):
        sample = self.test.sample(50)  # 50 random points
        y_true = sample['income']
        y_predict = self.fit_regression.predict(sample['education'])

        plt.scatter(y_true, y_predict, alpha=0.7)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect prediction')

        for i in range(len(sample)):
            plt.plot([y_true.iloc[i], y_true.iloc[i]], [y_true.iloc[i], y_predict.iloc[i]], 'gray', alpha=0.3)

        plt.xlabel('Real value (income)')
        plt.ylabel('Predict value (income)')
        plt.title('Comparison between actual and predicted values (with errors)')
        plt.legend()
        plt.show()


class_regression = LinearRegression()
class_regression.linearity_assumption()
class_regression.train_test()
class_regression.fit_model()
class_regression.predict_model()
class_regression.homoscedasticity()
class_regression.evaluating_model()
class_regression.plot_linear_regression()
