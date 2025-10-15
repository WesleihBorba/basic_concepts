# Goal:
import pandas as pd
import statsmodels.api as sm
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
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class LinearRegression:
    def __init__(self):
        self.train, self.test = None, None
        self.predict_values, self.fit_regression = None, None

        self.X, self.y = make_regression(
            n_samples=15000,
            n_features=1,
            noise=10,
            random_state=42)

        self.data = pd.DataFrame({
            "education": self.X.flatten(),
            "income": self.y
        })

    def train_test(self):
        logger.info("Divide train and test")
        self.train, self.test = train_test_split(self.data['education'], self.data['income'], test_size=0.3,
                                                 random_state=42)

    def fit_model(self):
        logger.info('Starting to fit our regression')

        model = sm.OLS.from_formula('income ~ education', data=self.data)
        self.fit_regression = model.fit()
        print(self.fit_regression.predict(self.data))
        exit()
        logger.info(f"Coefficients: {self.fit_regression.params}")

    def predict_model(self):
        self.predict_values = self.fit_regression.predict(self.data)
        print(self.predict_values)
        #logger.info('')


    def durbin_resid(self):
        exit()
        residuals = self.data.income - self.fit_regression
        dw_stat = durbin_watson(resid)
        print("Rule1: A value around 2 suggests no autocorrelation. - PERFECT",
              "Rule2: Values substantially less than 2 indicate positive autocorrelation.",
              "Rule3: Values much greater than 2 near to 4 suggest negative autocorrelation.")

        print('Durbin-Watson', dw_stat)
        print("If ACF don't show repetitive patterns then it'll correct")
        plot_acf(resid)


        plt.show()
    # Criar regra (def) para ver se a assumption residuals passa como um modelo bom usando também try and excpt



    def summary(self):
        pass

    def plot_linear_regression(self):
        pass

# https://medium.com/@vaibhavkhamitkar12/assumptions-of-linear-regression-a-journey-into-the-world-of-predictive-4a397ed2abf2

LinearRegression().fit_model()
#LinearRegression().predict_model()





exit()



# Fazer a mesma coisa normality and homoscedasticity - TRY AND EXCEPT - TENHO NO NOTE DO TRABALHO
plt.hist(residuals)
plt.show()

plt.scatter(fitted_values, residuals)
plt.show()

