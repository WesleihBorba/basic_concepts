# Goal: Understanding Assumption, simpson's paradox and finally predict
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import logging
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class MultipleRegression:
    def __init__(self):
        self.train, self.test = None, None
        self.predict_values, self.fit_regression = None, None
        self.resid = None

        self.X, self.y = make_regression(
            n_samples=15000,
            n_features=2,
            noise=10,
            random_state=42)

        self.data = pd.DataFrame(self.X, columns=["education", "working_time"])
        self.data['income'] = self.y

    def linearity_assumption(self):
        logger.info("A single predictor variable need to have straight-line relationship with the dependent variable")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(self.data["education"], self.data["income"], alpha=0.5)
        axes[0].set_xlabel("Education")
        axes[0].set_ylabel("Income")
        axes[0].set_title("Education vs Income")

        axes[1].scatter(self.data["working_time"], self.data["income"], alpha=0.5)
        axes[1].set_xlabel("Working Time")
        axes[1].set_ylabel("Income")
        axes[1].set_title("Working Time vs Income")

        plt.tight_layout()
        plt.show()

    def correlation(self):
        logger.info("Look at the correlation between the dependent and independent variables")
        sns.heatmap(self.data.corr(numeric_only=True), annot=True, cmap="coolwarm")
        plt.show()

    def test_multicollinearity(self):
        logger.info("Test of Multicollinearity")
        new_X_test = sm.add_constant(self.data[['education', 'working_time']])
        vif_data = pd.DataFrame()
        vif_data["variable"] = new_X_test.columns
        vif_data["VIF"] = [variance_inflation_factor(new_X_test.values, i) for i in range(new_X_test.shape[1])]

        # Drop constant
        vif_data = vif_data[vif_data["variable"] != "const"]

        for _, row in vif_data.iterrows():
            logger.debug(f"Variable: {row['variable']}, VIF: {row['VIF']:.4f}")

        # Attention of something wrong
        high_vif = vif_data[vif_data["VIF"] > 5]
        if not high_vif.empty:
            logger.warning(f"High multicollinearity detected:\n{high_vif}")
        else:
            logger.info("No significant multicollinearity detected.")

    def train_test(self):
        logger.info("Divide train and test")
        self.train, self.test = train_test_split(self.data, test_size=0.3,
                                                 random_state=42)
        logger.debug(f"Shapes - test: {self.test.shape}, train: {self.train.shape}")

    def fit_model(self):
        logger.info('Starting to fit our regression')
        try:
            model = sm.OLS.from_formula('income ~ education + working_time', data=self.train)
            self.fit_regression = model.fit()
            logger.info("Success!")
        except Exception as e:
            logger.error(f"Fail to training: {e}")

        logger.info(f"Coefficients: {self.fit_regression.params}")

    def predict_model(self):
        logger.info('Predict Test Data')
        self.predict_values = self.fit_regression.predict(self.test[['education', 'working_time']])
        self.resid = self.test['income'] - self.predict_values

    def homoscedasticity(self):
        logger.info("Homoscedasticity assumption")

        resid = self.fit_regression.resid
        exog = sm.add_constant(self.train[['education', 'working_time']])

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
        y_predict = self.fit_regression.predict(sample[['education', 'working_time']])

        plt.scatter(y_true, y_predict, alpha=0.7)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect prediction')

        for i in range(len(sample)):
            plt.plot([y_true.iloc[i], y_true.iloc[i]], [y_true.iloc[i], y_predict.iloc[i]], 'gray', alpha=0.3)

        plt.xlabel('Real value (income)')
        plt.ylabel('Predict value (income)')
        plt.title('Comparison between actual and predicted values (with errors)')
        plt.legend()
        plt.show()


class_regression = MultipleRegression()
class_regression.linearity_assumption()
class_regression.correlation()
class_regression.test_multicollinearity()
class_regression.train_test()
class_regression.fit_model()
class_regression.predict_model()
class_regression.homoscedasticity()
class_regression.normality_of_residuals()
class_regression.evaluating_model()
class_regression.plot_linear_regression()

# ASSMPTIONS: independence of errors, normality of errors, and no multicollinearity, CORRELATION BETWEEN VARIABLES



exit()


# Simpson's Paradox: COLOCAR DEPOIS DE CRIAR TUDO, VOU TER QUE CRIAR OUTRA VÁRIAVEL APENAS PARA ISSO EM UMA NOVA DEF ADICIONANDO UMA NOVA VÁRIAVEL EM SELF.DATA
