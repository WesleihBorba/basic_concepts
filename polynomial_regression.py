# Goal: Running polynomial regression, check assumptions and predict values
from statsmodels.stats.stattools import durbin_watson
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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np

# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class PolynomialRegression:
    def __init__(self):
        self.train, self.test = None, None
        self.predict_values, self.fit_regression = None, None
        self.resid = None

        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=3, include_bias=False)

        n = 15000

        education = np.random.normal(10, 3, n)
        working_time = np.random.uniform(20, 60, n)

        noise = np.random.normal(0, 30, n)

        income = (
                0.8 * education ** 3
                - 5 * education ** 2
                + 2 * working_time
                + 0.5 * working_time ** 2
                + noise
        )

        self.X = np.column_stack([education, working_time])
        self.y = income

        self.data = pd.DataFrame({
            "education": education,
            "working_time": working_time,
            "income": income
        })

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

    def detect_non_linear_relation(self):
        logger.debug("X variable and Y variable must be non linear relation to use poly regression")

        for X in self.data.columns:
            if (X == 'education') or (X == 'working_time'):
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=self.data[f'{X}'], y=self.data['income'])
                plt.title("Scatter Plot of Nonlinear Data")
                plt.show()

    def train_test_scaling(self):
        logger.info("Divide train and test")

        train_df, test_df = train_test_split(
            self.data,
            test_size=0.3,
            random_state=42
        )

        X_train = train_df.drop(columns={'income'})
        y_train = train_df['income']

        X_test = test_df.drop(columns={'income'})
        y_test = test_df['income']

        logger.info("Transforming our X in Non-Linear giving other variables")
        x_train_poly = self.poly.fit_transform(X_train)
        x_test_poly = self.poly.fit_transform(X_test)

        feature_names = self.poly.get_feature_names_out(['education', 'working_time'])

        x_poly_train = pd.DataFrame(x_train_poly, columns=feature_names)
        x_poly_test = pd.DataFrame(x_test_poly, columns=feature_names)

        logger.info('Normalized scale poly regression')
        x_train_scaled = self.scaler.fit_transform(x_poly_train)
        x_test_scaled = self.scaler.transform(x_poly_test)

        self.train = pd.DataFrame(x_train_scaled, columns=feature_names)
        self.train['income'] = y_train.values

        self.test = pd.DataFrame(x_test_scaled, columns=feature_names)
        self.test['income'] = y_test.values

        logger.debug(f"Train: {self.train.shape}, Test: {self.test.shape}")

    def fit_model(self):
        logger.info('Starting to fit our regression')
        x_train = self.train.drop(columns={'income'})
        try:
            model = sm.OLS(self.train['income'], sm.add_constant(x_train))
            self.fit_regression = model.fit()
            logger.info("Success!")
        except Exception as e:
            logger.error(f"Fail to training: {e}")

        logger.info(f"Coefficients: {self.fit_regression.params}")

    def predict_model(self):
        logger.info('Predict Test Data')
        x_test = self.test.drop(columns={'income'})

        self.predict_values = self.fit_regression.predict(x_test)
        self.resid = self.test['income'] - self.predict_values

    def independence_of_errors(self):
        logger.info("Testing independence of residuals (Durbin-Watson Test)")
        dw_stat = durbin_watson(self.resid)
        logger.info(f"Durbin-Watson statistic: {dw_stat:.4f}")

        if 1.5 <= dw_stat <= 2.5:
            logger.debug("Residuals appear to be independent.")
        else:
            logger.warning("Residuals may be auto correlated — check model specification.")

class_regression = PolynomialRegression()
#class_regression.correlation()
#class_regression.test_multicollinearity()
#class_regression.detect_non_linear_relation()
class_regression.train_test_scaling()
class_regression.fit_model()
class_regression.predict_model()
#class_regression.independence_of_errors()

#class_regression.homoscedasticity()
#class_regression.normality_of_residuals()
#class_regression.evaluating_model()
#class_regression.plot_linear_regression()

exit()





# Normality of errors
# Homoscedasticity
# Mean of errors is zero
# Cost Function (como MSE ou RMSE)
# validação cruzada para achar o degree






















class teste:




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
            logger.warning("We will need to adjust our model")
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



