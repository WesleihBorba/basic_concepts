# Goal: Use Lasso and Ridge to reduce the coefficient of variables with little relevance.
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
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


class LassoRidgeModel:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.best_model_lasso, self.best_model_ridge = [None] * 2
        self.fitted_values_lasso, self.fitted_values_ridge = [None] * 2
        self.resid_lasso, self.resid_ridge = [None] * 2
        self.predict_values_lasso, self.predict_values_ridge = [None] * 2
        self.results_map = None

        X, y = make_regression(
            n_samples=15000,
            n_features=10,
            noise=15,
            random_state=42
        )

        self.variables = [
            "education", "working_time", "age", "hours_per_week",
            "industry_score", "tech_skills", "location_cost",
            "experience_years", "management_level", "certifications"
        ]

        self.data = pd.DataFrame(X, columns=self.variables)  # Values are not realistic

        self.data['income'] = y

    def linearity_assumption(self):
        logger.info("A single predictor variable need to have straight-line relationship with the dependent variable")
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for i, col in enumerate(self.variables):
            axes[i].scatter(self.data[col], self.data["income"], alpha=0.3, s=10)
            axes[i].set_xlabel(col.replace('_', ' ').title())
            axes[i].set_ylabel("Income")
            axes[i].set_title(f"{col.title()} vs Income")

        plt.tight_layout()
        plt.show()

    def correlation(self):
        logger.info("Look at the correlation between the dependent and independent variables")
        sns.heatmap(self.data.corr(numeric_only=True), annot=True, cmap="coolwarm")
        plt.show()

    def test_multicollinearity(self):
        logger.info("Test of Multicollinearity")
        X = self.data.drop(columns={'income'})
        new_X_test = sm.add_constant(X)
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
        X = self.data.drop(columns=['income'])
        y = self.data['income']
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        logger.info('Scaling our data')
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)

        logger.debug(f"Shapes - Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    def best_parameters(self):
        logger.info('Finding best parameters of our models')
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

        grid_lasso = GridSearchCV(Lasso(max_iter=10000), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_lasso.fit(self.X_train, self.y_train)
        self.best_model_lasso = grid_lasso.best_estimator_

        grid_ridge = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_ridge.fit(self.X_train, self.y_train)
        self.best_model_ridge = grid_ridge.best_estimator_

        logger.debug(f"Best Alpha Lasso: {grid_lasso.best_params_['alpha']}")
        logger.debug(f"Best Alpha Ridge: {grid_ridge.best_params_['alpha']}")

    def generate_residuals(self):
        logger.info('Save residues of each model')
        self.results_map = {
            'Lasso': {
                'fitted': self.best_model_lasso.predict(self.X_train),
                'resid': self.y_train - self.best_model_lasso.predict(self.X_train),
                'predict': self.best_model_lasso.predict(self.X_test)
            },
            'Ridge': {
                'fitted': self.best_model_ridge.predict(self.X_train),
                'resid': self.y_train - self.best_model_ridge.predict(self.X_train),
                'predict': self.best_model_ridge.predict(self.X_test)
            }
        }

    def homoscedasticity(self):
        logger.info("Homoscedasticity assumption")

        exog = sm.add_constant(self.X_train)
        labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']

        for model_name, data in self.results_map.items():
            resid = data['resid']
            fitted = data['fitted']

            bp_test = het_breuschpagan(resid, exog)
            result = dict(zip(labels, bp_test))

            plt.figure(figsize=(8, 4))
            plt.scatter(fitted, resid, alpha=0.3, s=10)
            plt.axhline(0, linestyle='--', color='red')
            plt.xlabel('Fitted values')
            plt.ylabel('Residuals')
            plt.title(f'Residuals vs Fitted Values - {model_name}')
            plt.show()

            if result['LM-Test p-value'] >= 0.05:
                logger.info(f"[{model_name}] Homoscedasticity confirmed (p={result['LM-Test p-value']:.4f})")
            else:
                logger.warning(f"[{model_name}] Heteroscedasticity detected (p={result['LM-Test p-value']:.4f})")

    def normality_of_residuals(self):
        logger.info("Resid of our model need to follow a normal distribution")

        for model_name, data in self.results_map.items():
            resid = data['resid']
            sm.qqplot(resid, line='r')
            plt.title(f'Q-Q Plot for Normality Test - {model_name}')
            plt.xlabel('Theoretical Quantiles (Normal Distribution)')
            plt.ylabel('Observed Quantiles (Data Sample)')
            plt.show()

    def independence_of_errors(self):
        logger.info("Testing independence of residuals (Durbin-Watson Test)")

        for model_name, data in self.results_map.items():
            resid = data['resid']
            dw_stat = durbin_watson(resid)
            logger.info(f"Durbin-Watson statistic: {dw_stat:.4f} - {model_name}")

            if 1.5 <= dw_stat <= 2.5:
                logger.debug("Residuals appear to be independent.")
            else:
                logger.warning("Residuals may be auto correlated — check model specification.")

    def evaluating_model(self):
        logger.info("Looking if our model is good to use")
        for model_name, data in self.results_map.items():
            predict_data = data['predict']
            mse = mean_squared_error(self.y_test, predict_data)
            r2 = r2_score(self.y_test, predict_data)

            logger.info(f'Mean Squared Error of {model_name}: {mse}')
            logger.info(f'R-squared of {model_name}: {r2}')


class_regression = LassoRidgeModel()
class_regression.linearity_assumption()
class_regression.correlation()
class_regression.test_multicollinearity()
class_regression.train_test()
class_regression.best_parameters()
class_regression.generate_residuals()
class_regression.homoscedasticity()
class_regression.normality_of_residuals()
class_regression.independence_of_errors()
class_regression.evaluating_model()