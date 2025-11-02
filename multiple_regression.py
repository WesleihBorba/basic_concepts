# Goal:
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import pandas as pd


# LER 7.6 ATÉ 7.6.9

# PAREI AQUI: Interpretation of Quantitative Variables

model = sm.OLS.from_formula('score ~ hours_studied + breakfast', data=survey).fit()


# VER PARA QUE SERVE O sns.lmplot

# Simpson's Paradox: Procurar no chatgpt

# Ver a difrencça de variance inflation factor para multicolinearity



# ASSMPTIONS: linearity, independence of errors, homoscedasticity (constant variance of errors), normality of errors, and no multicollinearity, CORRELATION BETWEEN VARIABLES

# TESTAR O ASSUMPTION DOS SIMPSONS

# MULTICOLINEARITY
class TestOfMulticolinearity():
    def regression(X):
        new_X_test = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["Variável"] = new_X_test.columns
        vif_data["VIF"] = [variance_inflation_factor(new_X_test.values, i) for i in range(new_X_test.shape[1])]
        print(vif_data)
