# Goal:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import patsy

# Introduction to Linear Regression with sklearn and gradient
students = pd.read_csv('test_data.csv')
model = sm.OLS.from_formula('score ~ hours_studied', data = students)
results = model.fit()
print(results.params)

fitted_values = results.predict(students)

residuals = students.score - fitted_values # Criar regra (def) para ver se a assumption residuals passa como um modelo bom usando também try and excpt

# Fazer a mesma coisa normality and homoscedasticity - TRY AND EXCEPT - TENHO NO NOTE DO TRABALHO
plt.hist(residuals)
plt.show()

plt.scatter(fitted_values, residuals)
plt.show()




# Linear Regression with a Categorical Predictor

y, X = patsy.dmatrices('rent ~ borough', rentals)  #  SE EU CONSIGO USAR ISSO




# PRÓXIMO:
# Matrix Representation of Linear Regression: Understand the math and new libraries
