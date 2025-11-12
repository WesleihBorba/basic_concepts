# Goal:
import statsmodels.api as sm

## USAR MAKE REGRESSION E ALGUNS CALCULOS PARA FAZER A LINHA DE REGRESSÃO COM 1 OU 2 VARIÁVEIS

##  Why We Need Interaction and Polynomial Terms: Relations of features
# Fit model2 regression here:
model2 = sm.OLS.from_formula('height ~ weight + species + weight:species', data=plants).fit() # Entender como funcionar o weight:species
# Print model2 results here:
print(model2.params)

modelP = sm.OLS.from_formula('happy ~ sleep + np.power(sleep,2)', data=happiness).fit()
print(modelP.params)

# PAREI AQUI: Interpreting Polynomial Terms

## Log Transformations (And More)

## E assumptions pegar em sites

## Criar da mesma forma que foi feito nos demais