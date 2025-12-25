# Goal: Classifying e-mail spam
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


model = LogisticRegression()
model.fit(min_on_site, purchase)
log_odds = model.intercept_ + model.coef_ * min_on_site
print(log_odds)
np.exp(log_odds)/(1+ np.exp(log_odds))
print(model.predict(features))
print(model.predict_proba(features)[:,1])

# We will need to normalization again
zscore = StandardScaler()
data = pd.DataFrame(zscore.fit_transform(self.skew_data), columns=self.skew_data.columns)

# Evaluating with confusion matrix
print(confusion_matrix(y_true, y_pred))  # Ver se tem como fazer um plot disso
accuracy_score, precision_score, precision_score, f1_score # Escolher qual usar para spam

# Olhar Assumptions
#1 - Independent observations
#2 - Large enough sample size - Entender como isso funciona
df = pd.read_csv('breast_cancer_data.csv')
#encode malignant as 1, benign as 0
df['diagnosis'] = df['diagnosis'].replace({'M':1,'B':0})
max_features = min(df.diagnosis.value_counts()/10)
print(max_features)

#3 - Features linearly related to log odds - Ver se é o mesmo usado em multiple regression, senão; To test this visually, we can use Seaborn’s regplot, with the parameter logistic= True and the x value as our feature of interest. If this condition is met, the fit model will resemble a sigmoidal curve (as is the case when x=radius_mean).

#4 - Multicollinearity - Eu tenho isso no multiple regression
