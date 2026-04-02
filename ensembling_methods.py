# Goal:
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier

bag_dt_10 = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=10, max_features=10)
bag_dt_10.fit(x_train, y_train)

# 2. Create an Adaptive Boost Classifier and print its parameters
ada_classifier = AdaBoostClassifier(base_estimator=decision_stump, n_estimators=5) 