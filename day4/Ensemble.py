import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#load dataset
from sklearn.datasets import load_iris
data = load_iris()
x = data.data
y = data.target

#split datasets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#1. Bagging - parallel
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("Random Forest Accuracy : ", accuracy_score(y_test, rf_preds))

#2. Boosting - sequential, learn from mistakes
#types - ada, gradient, XG+

#adaBoosting
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_model.fit(X_train, y_train)
ada_preds = ada_model.predict(X_test)
print('AdaBoost Accuracy : ', accuracy_score(y_test, ada_preds))

#GradientBoosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)
print('Gradient Boosting Accuracy : ', accuracy_score(y_test, gb_preds))

#XGBoosting
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
print('XGBoost Accuracy : ', accuracy_score(y_test, xgb_preds))

#3. Stacking - considers results
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42))
]
meta_model = LogisticRegression()
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
stacking_model.fit(X_train, y_train)
stacking_preds = stacking_model.predict(X_test)
print('Stacking Accuracy : ', accuracy_score(y_test, stacking_preds))
