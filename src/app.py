# imports:
import joblib
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


# data loading:
url = 'https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
df_raw = pd.read_csv(url)

df_raw.to_csv('../data/raw/titanic_train.csv')
df_interim = df_raw.copy()


# Data cleaning & transforming

# drop unnecesary columns:
df_interim = df_interim.drop(columns=['Name', 'Ticket', 'Cabin'])

# 'Sex' to categorical: {'male':1, 'female':0}
df_interim['Sex'] = df_interim['Sex'].map({'male':1, 'female':0})
# 'Embarked' to categorical: {'S':2, 'C':1, 'Q':0}
df_interim['Embarked'] = df_interim['Embarked'].map({'S':2, 'C':1, 'Q':0})


df_interim.to_csv('../data/interim/titanic_train.csv')
df = df_interim.copy()


# train-test split:
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y , random_state=13, test_size=0.2)

# model & training
model_XGBC = XGBClassifier(random_state=13)
model_XGBC.fit(X_train, y_train)

# predictions
y_train_pred = model_XGBC.predict(X_train)
y_test_pred = model_XGBC.predict(X_test)

# Metrics & evaluation:
y_pred = model_XGBC.predict(X_test)
print(f'CLASSIFICATION REPORT: "Survived" \n {classification_report(y_test, y_pred)}')
# Get the score of train data just to verify its 1.
score = model_XGBC.score(X_train, y_train)
print(f'The score for XGBoost with X_train & y_trains is: {score}')
#Get the score for the predictions:
score = model_XGBC.score(X_test, y_test)
print(f'The score for XGBoost with X_test & y_test is: {score}')
# Tree params
print(f'Tree params: \n {model_XGBC.get_params()}')


# GridSearch:
params = {
    'learning_rate':[0.4, 0.045, 0.5],
    'max_depth':[3],
    'max_leaves':[1, 2]
}
tuning = GridSearchCV(estimator=XGBClassifier(random_state=13), param_grid=params)
tuning.fit(X_train, y_train)
print(f'Best Parameters: {tuning.best_params_}, Score: {tuning.best_score_}')
estimator=tuning.best_estimator_
print(f'Best Estimator:  {estimator}')

print(f'BEST HYPERPARAMETERS:')
print(f'learning_rate: {estimator.learning_rate}')
print(f'max_depth: {estimator.max_depth}')
print(f'max_leaves: {estimator.max_leaves}')

model_XGBC = XGBClassifier(
    learning_rate=estimator.learning_rate, max_depth=estimator.max_depth,
    max_leaves=estimator.max_leaves, random_state=13)
model_XGBC.fit(X_train, y_train)

y_train_pred = model_XGBC.predict(X_train)
y_test_pred = model_XGBC.predict(X_test)

y_pred = model_XGBC.predict(X_test)

print(f'CLASSIFICATION REPORT: "Survived" \n {classification_report(y_test, y_pred)}')
# Get the score of train data just to verify its 1.
score = model_XGBC.score(X_train, y_train)
print(f'The score for XGBoost with X_train & y_trains is: {score}')
#Get the score for the predictions:
score = model_XGBC.score(X_test, y_test)
print(f'The score for XGBoost with X_test & y_test is: {score}')
# Tree params
print(f'Tree params: \n {model_XGBC.get_params()}')


# Export model
joblib.dump(model_XGBC, '../models/XGBC_Titanic.pkl')