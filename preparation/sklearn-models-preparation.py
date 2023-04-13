import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import joblib

X_train = pd.read_csv('../data/train.csv')
y_train = X_train.pop('Cover_Type')

X_valid = pd.read_csv('../data/valid.csv')
y_valid = X_valid.pop('Cover_Type')


lr_classifier = LogisticRegression(random_state=42, solver='saga', max_iter=6000)
lr_classifier.fit(X_train, y_train)
print("Linear regression score: ", lr_classifier.score(X_valid, y_valid))

joblib.dump(lr_classifier, '../models/model_lr.save')

rf_param_grid = {'n_estimators': [20, 50, 100, 200],
                 'criterion': ['gini', 'entropy'],
                 'max_depth': [2, 5, 10],
                 'min_samples_leaf': [2, 5, 10]}

rf_classifier = RandomForestClassifier(random_state=42)

rf_grid_search = GridSearchCV(rf_classifier, param_grid=rf_param_grid, n_jobs=-1, scoring='accuracy', cv=10)
rf_grid_search.fit(X_train, y_train)
print("Random forest score: ", rf_grid_search.score(X_valid, y_valid))

joblib.dump(rf_grid_search, '../models/model_rf.save')




