import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib

X_train = pd.read_csv('../data/train.csv', index_col=0)
y_train = X_train.pop('Cover_Type')

X_valid = pd.read_csv('../data/valid.csv', index_col=0)
y_valid = X_valid.pop('Cover_Type')


# k nearest neighbors classification
kn_param_grid = {'n_neighbors': [2, 3, 4, 5, 7, 10, 15, 20]}
kn_classifier = KNeighborsClassifier()

kn_grid_search = GridSearchCV(kn_classifier, param_grid=kn_param_grid, n_jobs=-1, scoring='accuracy', cv=10)
kn_grid_search.fit(X_train, y_train)
y_pred_kn = kn_grid_search.predict(X_valid)
print("Linear regression score: ", kn_grid_search.score(X_valid, y_valid))

joblib.dump(kn_grid_search, '../models/model_kn.save')


# random forest classification
rf_param_grid = {'n_estimators': [20, 50, 100, 200],
                 'criterion': ['gini', 'entropy'],
                 'max_depth': [2, 5, 10],
                 'min_samples_leaf': [2, 5, 10]}

rf_classifier = RandomForestClassifier(random_state=42)

rf_grid_search = GridSearchCV(rf_classifier, param_grid=rf_param_grid, n_jobs=-1, scoring='accuracy', cv=10)
rf_grid_search.fit(X_train, y_train)
print("Random forest score: ", rf_grid_search.score(X_valid, y_valid))

joblib.dump(rf_grid_search, '../models/model_rf.save')
