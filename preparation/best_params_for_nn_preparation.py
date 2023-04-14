import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

np.random.seed(42)


def create_model(lr=0.001):
    model = Sequential()
    model.add(Dense(54, activation='relu', input_dim=54))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr),
                  metrics=['accuracy'])

    return model


def find_best_params():
    X_train = pd.read_csv('../data/train.csv', index_col=0)
    y_train = X_train.pop('Cover_Type')

    nn_param_grid = {
        'lr': [0.001, 0.0001],
        'batch_size': [100, 200, 400],
        'epochs': [20, 50, 100]
    }
    classifier_for_cv = KerasClassifier(build_fn=create_model, verbose=1)

    grid = GridSearchCV(estimator=classifier_for_cv,
                        param_grid=nn_param_grid,
                        cv=3)

    grid_result = grid.fit(X_train, y_train)
    return grid_result.best_params_


