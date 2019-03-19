import pandas as pd
import os
import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, KFold
from scipy.stats import sem


def kf_cv(X_train, y_train, model):
    scores = np.zeros(X_train[:].shape[0])

    kf = KFold(n_splits=2, shuffle=False)  # Define the split - into 2 folds
    kf.get_n_splits(X_train)  # returns the number of splitting iterations in the cross-validator
    print(kf)
    KFold(n_splits=2, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X_train):
        X_train_cv, X_test_cv = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
        model = model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_test_cv)
        scores[test_index]= mean_absolute_error(y_test_cv.astype(int), y_pred.astype(int))
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))
    return model


def predict(X_train, X_test, y_train, y_test):
    linear_regression_model = linear_model.LinearRegression()

    cv_model = kf_cv(X_train, y_train, linear_regression_model)

    fit_model = linear_model.LinearRegression()
    fit_model.fit(X_train, y_train)
    # Predict
    cv_predict = cv_model.predict(X_test)
    fit_predict = fit_model.predict(X_test)

    print("final cv model error: ", mean_absolute_error(y_test, cv_predict))
    print("final fit model error: ", mean_absolute_error(y_test, fit_predict))

    print("final cv model score: ", cv_model.score(X_test, y_test))
    print("final fit model score: ", fit_model.score(X_test, y_test))

