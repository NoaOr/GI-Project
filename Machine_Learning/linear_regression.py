import os
import sklearn

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut, KFold
from scipy.stats import sem
import matplotlib.pyplot as plt

from Machine_Learning import Plot_output


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


def predict(X_train, X_test, y_train, y_test, feaures, pic_name, dir):
    """
    The function predicts the tags of X_test by the linear regression model
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param labels:
    :param pic_name:
    :param dir:
    :return:
    """
    linear_regression_model = linear_model.LinearRegression()
    linear_regression_model.fit(X_train, y_train)

    # Predict
    predict = linear_regression_model.predict(X_test)

    print("mean absolute error: ", mean_absolute_error(y_test, predict))
    print("r2 error: ", sklearn.metrics.r2_score(y_test, predict))

    coefficients = [(d, c) for d, c in zip(feaures, linear_regression_model.coef_)]
    coefficients_str = ""
    for a, b in coefficients:
        coefficients_str += a + ": " + str("%.1f" % b) + ", "
    coefficients_str = coefficients_str[:-2]

    print(coefficients_str)

    Plot_output.plot_graph(X_test, y_test, predict, pic_name, dir, coefficients_str)
