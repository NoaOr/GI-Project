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
import matplotlib.pyplot as plt
import sys

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


def predict(X_train, X_test, y_train, y_test, feaures, pic_name):
    linear_regression_model = linear_model.LinearRegression()
    linear_regression_model.fit(X_train, y_train)

    # Predict
    predict = linear_regression_model.predict(X_test)

    print("final model error: ", mean_absolute_error(y_test, predict))
    print("final model score: ", linear_regression_model.score(X_test, y_test))

    coefficients = [(d, c) for d, c in zip(feaures, linear_regression_model.coef_)]
    # Plot outputs

    coefficients_str = ""
    for a, b in coefficients:
        coefficients_str += a + ": " + str(b) + ", "
    coefficients_str = coefficients_str[:-2]

    print(coefficients_str)

    plt.scatter(X_test['Carbohydrt_(g)'], y_test, color='blue', s = 15)
    plt.scatter(X_test['Carbohydrt_(g)'], predict, color='red', s = 10)

    plt.xticks(())
    plt.yticks(())

    plt.legend(('GI vlaue', 'predict GI value'),
               shadow=True, loc=(0.67, 0.85), handlelength=1.5, fontsize=10)

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 16,
            }
    plt.title(pic_name, fontdict=font)
    plt.xlabel('Model Error = ' + str(mean_absolute_error(y_test, predict)) + '\n' +
               "coefficients: " + '\n' +
               coefficients_str, fontsize = 5)


    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig(pic_name + '.png')
    plt.show()

