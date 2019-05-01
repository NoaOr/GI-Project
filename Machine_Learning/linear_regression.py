import os
import sklearn

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut, KFold
from scipy.stats import sem
import matplotlib.pyplot as plt

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

    print("mean absolute error: ", mean_absolute_error(y_test, predict))
    print("r2 error: ", sklearn.metrics.r2_score(y_test, predict))

    coefficients = [(d, c) for d, c in zip(feaures, linear_regression_model.coef_)]
    coefficients_str = ""
    for a, b in coefficients:
        coefficients_str += a + ": " + str("%.4f" % b) + ", "
    coefficients_str = coefficients_str[:-2]

    print(coefficients_str)

    # Plot outputs
    plt.figure(figsize=(17, 12))

    plt.scatter(X_test['Carbohydrt_(g)'], y_test, color='blue', s = 40)
    plt.scatter(X_test['Carbohydrt_(g)'], predict, color='red', s = 35)

    plt.xticks(())
    plt.yticks(())

    plt.legend(('GI vlaue', 'predict GI value'),
               shadow=True, loc=(0.75, 0.85), handlelength=1.5, fontsize=20)

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 30,
            }
    plt.title(pic_name, fontdict=font)
    plt.xlabel('Mean absolute Error = ' + str(mean_absolute_error(y_test, predict)) + '\n' +
                'R2 score = ' + str(sklearn.metrics.r2_score(y_test, predict)) +
               'coefficients: ' + '\n' +
               coefficients_str, fontsize=15)


    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig(pic_name + '.png')
    # plt.show()

