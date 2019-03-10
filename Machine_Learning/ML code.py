from sklearn import tree
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):
    y_pred = clf.predict(X)
    y_pred = np.hstack(y_pred)
    y = y.ravel()
    if show_accuracy:
        acc = metrics.accuracy_score(y, y_pred)
        print(acc)
       # print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)), "\n")

    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y, y_pred), "\n")

    if show_confusion_matrix:
        print("Confusion matrix")
        print(metrics.confusion_matrix(y, y_pred), "\n")


if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    df = pd.read_excel("GI_USDA_clean.xlsx")

    ml_df = df.drop(['CSFII 1994-96 Food Code', 'Food Description in 1994-96 CSFII',
                     'source table', 'reference food & time period', 'serve Size g',
                     'available cerbo hydrate', 'GL per serve', 'GI_2', 'acc', 'match-sent',
                     'GmWt_Desc2', 'GmWt_Desc1', 'Manganese_(mg)',
                     'GmWt_1', 'GmWt_2', 'Panto_Acid_mg)'], axis='columns')

    ml_df = ml_df.fillna(0)

    X = ml_df.drop('GI Value', axis=1, inplace=False)
    y = ml_df['GI Value']

    model = LinearRegression()
    scores = []
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    z = kfold.split(X, y)
    for i, (train, test) in enumerate(kfold.split(X, y)):
        a = X.iloc[train, :]
        b = y.iloc[train, :]
        model.fit(X.iloc[train, :], y.iloc[train, :])
        score = model.score(X.iloc[test, :], y.iloc[test, :])
        scores.append(score)
    print(scores)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=33)
    # clf = tree.DecisionTreeRegressor()
    # clf = clf.fit(X_train, y_train)
    #
    # measure_performance(X_train, y_train, clf, show_classification_report=False, show_confusion_matrix=False)
    #



