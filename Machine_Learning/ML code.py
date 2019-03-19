from sklearn import tree
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn import metrics
import numpy as np

from Machine_Learning import decision_tree, linear_regression




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
                     'GmWt_1', 'GmWt_2', 'Panto_Acid_mg)', 'Choline_Tot_ (mg)'], axis='columns')

    median_df = ml_df.median(skipna=True, numeric_only=True)
    for column in ml_df:
        ml_df[column] = ml_df[column].fillna(median_df[column])

    X = ml_df.drop('GI Value', axis=1, inplace=False)
    y = ml_df['GI Value'].values
    labels = list(ml_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
    print("Decision tree model:\n")
    decision_tree.predict(X_train, X_test, y_train, y_test, labels)
    print("\n\nLinear regression model:\n")
    #linear_regression.predict(X_train, X_test, y_train, y_test)
