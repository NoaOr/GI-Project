import os
import numpy as np
import pandas as pd
from sklearn.svm.libsvm import cross_validation
#from cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder




if __name__ == '__main__':
    print (os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    balance_data = pd.read_excel("GI_USDA_example.xlsx")

    print ("Dataset Lenght:: ", len(balance_data))
    print ("Dataset Shape:: ", balance_data.shape)

    X = balance_data.values[:, :]
    Y = balance_data.values[:, 2]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100,
                                      max_depth=3, min_samples_leaf=5)

    #print(X_train, y_train)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X_train)
    #enc.fit(y_train)
    x_train_2 = enc.transform(X_train).toarray()
    y_train_2 = enc.transform(y_train)

    clf_gini.fit(x_train_2, y_train_2)

    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                         max_depth=3, min_samples_leaf=5)
    clf_entropy.fit(X_train, y_train)

    clf_gini.predict([[4, 4, 3, 3]])

    y_pred = clf_gini.predict(X_test)
    y_pred

    y_pred_en = clf_entropy.predict(X_test)
    y_pred_en

    print ("Accuracy is ", accuracy_score(y_test, y_pred) * 100)
    print ("Accuracy is ", accuracy_score(y_test, y_pred_en) * 100)