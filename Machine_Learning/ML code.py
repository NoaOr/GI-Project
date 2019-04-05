from sklearn import tree
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn import metrics
import numpy as np

from Machine_Learning import decision_tree, linear_regression, elastic_net, random_forest




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


def split_to_train_test(df):
    X = df.drop('GI Value', axis=1, inplace=False)
    y = df['GI Value'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=33)

    euclidean_df = pd.read_excel("Euclidean_distance.xlsx")

    for i in range(X_test.shape[0]):
        food_name = X_test.iloc[i]['Food Description in 1994-96 CSFII']
        for j in range(X_train.shape[0]):
            compared_food = X_train.iloc[j]['Food Description in 1994-96 CSFII']
            row_index = euclidean_df.columns.get_loc(food_name)
            if euclidean_df.iloc[row_index][compared_food] < 10:
                print("x_train:", X_train.shape[0])
                print("x_test:", X_test.shape[0])
                print("y_train:", y_train.shape[0])
                print("y_test:", y_test.shape[0])

                index = X_train.index[X_train['Food Description in 1994-96 CSFII'] == compared_food].tolist()[0]
                X_test.loc[X_test.shape[0]] = X_train['Food Description in 1994-96 CSFII'] == compared_food
                a = y_train[index]
                b = np.array(a)

                # todo: add and delete from the Y arrays

                # np.append(y_test, y_train[index], 0)
                # y_test.append(y_train[index])
                X_train = X_train[X_train['Food Description in 1994-96 CSFII'] != compared_food]
                # np.delete(y_train, index, 0)
                # y_train.delete(y_train[index])

                print("x_train:", X_train.shape[0])
                print("x_test:", X_test.shape[0])
                print("y_train:", y_train.shape[0])
                print("y_test:", y_test.shape[0])


def get_train_and_test(df):
    X = df.drop('GI Value', axis=1, inplace=False)
    y = df['GI Value'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
    return X_train, X_test, y_train, y_test

def get_df_by_features(df, features):
    features.append('GI Value')
    new_df = df[features]

    #writer = pd.ExcelWriter('temp.xlsx', engine='xlsxwriter')
    #new_df.to_excel(writer, sheet_name='Sheet1')
    #writer.save()

    return new_df

def linear_regression_by_features(features, pic_name):
    filter_df = get_df_by_features(ml_df, features)
    X_train, X_test, y_train, y_test = get_train_and_test(filter_df)
    linear_regression.predict(X_train, X_test, y_train, y_test, features, pic_name)

if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    df = pd.read_excel("GI_USDA_clean.xlsx")

    ml_df = df.drop(['CSFII 1994-96 Food Code',
                     'source table', 'NDB_No', 'reference food & time period', 'serve Size g',
                     'available cerbo hydrate', 'GL per serve', 'GI_2', 'acc', 'match-sent',
                     'GmWt_Desc2', 'GmWt_Desc1', 'Manganese_(mg)',
                     'GmWt_1', 'GmWt_2', 'Panto_Acid_mg)', 'Choline_Tot_ (mg)'], axis='columns')

    median_df = ml_df.median(skipna=True, numeric_only=True)
    for column in ml_df:
        if column == "Food Description in 1994-96 CSFII":
            continue
        ml_df[column] = ml_df[column].fillna(median_df[column])



    split_to_train_test(ml_df)

    X_train, X_test, y_train, y_test = get_train_and_test(ml_df)

    ##########################################################
    # decision tree
    ##########################################################

    # print("Decision tree model:\n")
    # labels = list(ml_df)
    # decision_tree.predict(X_train, X_test, y_train, y_test, labels)

    ##########################################################
    # linear regression
    ##########################################################

    # print("\n\nLinear regression model:\n")
    # linear_regression_by_features(['Carbohydrt_(g)'], 'LR_carbo')
    # linear_regression_by_features(['Carbohydrt_(g)', 'Protein_(g)', 'Fiber_TD_(g)', 'Sugar_Tot_(g)'], 'LR_carbo_pro_fibe_sug')

    ##########################################################
    # elastic net
    ##########################################################

    print("\n\nElastic net model:\n")
    print("features: ", list(ml_df.columns.values))
    features = list(ml_df.columns.values)
    features.remove('GI Value')
    elastic_net.predict(X_train, X_test, y_train, y_test, features, "Elastic_net")

    ##########################################################
    # random forest
    ##########################################################


    print("\n\nRandom Forest model:\n")
    features = list(ml_df.columns.values)
    features.remove('GI Value')
    random_forest.predict(X_train, X_test, y_train, y_test, features, 'RF_variable_importance', 'Random_Forest')










