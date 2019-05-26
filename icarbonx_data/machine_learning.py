import sklearn
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.metrics import mean_absolute_error
from Machine_Learning import decision_tree, linear_regression,\
    elastic_net, random_forest, Plot_output, improve_ml_code


def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):
    """
    Measure performance function.
    """

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


def split_to_train_test(df, with_food_groups=0):
    """
    This function splits the data to train and test by euclidean distance
    :param df: The data
    :param with_food_groups: flag
    :return: X_train, X_test, y_train, y_test
    """

    X = df.drop('GI Value', axis=1, inplace=False)
    y = df['GI Value'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=33)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    euclidean_df = pd.read_excel("Euclidean_distance.xlsx")
    x_test_size = X_test.shape[0]
    for i in range(x_test_size):
        food_name = X_test.iloc[i]['Food Description in 1994-96 CSFII']
        for j in range(X_train.shape[0]):
            if j >= X_train.shape[0]:
                break

            compared_food = X_train.iloc[j]['Food Description in 1994-96 CSFII']
            row_index = euclidean_df.columns.get_loc(food_name)
            if euclidean_df.iloc[row_index][compared_food] < 2:
                index = X_train.index[X_train['Food Description in 1994-96 CSFII'] == compared_food].tolist()[0]
                b = X_train.loc[X_train['Food Description in 1994-96 CSFII'] == compared_food]
                X_test = X_test.append(b, ignore_index=True)
                y_test = np.append(y_test, y_train[index])
                X_train = X_train[X_train['Food Description in 1994-96 CSFII'] != compared_food]
                X_train.reset_index(drop=True, inplace=True)
                y_train = np.delete(y_train, index)

    X_train = X_train.drop(['Food Description in 1994-96 CSFII'], axis='columns')
    X_test = X_test.drop(['Food Description in 1994-96 CSFII'], axis='columns')

    if not with_food_groups:
        X_train = X_train.drop(['FdGrp_desc'], axis='columns')
        X_test = X_test.drop(['FdGrp_desc'], axis='columns')
    else:
        food_groups = {'Dairy and Egg Products' : 1,
                        'Spices and Herbs' : 2,
                        'Baby Foods' : 3,
                        'Fats and Oils' : 4,
                        'Poultry Products' : 5,
                        'Soups, Sauces, and Gravies' : 6,
                        'Sausages and Luncheon Meats' : 7,
                        'Breakfast Cereals' : 8,
                        'Fruits and Fruit Juices' : 9,
                        'Pork Products' : 10,
                        'Vegetables and Vegetable Products' : 11,
                        'Nut and Seed Products' : 12,
                        'Beef Products' :13,
                        'Beverages': 14,
                        'Finfish and Shellfish Products' : 15,
                        'Legumes and Legume Products' : 16,
                        'Lamb, Veal, and Game Products' : 17,
                        'Baked Products' : 18,
                        'Sweets' : 19,
                        'Cereal Grains and Pasta' : 20,
                        'Fast Foods' : 21,
                        'Meals, Entrees, and Side Dishes' : 22,
                        'Snacks' : 23,
                        'American Indian/Alaska Native Foods' : 24,
                        'Restaurant Foods' : 25 }
        X_train.FdGrp_desc = [food_groups[item] for item in X_train.FdGrp_desc]
        X_test.FdGrp_desc = [food_groups[item] for item in X_test.FdGrp_desc]

    return X_train, X_test, y_train, y_test


def get_train_and_test(df):
    """
    Return split to train test in the regular way
    :param df: The data
    :return: X_train, X_test, y_train, y_test
    """
    X = df.drop('2h-iAUC', axis=1, inplace=False)
    y = df['2h-iAUC'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
    return X_train, X_test, y_train, y_test


def get_df_by_features(df, features):
    """
    df by features
    :param df:
    :param features:
    :return: df by features

    """
    features.append('GI Value')
    new_df = df[features]

    return new_df


def linear_regression_by_features(ml_df, features, pic_name, dir):
    """
    this function performs linear regression learning by certain features.
    :param ml_df: the data
    :param features: what features to work on
    :param pic_name:
    :return:
    """
    filter_df = get_df_by_features(ml_df, features)
    X_train, X_test, y_train, y_test = get_train_and_test(filter_df)
    linear_regression.predict(X_train, X_test, y_train, y_test, features, pic_name, dir)


def learn(ml_df, pic_name="", dir=""):
    """
    THe function creates train and test and
     calls the learning models to predict
    :param ml_df:
    :param pic_name:
    :return:
    """
    # X_train, X_test, y_train, y_test = split_to_train_test(ml_df)
    # RF_X_train, RF_X_test, RF_y_train, RF_y_test = split_to_train_test(ml_df, with_food_groups=1)


    ##########################################################
    # test on train
    ##########################################################

    # X_test = X_train
    # y_test = y_train
    # RF_X_test = RF_X_train
    # RF_y_test = RF_y_train

    ##########################################################
    # separate train and test randomly
    ##########################################################

    ml_df = ml_df.replace([-np.inf], 0).dropna(axis=1)

    RF_X_train, RF_X_test, y_train, y_test = get_train_and_test(ml_df)
    RF_X_train = RF_X_train.drop(['food_names'], axis='columns')
    RF_X_test = RF_X_test.drop(['food_names'], axis='columns')
    # X_train = X_train.drop(['FdGrp_desc'], axis='columns')
    # X_test = X_test.drop(['FdGrp_desc'], axis='columns')

    X_train = RF_X_train
    X_test = RF_X_test

    features = list(ml_df.columns.values)
    features.remove('2h-iAUC')
    features.remove('food_names')
    # features.remove('FdGrp_desc')

    # Plot_output.plot_two_cols(x='Carbohydrt_(g)', y='GI Value', df=ml_df, pic_name="carbo_vs_gi" + pic_name)

    pic_name = "_" + pic_name

    ##########################################################
    # naive model
    ##########################################################

    # print("Naive model:\n")
    # median = ml_df["GI Value"].mean()
    # predict = [median] * len(X_test)
    # Plot_output.plot_graph(X_test, y_test, predict, "naive_model", dir)


    ##########################################################
    # decision tree
    ##########################################################



    # print("\n\nDecision tree model:\n")
    # decision_tree.predict(X_train, X_test, y_train, y_test, features, 'Decision_tree' + pic_name, dir)


    ##########################################################
    # linear regression
    ##########################################################

    # print("\n\nLinear regression model:\n")
    # linear_regression_by_features(ml_df, ['Carbohydrt_(g)'], 'LR_carbo' + pic_name, dir)
    # linear_regression_by_features(ml_df,['Carbohydrt_(g)', 'Sugar_Tot_(g)'], 'LR_carbo_sugar' + pic_name, dir)
    # linear_regression_by_features(ml_df,['Carbohydrt_(g)', 'Lipid_Tot_(g)'], 'LR_carbo_lipid'+ pic_name, dir)
    # linear_regression_by_features(ml_df,['Carbohydrt_(g)', 'Lipid_Tot_(g)','Protein_(g)', 'Fiber_TD_(g)', 'Sugar_Tot_(g)'],
    #                               'LR_carbo_lipid_pro_fibe_sug'+ pic_name, dir)


    # ##########################################################
    # # elastic net
    # ##########################################################

    # print("\n\nElastic net model:\n")
    # print("features: ", list(ml_df.columns.values))
    # elastic_net.predict(X_train, X_test, y_train, y_test, features, "Elastic_net" + pic_name, dir)

    ##########################################################
    # random forest
    ##########################################################

    print("\n\nRandom Forest model:\n")
    # features.append(('FdGrp_desc'))
    random_forest.predict(RF_X_train, RF_X_test, y_train, y_test, features,
                          'RF_variable_importance' + pic_name, 'Random_Forest'+ pic_name, dir)


if __name__ == '__main__':
    if not os.getcwd().__contains__("Excel_files"):
        os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")

    ml_df = pd.read_excel("GI_USDA_IMPROVED.xlsx")
    os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Excel_files/train&test")

    X_train = pd.read_excel("X_train.xlsx")
    X_test = pd.read_excel("X_test.xlsx")
    y_train = pd.read_excel("y_train.xlsx").as_matrix()
    y_train = np.ravel(y_train)
    y_test = pd.read_excel("y_test.xlsx").as_matrix()
    y_test = np.ravel(y_test)
    RF_X_train = pd.read_excel("RF_X_train.xlsx")
    RF_X_test = pd.read_excel("RF_X_test.xlsx")

    learn(ml_df, X_train, X_test, y_train, y_test, RF_X_train, RF_X_test, pic_name='best_model', dir="best_model")



