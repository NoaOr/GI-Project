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


def split_to_train_test(df, with_food_groups=0):
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

    print("x_train:", X_train.shape[0])
    print("x_test:", X_test.shape[0])
    print("y_train:", y_train.shape[0])
    print("y_test:", y_test.shape[0])
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
    df = pd.read_excel("GI_USDA_CLEAN_FOOD_GROUPS.xlsx")

    ml_df = df.drop(['CSFII 1994-96 Food Code',
                     'source table', 'NDB_No', 'reference food & time period', 'serve Size g',
                     'available cerbo hydrate', 'GL per serve', 'GI_2', 'acc', 'match-sent',
                     'GmWt_Desc2', 'GmWt_Desc1', 'Manganese_(mg)',
                     'GmWt_1', 'GmWt_2', 'Panto_Acid_mg)', 'Choline_Tot_ (mg)'], axis='columns')

    median_df = ml_df.median(skipna=True, numeric_only=True)
    for column in ml_df:
        if column == "Food Description in 1994-96 CSFII" or column == "FdGrp_desc":
            continue
        ml_df[column] = ml_df[column].fillna(median_df[column])



    X_train, X_test, y_train, y_test = split_to_train_test(ml_df)
    RF_X_train, RF_X_test, RF_y_train, RF_y_test = split_to_train_test(ml_df, with_food_groups=1)

    # X_train, X_test, y_train, y_test = get_train_and_test(ml_df)
    # X_train = X_train.drop(['Food Description in 1994-96 CSFII'], axis='columns')
    # X_test = X_test.drop(['Food Description in 1994-96 CSFII'], axis='columns')
    # X_train = X_train.drop(['FdGrp_desc'], axis='columns')
    # X_test = X_test.drop(['FdGrp_desc'], axis='columns')

    features = list(ml_df.columns.values)
    features.remove('GI Value')
    features.remove('Food Description in 1994-96 CSFII')
    features.remove('FdGrp_desc')
    ##########################################################
    # decision tree
    ##########################################################

    # print("Decision tree model:\n")
    # decision_tree.predict(X_train, X_test, y_train, y_test, features, 'Decision_tree_new_test_2')

    ##########################################################
    # linear regression
    ##########################################################

    # print("\n\nLinear regression model:\n")
    # linear_regression_by_features(['Carbohydrt_(g)'], 'LR_carbo_new_test_2')
    # linear_regression_by_features(['Carbohydrt_(g)', 'Lipid_Tot_(g)'], 'LR_carbo_lipid_new_test_2')
    # linear_regression_by_features(['Carbohydrt_(g)', 'Lipid_Tot_(g)','Protein_(g)', 'Fiber_TD_(g)', 'Sugar_Tot_(g)'],
    #                               'LR_carbo_lipid_pro_fibe_sug_new_test_2')

    # ##########################################################
    # # elastic net
    # ##########################################################

    print("\n\nElastic net model:\n")
    print("features: ", list(ml_df.columns.values))
    elastic_net.predict(X_train, X_test, y_train, y_test, features, "Elastic_net_new_test_2")

    ##########################################################
    # random forest
    ##########################################################

    # print("\n\nRandom Forest model:\n")
    # features.append(('FdGrp_desc'))
    # random_forest.predict(RF_X_train, RF_X_test, RF_y_train, RF_y_test, features,
    #                       'RF_variable_importance_new_test_fg', 'Random_Forest_new_test_fg_2')





