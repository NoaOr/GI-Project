import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from Machine_Learning import improve_ml_code


def split_to_train_test():
    """
    This function splits the data to train and test by euclidean distance
    :param df: The data
    :param with_food_groups: flag
    :return: X_train, X_test, y_train, y_test
    """
    improve_ml_code.imporve()

    if not os.getcwd().__contains__("Excel_files"):
        os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    df = pd.read_excel("GI_USDA_IMPROVED.xlsx")


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

    regular_X_train = X_train.drop(['FdGrp_desc'], axis='columns')
    regular_X_test = X_test.drop(['FdGrp_desc'], axis='columns')

    food_groups = {'Dairy and Egg Products': 1,
                   'Spices and Herbs': 2,
                   'Baby Foods': 3,
                   'Fats and Oils': 4,
                   'Poultry Products': 5,
                   'Soups, Sauces, and Gravies': 6,
                   'Sausages and Luncheon Meats': 7,
                   'Breakfast Cereals': 8,
                   'Fruits and Fruit Juices': 9,
                   'Pork Products': 10,
                   'Vegetables and Vegetable Products': 11,
                   'Nut and Seed Products': 12,
                   'Beef Products': 13,
                   'Beverages': 14,
                   'Finfish and Shellfish Products': 15,
                   'Legumes and Legume Products': 16,
                   'Lamb, Veal, and Game Products': 17,
                   'Baked Products': 18,
                   'Sweets': 19,
                   'Cereal Grains and Pasta': 20,
                   'Fast Foods': 21,
                   'Meals, Entrees, and Side Dishes': 22,
                   'Snacks': 23,
                   'American Indian/Alaska Native Foods': 24,
                   'Restaurant Foods': 25}

    RF_X_train = X_train.copy()
    RF_X_test = X_test.copy()
    RF_X_train.FdGrp_desc = [food_groups[item] for item in X_train.FdGrp_desc]
    RF_X_test.FdGrp_desc = [food_groups[item] for item in X_test.FdGrp_desc]
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Excel_files/train&test")
    write_to_file(regular_X_train, 'X_train.xlsx')
    write_to_file(regular_X_test, 'X_test.xlsx')
    write_to_file(RF_X_train, 'RF_X_train.xlsx')
    write_to_file(RF_X_test, 'RF_X_test.xlsx')
    write_to_file(y_train, 'y_train.xlsx')
    write_to_file(y_test, 'y_test.xlsx')



def write_to_file(df, file_name):
    """
    THe function writes df to file
    :param df:
    :param file_name:
    :return:
    """
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

if __name__ == '__main__':
    split_to_train_test()
