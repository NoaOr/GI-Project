import pandas as pd
import numpy as np
import sys
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
    # improve_ml_code.imporve()
    df = pd.read_excel("final_dataset_with_median.xlsx")
    df = df.replace([-np.inf], 0).dropna(axis=1)


    X = df.drop('2h-iAUC', axis=1, inplace=False)
    X = X.drop('slope_of_decline', axis=1, inplace=False)
    X = X.drop('slope_of_incline', axis=1, inplace=False)

    y = df['2h-iAUC'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=33)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    euclidean_df = pd.read_excel("Euclidean_distance_icarbonx.xlsx")

    print("X_train: ", X_train.shape)
    print("X_test: ", X_test.shape)
    print("y_train: ", y_train.shape)
    print("y_test: ", y_test.shape)
    x_test_size = X_test.shape[0]
    # print("x_test_size: ", x_test_size)
    for i in range(x_test_size):
        print("________________________________________________")
        print("x_test_row: ", i)
        food_name = X_test.iloc[i]['food_names']
        for j in range(X_train.shape[0]):
            # print("x_train_row: ", j)

            if j >= X_train.shape[0]:
                print(j)
                break

            compared_food = X_train.iloc[j]['food_names']
            row_index = euclidean_df.columns.get_loc(food_name)
            if euclidean_df.iloc[row_index][compared_food] < 13:

                index = X_train.index[X_train['food_names'] == compared_food].tolist()[0]
                b = X_train.loc[X_train['food_names'] == compared_food]
                X_test = X_test.append(b, ignore_index=True)
                y_test = np.append(y_test, y_train[index])
                X_train = X_train[X_train['food_names'] != compared_food]
                X_train.reset_index(drop=True, inplace=True)
                y_train = np.delete(y_train, index)

    X_train = X_train.drop(['food_names'], axis='columns')
    X_test = X_test.drop(['food_names'], axis='columns')


    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    print("X_train: ", X_train.shape)
    print("X_test: ", X_test.shape)
    print("y_train: ", y_train.shape)
    print("y_test: ", y_test.shape)

    os.chdir(os.getcwd()[:os.getcwd().index("icarbonx_data")] + "icarbonx_data/train&test")

    write_to_file(X_train, 'X_train.xlsx')
    write_to_file(X_test, 'X_test.xlsx')
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



def insert_carbo_ratio(df, origin_column, ratio_column):
    """
    This function adds features of ratio with crabos to the df
    :param df:
    :param origin_column:
    :param ratio_column:
    :return: new df
    """
    df[ratio_column] = ""
    for index, row in df.iterrows():
        carbo_val = df.at[index, 'total_carbohydrate']
        col_val = df.at[index, origin_column]
        denominator = col_val + carbo_val
        df.loc[index, ratio_column] = round(carbo_val / denominator, 3)

    # df.fillna(sys.float_info.epsilon, inplace=True)
    df.fillna(0, inplace=True)
    return df


def insert_main_ftrs_ratio(df, main_col, column_1, column_2,
                           column_3, new_column):
    """
    This function adds features of ratio between main features
    and important columns.
    :param df:
    :param main_col:
    :param column_1:
    :param column_2:
    :param column_3:
    :param column_4:
    :param new_column:
    :return: new df
    """
    df[new_column] = ""
    for index, row in df.iterrows():
        numerator = df.at[index, main_col]
        denominator = df.at[index, column_1] + df.at[index, column_2] + \
                      df.at[index, column_3]
        if denominator == 0:
            denominator = sys.float_info.epsilon
        df.loc[index, new_column] = round(numerator / denominator, 3)
    return df


def add_features_to_df(origin_df):
    """
    add features to df
    :param origin_df:
    :return: new df
    """
    new_df = origin_df.copy()

    new_df = insert_carbo_ratio(new_df, 'protein', 'carbo-protein')
    new_df = insert_carbo_ratio(new_df, 'total_fat', 'carbo-lipid')
    new_df = insert_carbo_ratio(new_df, 'dietary_fiber', 'carbo-fiber_(availableCarbo)')

    # new_df = insert_main_ftrs_ratio(new_df, main_col='Water_(g)', column_1='total_carbohydrate',
    #                                 column_2='Lipid_Tot_(g)', column_3='Protein_(g)',
    #                                 column_4='Water_(g)', new_column='water:CLPW')
    new_df = insert_main_ftrs_ratio(new_df, main_col='total_carbohydrate', column_1='total_carbohydrate',
                                    column_2='total_fat', column_3='protein', new_column='carbo:CLPW')
    new_df = insert_main_ftrs_ratio(new_df, main_col='total_fat', column_1='total_carbohydrate',
                                    column_2='total_fat', column_3='protein', new_column='total_fat:CLPW')
    new_df = insert_main_ftrs_ratio(new_df, main_col='protein', column_1='total_carbohydrate',
                                    column_2='total_fat', column_3='protein', new_column='protein:CLPW')
    new_df = insert_main_ftrs_ratio(new_df, main_col='dietary_fiber', column_1='total_carbohydrate',
                                    column_2='total_fat', column_3='protein', new_column='fiber:CLPW')
    new_df = insert_main_ftrs_ratio(new_df, main_col='carbo-fiber_(availableCarbo)', column_1='total_carbohydrate',
                                    column_2='total_fat', column_3='protein', new_column='availableCarbos:CLPW')


    write_to_file(new_df, 'final_dataset_with_median.xlsx')

def create_train_and_test():
    split_to_train_test()
