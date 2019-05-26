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
    # improve_ml_code.imporve()
    df = pd.read_excel("final_dataset_with_median.xlsx")
    df = df.replace([-np.inf], 0).dropna(axis=1)


    X = df.drop('2h-iAUC', axis=1, inplace=False)
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


def create_train_and_test():
    split_to_train_test()
