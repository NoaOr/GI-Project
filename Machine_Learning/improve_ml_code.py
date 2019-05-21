import os
import pandas as pd
import sys
import numpy as np


def run_on_big_food_group():
    """
    run the ml code only on the biggest food groups
    :return:
    """
    if not os.getcwd().__contains__("Excel_files"):
        os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    df = pd.read_excel("GI_USDA_full.xlsx")
    test_df = pd.read_excel("GI_USDA_full.xlsx")

    print(df['FdGrp_desc'].value_counts())
    print(df.shape)

    # remove small food groups from df
    i = 0
    while i < len(df['FdGrp_desc'].value_counts()):
        if df['FdGrp_desc'].value_counts()[i] <= 10:
            fg = df['FdGrp_desc'].value_counts().index[i]
            df.drop(df[(df['FdGrp_desc'] == fg)].index, inplace=True)
            print(df.shape)
        i += 1


    print(df.shape)

    # put in df only biggest food groups

    # biggest_food_group_1 = df['FdGrp_desc'].value_counts().index[0]
    # biggest_food_group_2 = df['FdGrp_desc'].value_counts().index[1]

    # ml_df = df.loc[(df['FdGrp_desc'] == biggest_food_group_1) | (df['FdGrp_desc'] == biggest_food_group_2)]

    # learn(ml_df, "biggest_fg")
    # ml_code.learn(df, "without_small_fg")
    return df


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
        carbo_val = df.at[index,'Carbohydrt_(g)']
        col_val = df.at[index, origin_column]
        denominator = col_val + carbo_val
        df.loc[index, ratio_column] = round(carbo_val / denominator, 3)

    # df.fillna(sys.float_info.epsilon, inplace=True)
    df.fillna(0, inplace=True)
    return df


def insert_main_ftrs_ratio(df, main_col, column_1, column_2,
                           column_3, column_4, new_column):
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
                      df.at[index, column_3] + df.at[index, column_4]
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

    new_df = insert_carbo_ratio(new_df, 'Protein_(g)', 'carbo-protein')
    new_df = insert_carbo_ratio(new_df, 'Lipid_Tot_(g)', 'carbo-lipid')
    new_df = insert_carbo_ratio(new_df, 'Fiber_TD_(g)', 'carbo-fiber_(availableCarbo)')

    new_df = insert_main_ftrs_ratio(new_df, main_col='Water_(g)', column_1='Carbohydrt_(g)',
                                    column_2='Lipid_Tot_(g)', column_3='Protein_(g)',
                                    column_4='Water_(g)', new_column='water:CLPW')
    new_df = insert_main_ftrs_ratio(new_df, main_col='Carbohydrt_(g)', column_1='Carbohydrt_(g)',
                                    column_2='Lipid_Tot_(g)', column_3='Protein_(g)',
                                    column_4='Water_(g)', new_column='carbo:CLPW')
    new_df = insert_main_ftrs_ratio(new_df, main_col='Lipid_Tot_(g)', column_1='Carbohydrt_(g)',
                                    column_2='Lipid_Tot_(g)', column_3='Protein_(g)',
                                    column_4='Water_(g)', new_column='lipidTot:CLPW')
    new_df = insert_main_ftrs_ratio(new_df, main_col='Protein_(g)', column_1='Carbohydrt_(g)',
                                    column_2='Lipid_Tot_(g)', column_3='Protein_(g)',
                                    column_4='Water_(g)', new_column='protein:CLPW')
    new_df = insert_main_ftrs_ratio(new_df, main_col='Fiber_TD_(g)', column_1='Carbohydrt_(g)',
                                    column_2='Lipid_Tot_(g)', column_3='Protein_(g)',
                                    column_4='Water_(g)', new_column='fiber:CLPW')
    new_df = insert_main_ftrs_ratio(new_df, main_col='carbo-fiber_(availableCarbo)', column_1='Carbohydrt_(g)',
                                    column_2='Lipid_Tot_(g)', column_3='Protein_(g)',
                                    column_4='Water_(g)', new_column='availableCarbos:CLPW')


    return new_df

def run_without_fill_sugar(full_df):
    """
    this function removes the blank lines of sugar
    :param full_df:
    :return:
    """
    # get the indices of places that sugar doesnt appear in
    gi_usda_df = pd.read_excel("GI_USDA_CLEAN_FOOD_GROUPS.xlsx")
    print(gi_usda_df.shape)
    none_list = gi_usda_df['Sugar_Tot_(g)'].isna()
    none_indices = list(none_list[none_list].index)

    no_sugar_df = full_df.drop(none_indices)
    print(full_df.shape)
    print(no_sugar_df.shape)

    return no_sugar_df


def learn_smaller_dataset(df):
    """
    executing the ml on smaller data set
    :param df:
    :return:
    """
    indexes = df.sample(frac=.50).index
    df = df.drop(indexes)
    df.reset_index(drop=True, inplace=True)

    return df


def add_ln_features(df):
    """
    This function adds ln features for each regular feature
    :param df:
    :return: new df
    """
    df_copy = df.copy()
    df_copy.replace(0, sys.float_info.epsilon)

    for col in df:

        if col == 'Food Description in 1994-96 CSFII' or col == 'GI Value' \
                or col == 'FdGrp_desc':
            continue
        col_name = col + '_ln'
        df[col_name] = np.log(df_copy[col])
        # print(df[col_name])

    return df


def imporve():
    """
    This function improves the current df
    :return: new df
    """
    if not os.getcwd().__contains__("Excel_files"):
        os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    df = pd.read_excel("GI_USDA_full.xlsx")

    df = run_without_fill_sugar(df)
    df = add_ln_features(df)
    df = add_features_to_df(df)

    df = df.replace([-np.inf], 0).dropna(axis=1)

    writer = pd.ExcelWriter('GI_USDA_IMPROVED.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

    # run_on_big_food_group()
    # learn_smaller_dataset(df)


    return df