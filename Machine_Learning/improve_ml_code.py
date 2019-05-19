import os
import pandas as pd
from Machine_Learning import ml_code
import sys
import numpy as np


def run_on_big_food_group():
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
    df[ratio_column] = ""
    for index, row in df.iterrows():
        carbo_val = df.at[index,'Carbohydrt_(g)']
        col_val = df.at[index, origin_column]
        if col_val == 0:
            col_val = sys.float_info.epsilon
        df.loc[index, ratio_column] = round(carbo_val / col_val, 3)

    return df


def insert_water_ratio(df, ratio_column, column_1, column_2,
                       column_3, column_4, new_column):
    df[new_column] = ""
    for index, row in df.iterrows():
        numerator = df.at[index, ratio_column]
        denominator = df.at[index, column_1] + df.at[index, column_2] + \
                      df.at[index, column_3] + df.at[index, column_4]
        if denominator == 0:
            denominator = sys.float_info.epsilon
        df.loc[index, new_column] = round(numerator / denominator, 3)

    return df


def add_features_to_df(origin_df):
    new_df = origin_df.copy()

    new_df = insert_carbo_ratio(new_df, 'Protein_(g)', 'carbo-protein')
    new_df = insert_carbo_ratio(new_df, 'Lipid_Tot_(g)', 'carbo-lipid')
    new_df = insert_carbo_ratio(new_df, 'Fiber_TD_(g)', 'carbo-fiber_(availableCarbo)')
    new_df = insert_water_ratio(new_df, ratio_column='Water_(g)', column_1='Carbohydrt_(g)',
                                column_2='Lipid_Tot_(g)', column_3='Protein_(g)',
                                column_4='Water_(g)', new_column='water:CLPW')

    # ml_code.learn(new_df, pic_name="with_new_ftrs")
    return new_df

def run_without_fill_sugar(full_df):
    # get the indices of places that sugar doesnt appear in
    gi_usda_df = pd.read_excel("GI_USDA_CLEAN_FOOD_GROUPS.xlsx")
    print(gi_usda_df.shape)
    none_list = gi_usda_df['Sugar_Tot_(g)'].isna()
    none_indices = list(none_list[none_list].index)

    no_sugar_df = full_df.drop(none_indices)
    print(full_df.shape)
    print(no_sugar_df.shape)

    return no_sugar_df
    # ml_code.learn(no_sugar_df, pic_name="no_none_sugar")

def learn_smaller_dataset(df):
    indexes = df.sample(frac=.50).index
    df = df.drop(indexes)
    df.reset_index(drop=True, inplace=True)

    return df
    # ml_code.learn(df, pic_name="smaller_dataset", dir="smaller_dataset")




if __name__ == '__main__':

    if not os.getcwd().__contains__("Excel_files"):
        os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    df = pd.read_excel("GI_USDA_full.xlsx")

    df = run_without_fill_sugar(df)
    df = add_features_to_df(df)
    # run_on_big_food_group()
    # learn_smaller_dataset(df)
    ml_code.learn(df, "features_&_no_sugar", dir='rf_improve')