import os
import pandas as pd
from Machine_Learning import ml_code
import sys


def run_on_big_food_group(df):


    biggest_food_group = df['FdGrp_desc'].value_counts().index[0]

    ml_df = df.loc[df['FdGrp_desc'] == biggest_food_group]

    # ML_code.learn(ml_df, "biggest_fg")


def insert_to_ratio_column(df, origin_column, ratio_column):
    df[ratio_column] = ""
    for index, row in df.iterrows():
        carbo_val = df.at[index,'Carbohydrt_(g)']
        col_val = df.at[index, origin_column]
        if col_val == 0:
            col_val = sys.float_info.epsilon
        df.loc[index, ratio_column] = round(carbo_val / col_val, 3)


    # median_df = df.median(skipna=True, numeric_only=True)
    # df[ratio_column] = df[ratio_column].fillna(median_df[ratio_column])
    return df


def add_features_to_df(origin_df):
    new_df = origin_df.copy()

    new_df = insert_to_ratio_column(new_df, 'Protein_(g)', 'carbo-protein')
    new_df = insert_to_ratio_column(new_df, 'Lipid_Tot_(g)', 'carbo-lipid')
    new_df = insert_to_ratio_column(new_df, 'Fiber_TD_(g)', 'carbo-fiber_(availableCarbo)')


    # df1 = new_df.copy()
    # median_df = df1.median(skipna=True, numeric_only=True)
    # for column in df1:
    #     df1[column] = df1[column].fillna(median_df[column])

    ml_code.learn(new_df, pic_name="with_new_ftrs")


if __name__ == '__main__':

    if not os.getcwd().__contains__("Excel_files"):
        os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    df = pd.read_excel("GI_USDA_full.xlsx")



    # median_df = df.median(skipna=True, numeric_only=True)
    # for column in df:
    #     df[column] = df[column].fillna(median_df[column])

    # run_on_big_food_group(df)

    add_features_to_df(df)
