import pandas as pd
import numpy as np
import math
from scipy.spatial import distance


def join():
    dataset_df = pd.read_excel("dataset.xlsx")
    statistics_df = pd.read_excel("statistics.xlsx")

    merge_df = pd.merge(dataset_df, statistics_df, on=['patient_identifier'])

    writer = pd.ExcelWriter('final_dataset.xlsx', engine='xlsxwriter')
    merge_df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def handle_nan():
    df = pd.read_excel("final_dataset.xlsx")
    # print(df.isnull().sum())

    df = df[pd.notnull(df['food_names'])]
    median_df = df.median(skipna=True, numeric_only=True)
    for column in df:
        if column == "food_names" or column == "BMI":
            continue
        df[column] = df[column].fillna(median_df[column])

    indices = list(np.where(df['BMI'].isna()))[0]
    for index in indices:
        df.loc[index, 'BMI'] = df.loc[index, 'weight'] / math.pow(df.loc[index, 'height'], 2)

    median_df = df.median(skipna=True, numeric_only=True)
    df['BMI'] = df['BMI'].fillna(median_df['BMI'])


    print(df.isnull().sum())

    writer = pd.ExcelWriter('final_dataset_with_median.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


def get_euclidean_matrix(df):
    food_examples = df['food_names']
    df = df.drop(['food_names'], axis='columns')

    df = df.replace([-np.inf], 0).dropna(axis=1)

    # median_df = df.median(skipna=True, numeric_only=True)
    # for column in df:
    #     df[column] = df[column].fillna(median_df[column])

    num_examples = df.shape[0]

    dis_df = pd.DataFrame(index=food_examples, columns=food_examples)
    dis_df = dis_df.fillna('-')

    for food_index in range(num_examples):
        print(food_index)
        food = df.iloc[food_index,:]
        for compare_food_index in range(num_examples):
            compare_food = df.iloc[compare_food_index,:]
            dis = distance.euclidean(food, compare_food)
            dis_df.iloc[food_index, compare_food_index] = dis

    writer = pd.ExcelWriter('Euclidean_distance_icarbonx.xlsx', engine='xlsxwriter')
    dis_df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

if __name__ == '__main__':
    # join()
    # handle_nan()
    df = pd.read_excel("final_dataset.xlsx")
    get_euclidean_matrix(df)