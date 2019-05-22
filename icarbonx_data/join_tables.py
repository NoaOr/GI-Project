import pandas as pd
import numpy as np
import math
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform



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
    df = df.drop(['food_names', 'height', 'weight', 'above_range', 'BMI', 'age', 'gender',
                  'glucose_tolerance_category','90-percentile_of_2h-iAUC', 'average_carbs_ratio',
                  'average_daily_carbs','average_meals_per_day', 'average_sleep_hours',
                  'average_glucose', 'baseline', 'coefficient_of_variation', 'max_2-hours_iAUC',
                  'median_fasting_glucose_level','median_of_2h-iAUC', 'night_baseline'], axis='columns')

    df = df.replace([-np.inf], 0).dropna(axis=1)

    num_examples = df.shape[0]

    distances = pdist(df.values, metric='euclidean')
    dis_array = squareform(distances)

    dis_df = pd.DataFrame(data = dis_array, index=food_examples, columns=food_examples)

    writer = pd.ExcelWriter('Euclidean_distance_icarbonx.xlsx', engine='xlsxwriter')
    dis_df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

if __name__ == '__main__':
    # join()
    # handle_nan()
    df = pd.read_excel("final_dataset.xlsx")
    get_euclidean_matrix(df)