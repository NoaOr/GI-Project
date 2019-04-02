from scipy.spatial import distance
import os
import pandas as pd

def get_train_and_test(df):
    print("")


def get_euclidean_matrix(df):
    food_examples = df['Food Description in 1994-96 CSFII']

    df = df.drop(['CSFII 1994-96 Food Code', 'Food Description in 1994-96 CSFII',
                     'source table', 'NDB_No', 'reference food & time period', 'serve Size g',
                     'available cerbo hydrate', 'GL per serve', 'GI_2', 'acc', 'match-sent',
                     'GmWt_Desc2', 'GmWt_Desc1', 'Manganese_(mg)',
                     'GmWt_1', 'GmWt_2', 'Panto_Acid_mg)', 'Choline_Tot_ (mg)'], axis='columns')

    median_df = df.median(skipna=True, numeric_only=True)
    for column in df:
        df[column] = df[column].fillna(median_df[column])

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

    writer = pd.ExcelWriter('Euclidean_distance.xlsx', engine='xlsxwriter')
    dis_df.to_excel(writer, sheet_name='Sheet1')
    writer.save()




if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    df = pd.read_excel("GI_USDA_clean.xlsx")

    get_euclidean_matrix(df)

