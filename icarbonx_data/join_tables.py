import pandas as pd

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

    median_df = df.median(skipna=True, numeric_only=True)
    for column in df:
        if column == "food_names":
            continue
        df[column] = df[column].fillna(median_df[column])

    print(df.isnull().sum())

    writer = pd.ExcelWriter('final_dataset_with_median.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

if __name__ == '__main__':
    # join()
    handle_nan()