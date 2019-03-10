import pandas as pd
import os

def clean_empty_rows(df):


    for i in range(df.shape[0], -1, -1):
        if i == 3265:
            continue
        if str(df.loc[i, 'acc']) == 'nan' or str(df.loc[i, 'NDB_No']) == 'nan':
            df.drop(i, inplace=True)

    return df


def create_df(table_path):
    os.chdir(os.getcwd()[:os.getcwd().index("tools")] + "Excel_files")
    table = pd.read_excel(table_path)
    df = pd.DataFrame(table)
    return df

def write_to_file(df):
    writer = pd.ExcelWriter('GI_USDA_clean.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


if __name__ == '__main__':
    df = create_df("GI_USDA_final.xlsx")
    df_2 = clean_empty_rows(df)
    write_to_file(df_2)