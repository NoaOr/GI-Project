import pandas as pd
import os

def clean_empty_rows(table_path):
    os.chdir(os.getcwd()[:os.getcwd().index("tools")] + "Excel_files")
    table = pd.read_excel(table_path)
    df = pd.DataFrame(table)

    for i in range(df.shape[0], 0, -1):
        if i == 3265:
            continue
        if str(df.loc[i, 'acc']) == 'nan':
            df.drop(i, inplace=True)

    writer = pd.ExcelWriter('GI_USDA_clean.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


if __name__ == '__main__':
    clean_empty_rows("GI_USDA_final.xlsx")