import pandas as pd
import os


def fix_GI_col(df):

    for row in range(df.shape[0]):
        print(row)
        for col in range(df.shape[1]):
            temp = df.loc[row][col]
            if isinstance(temp, str):
                if temp.__contains__('±'):
                    x = temp.split('±')[0]
                    df.iloc[row, col] = x

    writer = pd.ExcelWriter('GI_Src_2_new.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()



if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Merge_GI_src")] + "Excel_files/GI_tables")
    src2 = pd.read_excel("GI_Src_2.xlsx")
    df_src2 = pd.DataFrame(src2)
    fix_GI_col(df_src2)