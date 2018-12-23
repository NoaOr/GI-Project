import pandas as pd

if __name__ == '__main__':

    t1 = pd.read_excel('Excel_files/GI_final.xlsx')
    t1_df = pd.DataFrame(t1)

    t2 = pd.read_excel('Excel_files/USDA_data.xlsx')
    t2_df = pd.DataFrame(t2)

    col_USDA = t2_df.columns.values
    col_USDA = col_USDA[1:]

    for col_name in col_USDA:
        t1_df[col_name] = ""

    #add row 1 from usda to t1 in row 3(GI)
    for col_name in col_USDA:
        t1_df.loc[3, col_name] = t2_df.at[1, col_name]

    writer = pd.ExcelWriter('ans.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    t1_df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()