import pandas as pd

if __name__ == '__main__':
    t1 = pd.read_excel('Excel_files/tests/t1.xlsx')
    t1_df = pd.DataFrame(t1)

    row_df = t1_df[1:]
    #on= ['c1', 'c2']
    row_df.join(other=row_df,on='c1', how = 'right', sort=False)
    t1_df.loc[1, :] = row_df

    writer = pd.ExcelWriter('ans.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    t1_df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()