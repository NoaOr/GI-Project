import pandas as pd

if __name__ == '__main__':
    """
    main function - merge the usda tables
    """
    t1 = pd.read_excel('USDA_short&long.xlsx')
    t2 = pd.read_excel('USDA_short_only.xlsx')
    t1_df = pd.DataFrame(t1)
    t2_df = pd.DataFrame(t2)

    merge_df = t1_df.set_index('Shrt_Desc').join(t2_df.set_index('Shrt_Desc'))

    writer = pd.ExcelWriter('USDA_data.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    merge_df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()