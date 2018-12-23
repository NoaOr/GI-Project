from Merge_GI_USDA.HandleTable import *

if __name__ == '__main__':
    usda = pd.read_excel('Excel_files/USDA_data.xlsx')
    GI_table = pd.read_excel('Excel_files/GI_final.xlsx')
    usda_df = pd.DataFrame(usda)
    GI_df = pd.DataFrame(GI_table)
    usda_col_name = 'Long_Desc'
    GI_col_name = 'Food Description in 1994-96 CSFII'
    accuracy = 5


    # to 7794
    for i in range(21):
        USDA_desc = usda_df.loc[i, usda_col_name]

        add_sentence_to_df_by_match(USDA_desc, accuracy, GI_df, GI_col_name, usda_df, i)


    # sentence = 'Milk, lowfat, fluid, 1% milkfat, with added nonfat milk solids, vitamin A and vitamin D'
    # sentence_list = ['']
    # for s in sentence_list:
    #     add_sentence_to_df_by_match(s, accuracy, t1_df, t1_col_name)

    writer = pd.ExcelWriter('Excel_files/ans.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    GI_df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


