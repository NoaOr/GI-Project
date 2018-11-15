# from HandleTable import *
#
# if __name__ == '__main__':
#     t1 = pd.read_excel('table1.xls')
#     df = pd.DataFrame(t1)
#     col_name = 'Food Description in 1994-96 CSFII'
#     accuracy = 0.01
#     # sentence = 'Milk, lowfat, fluid, 1% milkfat, with added nonfat milk solids, vitamin A and vitamin D'
#     sentence = 'Yogurt, plain, low fat'
#     add_sentence_to_df_by_match(sentence, accuracy, df, col_name)
#     writer = pd.ExcelWriter('table_simple.xlsx', engine='xlsxwriter')
#     # Convert the dataframe to an XlsxWriter Excel object.
#     df.to_excel(writer, sheet_name='Sheet1')
#     # Close the Pandas Excel writer and output the Excel file.
#     writer.save()
