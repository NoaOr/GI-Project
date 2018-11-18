from HandleTable import *

if __name__ == '__main__':
    usda = pd.read_excel('USDA.xlsx')
    t1 = pd.read_excel('table_simple.xlsx')
    usda_df = pd.DataFrame(usda)
    t1_df = pd.DataFrame(t1)
    usda_col_name = 'Long_Desc'
    t1_col_name = 'Food Description in 1994-96 CSFII'
    accuracy = 5
    for i in range(1500, 2500):
        sentence = usda_df.loc[i, usda_col_name]
        add_sentence_to_df_by_match(sentence, accuracy, t1_df, t1_col_name)


    # sentence = 'Milk, lowfat, fluid, 1% milkfat, with added nonfat milk solids, vitamin A and vitamin D'
    # sentence_list = ['']
    # for s in sentence_list:
    #     add_sentence_to_df_by_match(s, accuracy, t1_df, t1_col_name)

    writer = pd.ExcelWriter('table_simple.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    t1_df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


