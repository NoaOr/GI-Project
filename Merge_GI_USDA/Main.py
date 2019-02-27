from Merge_GI_USDA.HandleTable import *
import os
global num_matches

if __name__ == '__main__':
    num_matches = 0
    os.chdir(os.getcwd()[:os.getcwd().index("Merge_GI_USDA")] + "Excel_files")

    usda = pd.read_excel("USDA_data.xlsx")
    GI_table = pd.read_excel("GI_tables/GI_merge.xlsx")

    usda_df = pd.DataFrame(usda)
    GI_df = pd.DataFrame(GI_table)
    usda_col_name = 'Long_Desc'
    GI_col_name = 'Food Description in 1994-96 CSFII'
    accuracy = 15
    print (GI_df.shape[0])
    #for i in range(4500, 5000):

    for i in range(GI_df.shape[0]):
        if i == 3265:
            continue
        GI_desc = GI_df.loc[i, GI_col_name]
        add_sentence_to_df_by_match(GI_desc, accuracy, usda_df, usda_col_name, GI_df, i)


    writer = pd.ExcelWriter('GI_USDA_final.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    GI_df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


