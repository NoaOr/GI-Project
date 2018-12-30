from Merge_GI_USDA.HandleTable import *
import os

if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Merge_GI_USDA")] + "Excel_files")
    #cwd = os.getcwd()
    #cwd = cwd[:cwd.index("Merge_GI_USDA")]
    # usda_loc = cwd + "Excel_files/USDA_data.xlsx"
    # GI_loc = cwd + "Excel_files/GI_final.xlsx"

    usda = pd.read_excel("USDA_data.xlsx")
    GI_table = pd.read_excel("GI_merge.xlsx")

    #usda = pd.read_excel("tests/t1.xlsx")
    #GI_table = pd.read_excel("tests/test1.xlsx")


    usda_df = pd.DataFrame(usda)
    GI_df = pd.DataFrame(GI_table)
    usda_col_name = 'Long_Desc'
    GI_col_name = 'Food Description in 1994-96 CSFII'
    accuracy = 15


    # to 7794
    for i in range(GI_df.shape[0]):
        GI_desc = GI_df.loc[i, GI_col_name]
        add_sentence_to_df_by_match(GI_desc, accuracy, usda_df, usda_col_name, GI_df, i)


    writer = pd.ExcelWriter('GI_USDA_final.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    GI_df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


