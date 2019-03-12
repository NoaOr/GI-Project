from Merge_GI_USDA.HandleTable import *
import config as cfg
import os
global num_matches


if __name__ == '__main__':
    num_matches = 0
    os.chdir(os.getcwd()[:os.getcwd().index("Merge_GI_USDA")] + "Excel_files")

    usda = pd.read_excel("USDA_data.xlsx")
    GI_table = pd.read_excel("GI_tables/GI_merge.xlsx")

    #usda = pd.read_excel("usda_temp.xlsx")
    #GI_table = pd.read_excel("GI_tables/gi_temp.xlsx")

    usda_df = pd.DataFrame(usda)
    cfg.GI_df_2 = pd.DataFrame(GI_table)
    usda_col_name = 'Long_Desc'
    GI_col_name = 'Food Description in 1994-96 CSFII'
    accuracy = 15
    print (cfg.GI_df_2.shape[0])
    for i in range(0, 200):

    # for i in range(cfg.GI_df_2.shape[0]):
        if i == 3265:
            continue
        GI_desc = cfg.GI_df_2.loc[i, GI_col_name]
        add_sentence_to_df_by_match(GI_desc, accuracy, usda_df, usda_col_name, i)

    writer = pd.ExcelWriter('GI_USDA_final.xlsx', engine='xlsxwriter')
    #writer = pd.ExcelWriter('GI_USDA_final_temp.xlsx', engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    x = cfg.GI_df_2.iloc[2, 4]
    cfg.GI_df_2.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


