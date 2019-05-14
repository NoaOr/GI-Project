import os
import pandas as pd
from Plot_Graphs import carbo_vs_GI, food_vs_food, features_vs_features

if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Plot_Graphs")] + "Excel_files")
    gi_usda_df = pd.read_excel("GI_USDA_CLEAN_FOOD_GROUPS.xlsx")

    gi_usda_df = gi_usda_df.drop(['CSFII 1994-96 Food Code',
                                  'source table', 'NDB_No', 'reference food & time period', 'serve Size g',
                                  'available cerbo hydrate', 'GL per serve', 'GI_2', 'acc', 'match-sent',
                                  'GmWt_Desc2', 'GmWt_Desc1', 'Manganese_(mg)',
                                  'GmWt_1', 'GmWt_2', 'Panto_Acid_mg)', 'Choline_Tot_ (mg)', 'FdGrp_desc'],
                                 axis='columns')

    median_df = gi_usda_df.median(skipna=True, numeric_only=True)
    for column in gi_usda_df:
        if column == "Food Description in 1994-96 CSFII" or column == "FdGrp_desc":
            continue
        gi_usda_df[column] = gi_usda_df[column].fillna(median_df[column])


    #############################
    # carbo vs gi by thiamin
    #############################

    # carbo_vs_GI.plot_corr(column_name='Thiamin_(mg)', color_col_name='Thiamin_color',
    #           gi_usda_df=gi_usda_df, pic_name='carbo_vs_gi_by_thiamin',
    #           title='Carbo vs Gi by Thiamin')

    #############################
    # carbo vs gi by selenium
    #############################

    # carbo_vs_GI.plot_corr(column_name='Selenium_(µg)', color_col_name='Selenium_color',
    #           gi_usda_df=gi_usda_df, pic_name='carbo_vs_gi_by_selenium',
    #           title='Carbo vs Gi by Selenium', range=0.5)


    #############################
    # correlation food vs food
    #############################

    food_vs_food.plot_corr(gi_usda_df)


    #############################
    # correlation features vs features
    #############################

    features_vs_features.plot_corr(gi_usda_df)
