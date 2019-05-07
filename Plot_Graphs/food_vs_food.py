import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Plot_Graphs")] + "Excel_files")
    gi_usda_df = pd.read_excel("GI_USDA_CLEAN_FOOD_GROUPS.xlsx")

    gi_usda_df = gi_usda_df.drop(['CSFII 1994-96 Food Code',
                     'source table', 'NDB_No', 'reference food & time period', 'serve Size g',
                     'available cerbo hydrate', 'GL per serve', 'GI_2', 'acc', 'match-sent',
                     'GmWt_Desc2', 'GmWt_Desc1', 'Manganese_(mg)',
                     'GmWt_1', 'GmWt_2', 'Panto_Acid_mg)', 'Choline_Tot_ (mg)', 'FdGrp_desc'], axis='columns')

    median_df = gi_usda_df.median(skipna=True, numeric_only=True)
    for column in gi_usda_df:
        if column == "Food Description in 1994-96 CSFII" or column == "FdGrp_desc":
            continue
        gi_usda_df[column] = gi_usda_df[column].fillna(median_df[column])

    gi_usda_df.set_index(['Food Description in 1994-96 CSFII'], inplace=True)

    gi_usda_df = gi_usda_df.transpose()

    writer = pd.ExcelWriter('trans_df.xlsx', engine='xlsxwriter')
    gi_usda_df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

    rows_corr = gi_usda_df.corr()

    #remove first row and first col
    #rows_corr = rows_corr.iloc[1:]
    # writer = pd.ExcelWriter('corr_df.xlsx', engine='xlsxwriter')
    # rows_corr.to_excel(writer, sheet_name='Sheet1')
    # writer.save()

    rows_corr.reset_index(drop=True, inplace=True)
    rows_corr.to_csv("rows_correlation.csv")

    food_groups = pd.read_excel("GI_USDA_CLEAN_FOOD_GROUPS.xlsx")

    species = food_groups.pop("FdGrp_desc")
    lut = dict(zip(species.unique(), sns.hls_palette(len(set(species.unique())), l=0.5, s=0.8)))
    row_colors = species.map(lut)
    g = sns.clustermap(rows_corr, row_colors=row_colors, figsize=(18, 14), metric="correlation")

    # Draw the legend bar for the classes
    for label in species.unique():
        g.ax_col_dendrogram.bar(0, 0, color=lut[label],
                                label=label, linewidth=0)
    g.ax_col_dendrogram.legend(loc="center", ncol=4)

    # Adjust the postion of the main colorbar for the heatmap
    g.cax.set_position([0, 0, .03, .45])
    # g.cax.set_position([.97, .2, .03, .45])



    # g = sns.clustermap(rows_corr, figsize=(13, 13), metric="correlation")


    # for tick_label in g.ax_heatmap.axes.get_yticklabels():
    #     tick_text = tick_label.get_text()
    #     species_name = species.loc[int(tick_text)]
    #     tick_label.set_color(lut[species_name])
    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig("food_vs_food2" + '.png')