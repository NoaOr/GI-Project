import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


def plot_corr(gi_usda_df):
    food_groups = pd.read_excel("GI_USDA_CLEAN_FOOD_GROUPS.xlsx")
    species = food_groups.pop("FdGrp_desc")
    species = pd.Series(species.unique())

    gi_usda_df = pd.read_excel("GI_USDA_CLEAN_FOOD_GROUPS.xlsx")

    gi_usda_df = gi_usda_df.drop(['CSFII 1994-96 Food Code',
                                  'source table', 'NDB_No', 'reference food & time period', 'serve Size g',
                                  'available cerbo hydrate', 'GL per serve', 'GI_2', 'acc', 'match-sent',
                                  'GmWt_Desc2', 'GmWt_Desc1', 'Manganese_(mg)',
                                  'GmWt_1', 'GmWt_2', 'Panto_Acid_mg)', 'Choline_Tot_ (mg)'],
                                 axis='columns')

    median_df = gi_usda_df.median(skipna=True, numeric_only=True)
    for column in gi_usda_df:
        if column == "Food Description in 1994-96 CSFII" or column == "FdGrp_desc":
            continue
        gi_usda_df[column] = gi_usda_df[column].fillna(median_df[column])

    print(gi_usda_df.shape)

    # remove small food groups from df
    i = 0
    while i < len(gi_usda_df['FdGrp_desc'].value_counts()):
        if gi_usda_df['FdGrp_desc'].value_counts()[i] <= 50:
            fg = gi_usda_df['FdGrp_desc'].value_counts().index[i]
            gi_usda_df.drop(gi_usda_df[(gi_usda_df['FdGrp_desc'] == fg)].index, inplace=True)
            # print(gi_usda_df.shape)
            species = species[species != fg]
        i += 1

    groups = gi_usda_df.pop("FdGrp_desc")
    # gi_usda_df = gi_usda_df.drop(['FdGrp_desc'], axis='columns')
    plt.cla()
    plt.clf()

    plt.figure(figsize=(20, 30))
    sns.set(font_scale=1.4)
    gi_usda_df.set_index(['Food Description in 1994-96 CSFII'], inplace=True)
    gi_usda_df = gi_usda_df.transpose()
    print(gi_usda_df.shape)
    rows_corr = gi_usda_df.corr()
    print(rows_corr.shape)

    rows_corr.reset_index(drop=True, inplace=True)

    rows_corr.to_csv("rows_correlation.csv")

    # os.chdir(os.getcwd()[:os.getcwd().index("Graphs & Photos")] + "Excel_files")

    x = species.unique()
    lut = dict(zip(species.unique(),
                   sns.hls_palette(len(set(species.unique())), l=0.5, s=0.8)))
    row_colors = groups.map(lut)
    g = sns.clustermap(rows_corr, row_colors=row_colors,
                       figsize=(18, 14), col_cluster=True,
                       metric="correlation")

    # Draw the legend bar for the classes

    for label in species.unique():
        g.ax_col_dendrogram.bar(0, 0, color=lut[label],
                                label=label, linewidth=0)

    g.ax_col_dendrogram.legend(loc="center", ncol=4, prop={'size': 14})
    g.ax_col_dendrogram.set_position([0.45, 0.7, 0.2, 0.4])
    g.ax_col_dendrogram.set_zorder(1)
    # Adjust the postion of the main colorbar for the heatmap
    g.cax.set_position([0.01, 0.5, .03, .45])

    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig("food_vs_food_food_groups" + '.png')





