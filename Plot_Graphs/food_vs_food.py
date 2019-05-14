import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


def plot_corr(gi_usda_df):
    plt.cla()
    plt.clf()

    plt.figure(figsize=(20, 30))
    sns.set(font_scale=1.4)

    gi_usda_df.set_index(['Food Description in 1994-96 CSFII'], inplace=True)

    gi_usda_df = gi_usda_df.transpose()

    rows_corr = gi_usda_df.corr()

    rows_corr.reset_index(drop=True, inplace=True)
    rows_corr.to_csv("rows_correlation.csv")

    os.chdir(os.getcwd()[:os.getcwd().index("Graphs & Photos")] + "Excel_files")
    food_groups = pd.read_excel("GI_USDA_CLEAN_FOOD_GROUPS.xlsx")

    species = food_groups.pop("FdGrp_desc")
    lut = dict(zip(species.unique(),
                   sns.hls_palette(len(set(species.unique())), l=0.5, s=0.8)))
    row_colors = species.map(lut)
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





