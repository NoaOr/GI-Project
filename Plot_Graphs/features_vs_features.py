import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_corr(gi_usda_df):
    sns.set()

    plt.figure(figsize=(20, 30))

    # Compute the correlation matrix
    corr = gi_usda_df.corr()
    corr.to_csv("correlation.csv")
    # Generate a mask for the upper triangle
    sns.set(font_scale=1.2)
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

    cmn = sns.clustermap(corr,  col_cluster=True,
                         figsize=(17, 13), metric="correlation")
    # font = {'family': 'serif',
    #         'color': 'black',
    #         'weight': 'normal',
    #         'ha': 'center',
    #         'size': 25,
    #         }
    # plt.title('Features vs Features Correlation', fontdict=font)
    cmn.fig.suptitle('Features vs Features Correlation', fontsize=25)
    # cmn.set()
    # sm = plt.cm.ScalarMappable(cmap="seismic")
    # hm = cmn.ax_heatmap.get_position()
    font = {'color': 'black',
            'weight': 'bold',
            'horizontalalignment': 'center',
            'size': 22
            }
    cmn.cax.set_position([0.01, 0.5, .03, .45])


    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig('features_vs_features' + '.png')



@DeprecationWarning
def plot_corr_temp(gi_usda_df):
    sns.set()

    plt.figure(figsize=(25, 20))

    # Compute the correlation matrix
    corr = gi_usda_df.corr()
    corr.to_csv("correlation.csv")
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    sns.set(font_scale=0.5)
    cmn = sns.clustermap(corr, cmap="seismic", col_cluster=True,
                         linewidths=0.5, figsize=(15, 15))
    sm = plt.cm.ScalarMappable(cmap="seismic")
    hm = cmn.ax_heatmap.get_position()
    font = {'color': 'black',
            'weight': 'normal',
            'horizontalalignment': 'right',
            'size': 17,
            }
    plt.title('Features vs Features Correlation', fontdict=font)

    plt.setp(cmn.ax_heatmap.yaxis.get_majorticklabels(), fontsize=6)
    cmn.ax_heatmap.set_position([hm.x0, hm.y0, hm.width, hm.height])

    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig('features_vs_features' + '.png')