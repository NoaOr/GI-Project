import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sns.set(style="white")

    os.chdir(os.getcwd()[:os.getcwd().index("Correlation")] + "Excel_files")
    GI_final = pd.read_excel("GI_USDA_clean.xlsx")
    GI_df = pd.DataFrame(GI_final)

    # corr = GI_df.corr()
    # corr.style.background_gradient(cmap='coolwarm')
    # plt.show()

   # Compute the correlation matrix
    corr = GI_df.corr()
    corr.to_csv("correlation.csv")
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    #mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(11, 9))



    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(200, 10, as_cmap=True)


    # sns.clustermap(corr)
    sns.set(font_scale=0.5)
    cmn = sns.clustermap(corr, cmap=cmap, col_cluster=False)

    hm = cmn.ax_heatmap.get_position()
    plt.setp(cmn.ax_heatmap.yaxis.get_majorticklabels(), fontsize=6)
    cmn.ax_heatmap.set_position([hm.x0, hm.y0, hm.width, hm.height])


    # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()