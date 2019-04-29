import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sns.set()

    os.chdir(os.getcwd()[:os.getcwd().index("Correlation")] + "Excel_files")
    GI_final = pd.read_excel("GI_USDA_clean.xlsx")
    GI_df = pd.DataFrame(GI_final)

   # Compute the correlation matrix
    corr = GI_df.corr()
    corr.to_csv("correlation.csv")
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(1500, 17, as_cmap=True)
    sns.set(font_scale=0.5)
    cmn = sns.clustermap(corr, cmap="seismic", col_cluster=True,
                         linewidths=0.5, figsize=(15, 15))
    sm = plt.cm.ScalarMappable(cmap="seismic")
    hm = cmn.ax_heatmap.get_position()
    plt.setp(cmn.ax_heatmap.yaxis.get_majorticklabels(), fontsize=6)
    cmn.ax_heatmap.set_position([hm.x0, hm.y0, hm.width, hm.height])
    # plt.show()