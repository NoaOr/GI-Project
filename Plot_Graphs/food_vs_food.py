import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Plot_Graphs")] + "Excel_files")
    distances = pd.read_excel("Euclidean_distance_without_names_in_rows.xlsx")

    distances[distances > 1500] = 1500
    # print(distances.value_counts())
    # x = distances.values  # returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # distances = pd.DataFrame(x_scaled)

    # plot the heatmap
    # sns.heatmap(distances,
    #            xticklabels="",
    #            yticklabels="", cmap='seismic')

    food_groups = pd.read_excel("GI_USDA_CLEAN_FOOD_GROUPS.xlsx")
    species = food_groups.pop("FdGrp_desc")
    lut = dict(zip(species.unique(), sns.hls_palette(len(set(species.unique())), l=0.5, s=0.8)))
    row_colors = species.map(lut)
    g = sns.clustermap(distances, row_colors=row_colors, figsize=(13, 13))

    for tick_label in g.ax_heatmap.axes.get_yticklabels():
        tick_text = tick_label.get_text()
        species_name = species.loc[int(tick_text)]
        tick_label.set_color(lut[species_name])

    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig("food_vs_food" + '.png')