import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


def plot_corr(column_name, color_col_name, gi_usda_df, pic_name, title, range=0.005):
    print(gi_usda_df[column_name].value_counts())
    print("-------------------------------------------------")
    max_index = gi_usda_df[column_name].idxmax()
    max_in_col = gi_usda_df[column_name].values[max_index]

    bins_values = np.arange(0, max_in_col, range)
    labels_values = np.arange(0, len(bins_values) - 1, 1)
    gi_usda_df['log_values'] = ""
    gi_usda_df['log_values'] = np.log(gi_usda_df[column_name])
    gi_usda_df['log_values'] += 6
    gi_usda_df[color_col_name] = ""
    gi_usda_df[color_col_name] = pd.cut(gi_usda_df['log_values'], bins=bins_values, labels=labels_values)

    # gi_usda_df[color_col_name] = gi_usda_df[column_name].fillna(max_in_col)
    gi_usda_df = gi_usda_df[pd.notnull(gi_usda_df[color_col_name])]

    color_arr = gi_usda_df[color_col_name]

    x = gi_usda_df['Carbohydrt_(g)']
    y = gi_usda_df['GI Value']
    plt.figure(figsize=(17, 12))

    plt.scatter(x=x, y=y, c=color_arr, cmap='gist_stern', s=75)
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 25,
            }
    plt.title(title, fontdict=font)
    font = {'color': 'black',
            'weight': 'bold',
            'size': 18,
            }
    plt.xlabel("Carbohydrt", fontdict=font)
    plt.ylabel("GI Value", fontdict=font)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    cbar = plt.colorbar()

    # cbar.ax.tick_params(labelsize=15)
    cbar.set_ticks([])
    cbar.set_label(column_name, weight='bold', size=18)

    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig(pic_name + '.png')
