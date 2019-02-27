import pandas as pd
import os
import seaborn as sns
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# import numpy as np
# import math
# import matplotlib.pyplot as plt
# import Merge_GI_USDA.Main
#
# def correlation_matrix(df):
#     col_1 = 'GI Value'
#     # carbo
#     col_2 = 'source table'
#
#     col_1_values = df.loc[:, col_1].values
#     z = col_1_values[93]
#     for i in range(len(col_1_values)):
#         x = col_1_values[i]
#         if col_1_values[i] == 'Null':
#             col_1_values[i] = 0
#         elif math.isnan(x):
#             col_1_values[i] = 0
#
#     col_2_values = df.loc[:, col_2].values
#     for i in range(len(col_2_values)):
#         x = col_2_values[i]
#         if col_2_values[i] == 'Null':
#             col_2_values[i] = 0
#         elif math.isnan(x):
#             col_2_values[i] = 0
#
#     corr = np.correlate(col_1_values, col_2_values, "same")
#     plt.plot(corr, 'bo')
#     plt.show()
#
#
#
#
if __name__ == '__main__':
    # os.chdir(os.getcwd()[:os.getcwd().index("Correlation")])
    # GI_final = pd.read_excel("GI_final.xlsx")
    # GI_df = pd.DataFrame(GI_final)
    # var_corr = GI_df.corr()
    # # plot the heatmap and annotation on it
    # sns.heatmap(var_corr, xticklabels=var_corr.columns, yticklabels=var_corr.columns, annot=True)
    # # correlation_matrix(GI_df)

    sns.set(style="white")

    # # Generate a large random dataset
    # rs = np.random.RandomState(33)
    # d = pd.DataFrame(data=rs.normal(size=(100, 26)),
    #                  columns=list(ascii_letters[26:]))
    os.chdir(os.getcwd()[:os.getcwd().index("Correlation")] + "Excel_files")
    GI_final = pd.read_excel("GI_USDA_example.xlsx")
    GI_df = pd.DataFrame(GI_final)
    # Compute the correlation matrix
    corr = GI_df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()