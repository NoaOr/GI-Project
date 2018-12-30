import pandas as pd
import os
import numpy as np
import math
import matplotlib.pyplot as plt



def correlation_matrix(df):
    col_1 = 'GI Value'
    # carbo
    col_2 = 'source table'

    col_1_values = df.loc[:, col_1].values
    z = col_1_values[93]
    for i in range(len(col_1_values)):
        x = col_1_values[i]
        if col_1_values[i] == 'Null':
            col_1_values[i] = 0
        elif math.isnan(x):
            col_1_values[i] = 0

    col_2_values = df.loc[:, col_2].values
    for i in range(len(col_2_values)):
        x = col_2_values[i]
        if col_2_values[i] == 'Null':
            col_2_values[i] = 0
        elif math.isnan(x):
            col_2_values[i] = 0

    corr = np.correlate(col_1_values, col_2_values, "same")
    plt.plot(corr, 'bo')
    plt.show()




if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Correlation")] + "Excel_files")
    GI_final = pd.read_excel("GI_final.xlsx")
    GI_df = pd.DataFrame(GI_final)
    correlation_matrix(GI_df)
